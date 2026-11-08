#!/usr/bin/env python3
"""
Fine-tune pretrained ResNet18 for cat/waste classification.

Usage:
    python train.py
    python train.py --epochs 20 --batch-size 32 --lr 3e-4
    python train.py --resume                 # continue from last checkpoint
    python train.py --patience 5              # early stop after 5 epochs no improvement
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from mlx.utils import tree_flatten
from PIL import Image, ImageEnhance, ImageOps
from mlxim.model import create_model

LITTERLOG_DIR = Path(__file__).resolve().parents[2] / "classifier"
FRAMES_DIR = LITTERLOG_DIR / "frames"
CACHE_DIR = LITTERLOG_DIR / "frames_cache"
SPLIT_PATH = LITTERLOG_DIR / "split.json"
CKPT_DIR = LITTERLOG_DIR / "checkpoints"
BEST_PATH = CKPT_DIR / "best.npz"
LAST_PATH = CKPT_DIR / "last.npz"
OPT_STATE_PATH = CKPT_DIR / "optimizer_state.npz"
STATE_JSON = CKPT_DIR / "train_state.json"
LABELS_PATH = LITTERLOG_DIR / "labels.txt"
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_FRAMES_PER_VIDEO = 12

# ── Data ──────────────────────────────────────────────────────────────────────

def augment(img: Image.Image, train: bool, color_aug: bool = True) -> np.ndarray:
    img = img.resize((IMG_SIZE + 32, IMG_SIZE + 32), Image.BILINEAR)
    if train:
        # No mirror flip: camera is fixed, scene isn't left-right symmetric
        # (e.g. mesh grate only on one side) — inference frames are never mirrored.
        if color_aug:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.75, 1.25))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        x0 = random.randint(0, 32)
        y0 = random.randint(0, 32)
        img = img.crop((x0, y0, x0 + IMG_SIZE, y0 + IMG_SIZE))
    else:
        img = img.crop((16, 16, 16 + IMG_SIZE, 16 + IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def load_labels(frames_dir: Path) -> list[str]:
    return sorted(d.name for d in frames_dir.iterdir() if d.is_dir())


def load_frozen_split(split_path: Path, frames_dir: Path, labels: list[str]) -> dict[str, tuple[list[Path], list[int]]]:
    """Load the clip-level train/val/test split produced by split_dataset.py.
    Splitting at the frame level leaks near-duplicate frames from the same
    clip across train/val/test, so the split is frozen and computed separately."""
    if not split_path.exists():
        raise SystemExit(
            f"[!] {split_path} not found. Run `python split_dataset.py` first "
            f"to create a frozen, leak-free train/val/test split."
        )
    label2idx = {l: i for i, l in enumerate(labels)}
    raw = json.loads(split_path.read_text())

    out = {}
    for split_name in ("train", "val", "test"):
        rel_paths = raw.get(split_name, [])
        paths = [frames_dir / rp for rp in rel_paths]
        targets = [label2idx[Path(rp).parent.name] for rp in rel_paths]
        out[split_name] = (paths, targets)
    return out


def resolve_cached_paths(paths: list[Path], frames_dir: Path, cache_dir: Path) -> list[Path]:
    """Prefer pre-resized cache images if available, else fall back to originals."""
    resolved = []
    for p in paths:
        cached = cache_dir / p.relative_to(frames_dir)
        resolved.append(cached if cached.exists() else p)
    return resolved


def _load_one(path: Path, train: bool) -> np.ndarray | None:
    try:
        img = Image.open(path).convert("RGB")
        return augment(img, train)
    except Exception:
        return None


_POOL = ThreadPoolExecutor(max_workers=4)


def batch_iter(paths, targets, batch_size, train=True):
    idx = list(range(len(paths)))
    if train:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        futures = [_POOL.submit(_load_one, paths[i], train) for i in batch]
        imgs, lbls = [], []
        for i, future in zip(batch, futures):
            arr = future.result()
            if arr is not None:
                imgs.append(arr)
                lbls.append(targets[i])
        if imgs:
            yield mx.array(np.stack(imgs)), mx.array(lbls)


def class_weights(targets: list[int], n_classes: int) -> mx.array:
    counts = np.bincount(targets, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    return mx.array(weights / weights.sum() * n_classes)


# ── Metrics ───────────────────────────────────────────────────────────────────

def full_val_metrics(model, paths, targets, labels, batch_size=64):
    n = len(labels)
    conf = np.zeros((n, n), dtype=int)
    all_preds, all_true = [], []

    for X, y in batch_iter(paths, targets, batch_size, train=False):
        logits = model(X)
        probs = np.array(nn.softmax(logits, axis=-1).tolist())
        preds = probs.argmax(axis=-1)
        y_np = np.array(y.tolist())
        for p, t in zip(preds, y_np):
            conf[t, p] += 1
        all_preds.extend(preds.tolist())
        all_true.extend(y_np.tolist())

    per_class = {}
    for i, l in enumerate(labels):
        tp = conf[i, i]
        fp = conf[:, i].sum() - tp
        fn = conf[i, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1,
                        "acc": conf[i, i] / max(conf[i, :].sum(), 1)}

    macro = {k: float(np.mean([v[k] for v in per_class.values()]))
             for k in ("precision", "recall", "f1", "acc")}

    return conf, per_class, macro, all_preds, all_true


def sample_val_images(paths, targets, labels, preds, n_per_class=4):
    by_class_correct = collections.defaultdict(list)
    by_class_wrong   = collections.defaultdict(list)
    for i, (p, t) in enumerate(zip(preds, targets)):
        (by_class_correct if p == t else by_class_wrong)[t].append(i)

    images = []
    for cls_idx, cls_name in enumerate(labels):
        for i in by_class_correct[cls_idx][:n_per_class]:
            img = Image.open(paths[i]).convert("RGB").resize((224, 224))
            images.append(wandb.Image(img, caption=f"correct {cls_name}"))
        for i in by_class_wrong[cls_idx][:n_per_class]:
            img = Image.open(paths[i]).convert("RGB").resize((224, 224))
            pred_name = labels[preds[i]]
            images.append(wandb.Image(img, caption=f"WRONG true={cls_name} pred={pred_name}"))
    return images


def grad_global_norm(grads) -> float:
    leaves = [g for _, g in tree_flatten(grads) if isinstance(g, mx.array)]
    sq_sum = sum(float((g * g).sum().item()) for g in leaves)
    return math.sqrt(sq_sum)


# ── Training ──────────────────────────────────────────────────────────────────

def loss_fn(model, X, y, weights):
    """Returns (loss, logits) — logits as aux output avoids a second forward pass."""
    logits = model(X)
    sample_loss = nn.losses.cross_entropy(logits, y)
    return (weights[y] * sample_loss).mean(), logits


def safe_wandb_log(payload: dict) -> None:
    try:
        wandb.log(payload)
    except Exception as e:
        print(f"  [!] wandb.log failed (ignored): {e}")


def save_checkpoint(model, optimizer, epoch, best_val_acc, epochs_since_improve, is_best):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(LAST_PATH))
    if is_best:
        model.save_weights(str(BEST_PATH))
    STATE_JSON.write_text(json.dumps({
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "epochs_since_improve": epochs_since_improve,
    }))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="max grad global norm, 0 disables")
    parser.add_argument("--patience", type=int, default=8, help="early stop after N epochs with no val improvement, 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/last.npz")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("[+] Loading frozen clip-level split …")
    # TODO: load video and extract NUM_FRAMES_PER_VIDEO
    labels = load_labels(FRAMES_DIR)
    n_classes = len(labels)
    split = load_frozen_split(SPLIT_PATH, FRAMES_DIR, labels)
    (train_p, train_t) = split["train"]
    (val_p, val_t) = split["val"]
    (test_p, test_t) = split["test"]

    if CACHE_DIR.exists():
        train_p = resolve_cached_paths(train_p, FRAMES_DIR, CACHE_DIR)
        val_p   = resolve_cached_paths(val_p, FRAMES_DIR, CACHE_DIR)
        test_p  = resolve_cached_paths(test_p, FRAMES_DIR, CACHE_DIR)
        print(f"[+] Using resized cache ({CACHE_DIR})")
    else:
        print(f"[!] No cache at {CACHE_DIR} — using frames/ directly "
              "(extract_frames.py already writes them at 256px, so this is fine)")

    print(f"    Classes ({n_classes}): {labels}")
    print(f"\n    Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")
    for name, t in (("Train", train_t), ("Val", val_t), ("Test", test_t)):
        counts = collections.Counter(t)
        print(f"    {name} class distribution: " + "  ".join(f"{labels[i]}={counts[i]}" for i in range(n_classes)))

    try:
        wandb.init(
            project="litterlog_video",
            config=vars(args) | {
                "model": "resnet18",
                "n_classes": n_classes,
                "train_size": len(train_p),
                "val_size": len(val_p),
                "test_size": len(test_p),
                "train_class_counts": {labels[i]: c for i, c in collections.Counter(train_t).items()},
            },
            resume="allow",
        )
    except Exception as e:
        print(f"[!] wandb.init failed, continuing without W&B logging: {e}")
        wandb.init(mode="disabled")

    print("\n[+] Loading pretrained ResNet18 …")
    model = create_model("resnet18", weights=True, num_classes=1000)
    
    # attention over the activations of each frame
    # learn a weight over each frame
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()) if isinstance(p, mx.array))
    print(f"[+] Model loaded  ({n_params/1e6:.1f}M params)")
    try:
        wandb.config.update({"n_params": n_params})
    except Exception:
        pass

    n_train_batches = math.ceil(len(train_p) / args.batch_size)
    total_steps = n_train_batches * args.epochs

    warmup = optim.linear_schedule(0.0, args.lr, args.warmup_steps)
    decay  = optim.cosine_decay(args.lr, max(total_steps - args.warmup_steps, 1))
    lr_schedule = optim.join_schedules([warmup, decay], [args.warmup_steps])
    optimizer = optim.Adam(learning_rate=lr_schedule)

    start_epoch = 1
    best_val_acc = 0.0
    epochs_since_improve = 0

    if args.resume and LAST_PATH.exists():
        print(f"[+] Resuming from {LAST_PATH}")
        model.load_weights(str(LAST_PATH))
        if STATE_JSON.exists():
            state = json.loads(STATE_JSON.read_text())
            start_epoch = state["epoch"] + 1
            best_val_acc = state["best_val_acc"]
            epochs_since_improve = state["epochs_since_improve"]
            print(f"    epoch={start_epoch}  best_val_acc={best_val_acc:.3f}")

    cw = class_weights(train_t, n_classes)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    LABELS_PATH.write_text("\n".join(labels))
    global_step = (start_epoch - 1) * n_train_batches

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for X, y in batch_iter(train_p, train_t, args.batch_size, train=True):
            (loss, logits), grads = loss_and_grad(model, X, y, cw)
            gnorm = grad_global_norm(grads)
            if args.grad_clip > 0:
                grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            batch_loss = loss.item()
            batch_acc = float((logits.argmax(axis=-1) == y).mean().item())
            train_loss += batch_loss
            train_acc  += batch_acc
            n_batches  += 1
            global_step += 1

            safe_wandb_log({
                "step": global_step,
                "train/step_loss": batch_loss,
                "train/step_acc":  batch_acc,
                "train/grad_norm": gnorm,
                "train/lr": float(optimizer.learning_rate.item()),
            })
            print(f"\r  epoch {epoch}/{args.epochs}  "
                  f"batch {n_batches}/{n_train_batches}  "
                  f"loss={train_loss/n_batches:.3f}  "
                  f"acc={train_acc/n_batches:.3f}", end="", flush=True)
        print()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_batches = 0.0, 0
        for X, y in batch_iter(val_p, val_t, args.batch_size, train=False):
            val_loss += loss_fn(model, X, y, cw)[0].item()
            val_batches += 1

        conf, per_class, macro, preds, true = full_val_metrics(model, val_p, val_t, labels, args.batch_size)

        tl = train_loss / max(n_batches, 1)
        ta = train_acc  / max(n_batches, 1)
        vl = val_loss   / max(val_batches, 1)
        elapsed = time.time() - t0
        remaining = (args.epochs - epoch) * elapsed

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={tl:.3f} acc={ta:.3f}  "
              f"val_loss={vl:.3f} val_acc={macro['acc']:.3f}  "
              f"({elapsed:.0f}s, ETA {remaining/60:.0f}m)")
        print("  per-class: " + "  ".join(
            f"{l} acc={v['acc']:.2f} f1={v['f1']:.2f}" for l, v in per_class.items()))

        # ── Checkpoint FIRST — never lose an epoch of training to a wandb hiccup ──
        is_best = macro["acc"] > best_val_acc
        if is_best:
            best_val_acc = macro["acc"]
            epochs_since_improve = 0
            print(f"  ↑ new best (val_acc={best_val_acc:.3f})")
        else:
            epochs_since_improve += 1

        save_checkpoint(model, optimizer, epoch, best_val_acc, epochs_since_improve, is_best)
        print(f"  checkpoint saved (epoch {epoch}, last.npz{' + best.npz' if is_best else ''})")

        # ── W&B logging — best effort, failures must not crash training ──────────
        try:
            conf_table = wandb.plot.confusion_matrix(y_true=true, preds=preds, class_names=labels)
        except Exception as e:
            print(f"  [!] wandb confusion matrix failed (ignored): {e}")
            conf_table = None

        log_images = epoch == start_epoch or epoch % 5 == 0 or epoch == args.epochs
        sample_imgs = sample_val_images(val_p, val_t, labels, preds) if log_images else []

        epoch_log: dict = {
            "epoch": epoch,
            "train/loss": tl,
            "train/acc":  ta,
            "val/loss":   vl,
            "val/acc":    macro["acc"],
            "val/precision_macro": macro["precision"],
            "val/recall_macro":    macro["recall"],
            "val/f1_macro":        macro["f1"],
            "train/epoch_time_s": elapsed,
            "train/eta_min": remaining / 60,
        }
        if conf_table is not None:
            epoch_log["val/confusion_matrix"] = conf_table
        for l, v in per_class.items():
            epoch_log[f"val/class/{l}/acc"]       = v["acc"]
            epoch_log[f"val/class/{l}/precision"] = v["precision"]
            epoch_log[f"val/class/{l}/recall"]    = v["recall"]
            epoch_log[f"val/class/{l}/f1"]        = v["f1"]
        if sample_imgs:
            epoch_log["val/sample_predictions"] = sample_imgs

        safe_wandb_log(epoch_log)

        if is_best:
            try:
                wandb.summary["best_val_acc"]      = best_val_acc
                wandb.summary["best_val_f1_macro"] = macro["f1"]
                wandb.summary["best_epoch"]        = epoch
            except Exception as e:
                print(f"  [!] wandb.summary update failed (ignored): {e}")

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"\n[+] Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n[+] Best val_acc: {best_val_acc:.3f}")
    print(f"[+] Best model: {BEST_PATH}")
    print(f"[+] Last model: {LAST_PATH}")

    # ── Final held-out test evaluation (never touched during training/tuning) ──
    if test_p:
        print(f"\n[+] Evaluating best checkpoint on held-out test set ({len(test_p)} frames) …")
        model.load_weights(str(BEST_PATH))
        model.eval()
        _, test_per_class, test_macro, test_preds, test_true = full_val_metrics(
            model, test_p, test_t, labels, args.batch_size)
        print(f"    test_acc={test_macro['acc']:.3f}  test_f1_macro={test_macro['f1']:.3f}")
        print("    per-class: " + "  ".join(
            f"{l} acc={v['acc']:.2f} f1={v['f1']:.2f}" for l, v in test_per_class.items()))
        try:
            wandb.summary["test_acc"] = test_macro["acc"]
            wandb.summary["test_f1_macro"] = test_macro["f1"]
            for l, v in test_per_class.items():
                wandb.summary[f"test_class/{l}/acc"] = v["acc"]
                wandb.summary[f"test_class/{l}/f1"] = v["f1"]
        except Exception as e:
            print(f"  [!] wandb test summary failed (ignored): {e}")
    else:
        print("\n[!] No test split found — skipping final test evaluation")

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
