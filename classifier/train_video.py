#!/usr/bin/env python3
"""
Train MLXVideoClassifier (model.py): ResNet18 + multi-head self-attention
with CLS-token pooling over a fixed-N frame sample per clip.

Usage:
    python train_video.py
    python train_video.py --epochs 20 --batch-size 4 --n-frames 12
    python train_video.py --resume

Note: MLXVideoClassifier.__call__ already applies softmax internally (see
model.py), so this script uses NLL loss on those probabilities rather than
train.py's raw-logit cross_entropy (which would double-apply softmax).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import wandb
from mlx.utils import tree_flatten

import logsetup
from model import MLXVideoClassifier
from train import CACHE_DIR, FRAMES_DIR, class_weights, grad_global_norm, load_labels, safe_wandb_log
from video_dataset import (batch_iter, load_clips, reset_trunc_stats,
                           trunc_summary)

LITTERLOG_DIR = Path(__file__).resolve().parent
CKPT_ROOT = LITTERLOG_DIR / "checkpoints"
CKPT_DIR = CKPT_ROOT / "video"
BEST_PATH = CKPT_DIR / "best.npz"
LAST_PATH = CKPT_DIR / "last.npz"
STATE_JSON = CKPT_DIR / "train_state.json"
LABELS_PATH = LITTERLOG_DIR / "labels.txt"


def loss_fn(model, X, y, weights):
    """model(X) returns softmaxed probabilities (see model.py), so this is
    NLL loss on those probabilities, not cross-entropy on raw logits."""
    probs = model(X)
    true_probs = mx.take_along_axis(probs, y[:, None], axis=-1).squeeze(-1)
    sample_loss = -mx.log(true_probs + 1e-8)
    return (weights[y] * sample_loss).mean(), probs


def full_val_metrics(model, clips, label2idx, labels, n_frames, batch_size=8):
    n = len(labels)
    conf = np.zeros((n, n), dtype=int)
    all_preds, all_true = [], []

    for X, y in batch_iter(clips, label2idx, n_frames, batch_size, train=False):
        probs = model(X)
        preds = np.array(probs.argmax(axis=-1).tolist())
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
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        per_class[l] = {"precision": prec, "recall": rec, "f1": f1,
                        "acc": conf[i, i] / max(conf[i, :].sum(), 1)}

    macro = {k: float(np.mean([v[k] for v in per_class.values()]))
             for k in ("precision", "recall", "f1", "acc")}
    return conf, per_class, macro, all_preds, all_true


EXTRACT_FPS = 2.0   # frames/sec used by extract_frames.py


def truncate_clips(clips, seconds: float):
    """Copy of the clip list with each clip cut to its first `seconds`.

    Mirrors what the server sees mid-event: Protect exports only the footage
    recorded so far, so inference regularly runs on a partial clip. Evaluating
    only on whole clips hides how the model does in that case.
    """
    keep = max(1, int(round(seconds * EXTRACT_FPS)))
    return [(label, paths[:keep]) for label, paths in clips]


def save_checkpoint(model, epoch, best_val_acc, epochs_since_improve, is_best):
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
    logsetup.install()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-frames", type=int, default=12)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                         help="Checkpoint dir name under litterlog/checkpoints/ (default: video_<n-frames>f)")
    parser.add_argument("--no-color-aug", action="store_true",
                         help="Disable brightness/contrast/color jitter (ablation: color may be the pee/poo signal)")
    parser.add_argument("--early-eval-seconds", type=float, default=30.0,
                         help="also evaluate val truncated to this many seconds "
                              "(0 disables); reported separately from the full-clip eval")
    parser.add_argument("--trunc-prob", type=float, default=0.0,
                         help="probability of truncating a clip to a random prefix "
                              "(matches the partial clips seen at inference)")
    parser.add_argument("--trunc-min", type=float, default=0.25,
                         help="shortest prefix kept, as a fraction of the clip")
    parser.add_argument("--balanced", action="store_true",
                         help="Class-balanced batch sampling with rare-class oversampling")
    parser.add_argument("--no-class-weights", action="store_true",
                         help="Disable loss class weighting (redundant double-correction when --balanced is on)")
    args = parser.parse_args()

    global CKPT_DIR, BEST_PATH, LAST_PATH, STATE_JSON
    suffix = (("_nocoloraug" if args.no_color_aug else "")
              + ("_balanced" if args.balanced else "")
              + (f"_trunc{args.trunc_prob:g}" if args.trunc_prob > 0 else ""))
    ckpt_dir_name = args.ckpt_dir or f"video_{args.n_frames}f{suffix}"
    CKPT_DIR = CKPT_ROOT / ckpt_dir_name
    BEST_PATH = CKPT_DIR / "best.npz"
    LAST_PATH = CKPT_DIR / "last.npz"
    STATE_JSON = CKPT_DIR / "train_state.json"
    print(f"[+] Checkpoints: {CKPT_DIR}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print("[+] Loading clip-level data (grouped from cached frames) …")
    labels = load_labels(FRAMES_DIR)
    n_classes = len(labels)
    label2idx = {l: i for i, l in enumerate(labels)}

    train_clips = load_clips("train")
    val_clips = load_clips("val")
    test_clips = load_clips("test")
    train_t = [label2idx[label] for label, _ in train_clips]

    print(f"    Classes ({n_classes}): {labels}")
    print(f"    Train: {len(train_clips)} clips  Val: {len(val_clips)} clips  Test: {len(test_clips)} clips")
    for name, clips in (("Train", train_clips), ("Val", val_clips), ("Test", test_clips)):
        counts = collections.Counter(label2idx[l] for l, _ in clips)
        print(f"    {name} class distribution: " + "  ".join(f"{labels[i]}={counts[i]}" for i in range(n_classes)))

    try:
        wandb.init(
            project="litterlog_video",
            config=vars(args) | {
                "model": "resnet18+attention+cls",
                "n_classes": n_classes,
                "train_clips": len(train_clips),
                "val_clips": len(val_clips),
                "test_clips": len(test_clips),
            },
            resume="allow",
        )
    except Exception as e:
        print(f"[!] wandb.init failed, continuing without W&B logging: {e}")
        wandb.init(mode="disabled")

    print("\n[+] Building MLXVideoClassifier …")
    model = MLXVideoClassifier(
        num_classes=n_classes,
        num_frames_per_video=args.n_frames,
        dim=args.dim,
        n_heads=args.n_heads,
    )
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()) if isinstance(p, mx.array))
    print(f"[+] Model built  ({n_params/1e6:.1f}M params)")
    try:
        wandb.config.update({"n_params": n_params})
    except Exception:
        pass

    if args.balanced:
        # balanced epoch = largest class covered once, others oversampled to match
        class_counts = collections.Counter(train_t)
        epoch_clips = max(class_counts.values()) * n_classes
    else:
        epoch_clips = len(train_clips)
    n_train_batches = math.ceil(epoch_clips / args.batch_size)
    total_steps = n_train_batches * args.epochs

    warmup = optim.linear_schedule(0.0, args.lr, args.warmup_steps)
    decay = optim.cosine_decay(args.lr, max(total_steps - args.warmup_steps, 1))
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

    cw = mx.ones(n_classes) if args.no_class_weights else class_weights(train_t, n_classes)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    LABELS_PATH.write_text("\n".join(labels))
    global_step = (start_epoch - 1) * n_train_batches

    # Accumulated per-class val history for grouped line_series panels.
    # wandb.plot.line_series plots whole arrays, so history must persist
    # across epochs (unlike scalar metrics, which W&B accumulates itself).
    acc_history: dict[str, list[float]] = collections.defaultdict(list)
    epoch_history: list[int] = []

    # class pairs that get their own combined panel (only if both exist)
    PAIRED_PANELS = [("pee", "poo"), ("cat1", "cat2")]

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        reset_trunc_stats()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0

        for X, y in batch_iter(train_clips, label2idx, args.n_frames, args.batch_size, train=True,
                               color_aug=not args.no_color_aug, balanced=args.balanced,
                               trunc_prob=args.trunc_prob, trunc_min=args.trunc_min):
            (loss, probs), grads = loss_and_grad(model, X, y, cw)
            gnorm = grad_global_norm(grads)
            if args.grad_clip > 0:
                grads, _ = optim.clip_grad_norm(grads, args.grad_clip)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            batch_loss = loss.item()
            batch_acc = float((probs.argmax(axis=-1) == y).mean().item())
            train_loss += batch_loss
            train_acc += batch_acc
            n_batches += 1
            global_step += 1

            safe_wandb_log({
                "step": global_step,
                "train/step_loss": batch_loss,
                "train/step_acc": batch_acc,
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
        for X, y in batch_iter(val_clips, label2idx, args.n_frames, args.batch_size, train=False):
            val_loss += loss_fn(model, X, y, cw)[0].item()
            val_batches += 1

        conf, per_class, macro, preds, true = full_val_metrics(
            model, val_clips, label2idx, labels, args.n_frames, args.batch_size)

        # Same weights, same val clips, but truncated — reported separately so a
        # gain on whole clips can't mask a regression on partial ones.
        early_macro = early_per_class = None
        if args.early_eval_seconds > 0:
            _, early_per_class, early_macro, _, _ = full_val_metrics(
                model, truncate_clips(val_clips, args.early_eval_seconds),
                label2idx, labels, args.n_frames, args.batch_size)

        tl = train_loss / max(n_batches, 1)
        ta = train_acc / max(n_batches, 1)
        vl = val_loss / max(val_batches, 1)
        elapsed = time.time() - t0
        remaining = (args.epochs - epoch) * elapsed

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"loss={tl:.3f} acc={ta:.3f}  "
              f"val_loss={vl:.3f} val_acc={macro['acc']:.3f}  "
              f"({elapsed:.0f}s, ETA {remaining/60:.0f}m)")
        print("  full  per-class: " + "  ".join(
            f"{l} acc={v['acc']:.2f} f1={v['f1']:.2f}" for l, v in per_class.items()))
        if early_macro:
            print(f"  early ({args.early_eval_seconds:.0f}s): "
                  f"acc={early_macro['acc']:.3f} f1={early_macro['f1']:.3f}   " +
                  "  ".join(f"{l}={v['acc']:.2f}" for l, v in early_per_class.items()))
        ts = trunc_summary()
        if ts["truncated"]:
            print(f"  train aug: {ts['truncated']}/{ts['clips']} clips truncated "
                  f"({ts['rate']:.0%}), mean kept {ts['mean_kept']:.0%}")

        # ── Checkpoint FIRST — a W&B hiccup must never lose a completed epoch ──
        is_best = macro["acc"] > best_val_acc
        if is_best:
            best_val_acc = macro["acc"]
            epochs_since_improve = 0
            print(f"  ↑ new best (val_acc={best_val_acc:.3f})")
        else:
            epochs_since_improve += 1

        save_checkpoint(model, epoch, best_val_acc, epochs_since_improve, is_best)
        print(f"  checkpoint saved (epoch {epoch}, last.npz{' + best.npz' if is_best else ''})")

        try:
            conf_table = wandb.plot.confusion_matrix(y_true=true, preds=preds, class_names=labels)
        except Exception as e:
            print(f"  [!] wandb confusion matrix failed (ignored): {e}")
            conf_table = None

        ts = trunc_summary()
        epoch_log: dict = {
            "epoch": epoch,
            "train/trunc_rate": ts["rate"],
            "train/trunc_clips": ts["truncated"],
            "train/trunc_mean_kept": ts["mean_kept"],
            "train/loss": tl,
            "train/acc": ta,
            "val/loss": vl,
            "val/acc": macro["acc"],
            "val/precision_macro": macro["precision"],
            "val/recall_macro": macro["recall"],
            "val/f1_macro": macro["f1"],
            "train/epoch_time_s": elapsed,
            "train/eta_min": remaining / 60,
        }
        if conf_table is not None:
            epoch_log["val/confusion_matrix"] = conf_table
        if early_macro:
            epoch_log["val_early/acc"] = early_macro["acc"]
            epoch_log["val_early/f1_macro"] = early_macro["f1"]
            epoch_log["val_early/precision_macro"] = early_macro["precision"]
            for l, v in early_per_class.items():
                epoch_log[f"val_early/class/{l}/acc"] = v["acc"]
                epoch_log[f"val_early/class/{l}/f1"] = v["f1"]

        for l, v in per_class.items():
            epoch_log[f"val/class/{l}/acc"] = v["acc"]
            epoch_log[f"val/class/{l}/precision"] = v["precision"]
            epoch_log[f"val/class/{l}/recall"] = v["recall"]
            epoch_log[f"val/class/{l}/f1"] = v["f1"]

        # Grouped panels: both classes of a confusable pair on one chart, so
        # anti-phase oscillation (one spikes while the other drops) is visible
        # without manually building a panel in the W&B UI.
        epoch_history.append(epoch)
        for l, v in per_class.items():
            acc_history[l].append(v["acc"])
        for a, b in PAIRED_PANELS:
            if a in acc_history and b in acc_history:
                try:
                    epoch_log[f"val/pair/{a}_vs_{b}"] = wandb.plot.line_series(
                        xs=epoch_history,
                        ys=[acc_history[a], acc_history[b]],
                        keys=[a, b],
                        title=f"val acc: {a} vs {b}",
                        xname="epoch",
                    )
                except Exception as e:
                    print(f"  [!] wandb line_series for {a}/{b} failed (ignored): {e}")

        safe_wandb_log(epoch_log)

        if is_best:
            try:
                wandb.summary["best_val_acc"] = best_val_acc
                wandb.summary["best_val_f1_macro"] = macro["f1"]
                wandb.summary["best_epoch"] = epoch
            except Exception as e:
                print(f"  [!] wandb.summary update failed (ignored): {e}")

        if args.patience > 0 and epochs_since_improve >= args.patience:
            print(f"\n[+] Early stopping: no improvement for {args.patience} epochs")
            break

    print(f"\n[+] Best val_acc: {best_val_acc:.3f}")
    print(f"[+] Best model: {BEST_PATH}")
    print(f"[+] Last model: {LAST_PATH}")

    if test_clips:
        print(f"\n[+] Evaluating best checkpoint on held-out test set ({len(test_clips)} clips) …")
        model.load_weights(str(BEST_PATH))
        model.eval()
        _, test_per_class, test_macro, _, _ = full_val_metrics(
            model, test_clips, label2idx, labels, args.n_frames, args.batch_size)
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

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
