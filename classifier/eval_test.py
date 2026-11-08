#!/usr/bin/env python3
"""
Evaluate a saved checkpoint on the held-out TEST split.

Standalone so it can be run against a checkpoint while training is still in
progress (reads best.npz, doesn't touch the running job).

Usage:
    python eval_test.py --ckpt-dir video_24f_balanced --n-frames 24
    python eval_test.py --ckpt-dir ... --split val     # sanity-check vs training log
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import MLXVideoClassifier
from train import FRAMES_DIR, load_labels
from train_video import full_val_metrics
from video_dataset import load_clips

LITTERLOG_DIR = Path(__file__).resolve().parent
CKPT_ROOT = LITTERLOG_DIR / "checkpoints"


def resolve_ckpt_dir(name: str) -> Path:
    """Checkpoints live under litterlog/checkpoints/<name>. Older runs used
    litterlog/checkpoints_<name> at the top level, so fall back to that if a
    caller (or an old command line) still refers to one."""
    p = CKPT_ROOT / name
    if p.exists():
        return p
    legacy = LITTERLOG_DIR / (name if name.startswith("checkpoints") else f"checkpoints_{name}")
    return legacy if legacy.exists() else p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="Checkpoint dir name under litterlog/ (e.g. video_24f_balanced)")
    parser.add_argument("--ckpt", type=str, default="best.npz", help="best.npz or last.npz")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--n-frames", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    args = parser.parse_args()

    ckpt_path = resolve_ckpt_dir(args.ckpt_dir) / args.ckpt
    if not ckpt_path.exists():
        raise SystemExit(f"[!] {ckpt_path} not found")

    labels = load_labels(FRAMES_DIR)
    label2idx = {l: i for i, l in enumerate(labels)}
    clips = load_clips(args.split)
    counts = collections.Counter(l for l, _ in clips)

    print(f"[+] Checkpoint: {ckpt_path}")
    print(f"[+] Split: {args.split} — {len(clips)} clips  " +
          "  ".join(f"{l}={counts[l]}" for l in labels))
    print(f"[+] n_frames={args.n_frames}  classes={labels}\n")

    model = MLXVideoClassifier(
        num_classes=len(labels),
        num_frames_per_video=args.n_frames,
        dim=args.dim,
        n_heads=args.n_heads,
    )
    model.load_weights(str(ckpt_path))
    mx.eval(model.parameters())
    model.eval()

    conf, per_class, macro, preds, true = full_val_metrics(
        model, clips, label2idx, labels, args.n_frames, args.batch_size)

    print(f"[+] {args.split}_acc={macro['acc']:.4f}  f1_macro={macro['f1']:.4f}  "
          f"precision_macro={macro['precision']:.4f}  recall_macro={macro['recall']:.4f}\n")

    print("    per-class:")
    for l, v in per_class.items():
        print(f"      {l:6s} acc={v['acc']:.3f}  f1={v['f1']:.3f}  "
              f"prec={v['precision']:.3f}  rec={v['recall']:.3f}")

    print("\n    confusion matrix (rows=true, cols=pred):")
    header = "            " + "".join(f"{l:>7s}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row = "".join(f"{conf[i, j]:7d}" for j in range(len(labels)))
        print(f"      {l:6s}{row}")


if __name__ == "__main__":
    main()
