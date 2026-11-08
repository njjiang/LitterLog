#!/usr/bin/env python3
"""
Measure accuracy when only the FIRST N seconds of each clip are available —
i.e. what early/streaming classification would see mid-event.

Frames were extracted at 2 fps, so N seconds == first 2N frames. Everything
else (24-frame even sampling, center crop) matches the eval path exactly, so
the only variable is how much of the clip exists yet.

Expectation to test: cat identity should survive truncation (the cat is
visible immediately), waste should not (pee/poo only become visible after the
Litter-Robot sifts, ~50s in) — if so, early notification is viable for cat ID
only, and the waste call has to wait.

Usage:
    python eval_truncated.py
    python eval_truncated.py --seconds 5 10 20 30 45 60 --split test
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import MLXVideoClassifier
from train import FRAMES_DIR, load_labels
from video_dataset import load_clips, sample_clip_frames

LITTERLOG_DIR = Path(__file__).resolve().parent
CKPT_ROOT = LITTERLOG_DIR / "checkpoints"
EXTRACT_FPS = 2.0   # extract_frames.py --fps 2


def resolve_ckpt_dir(name: str) -> Path:
    """Checkpoints live under litterlog/checkpoints/<name>. Older runs used
    litterlog/checkpoints_<name> at the top level, so fall back to that if a
    caller (or an old command line) still refers to one."""
    p = CKPT_ROOT / name
    if p.exists():
        return p
    legacy = LITTERLOG_DIR / (name if name.startswith("checkpoints") else f"checkpoints_{name}")
    return legacy if legacy.exists() else p


def evaluate(model, clips, label2idx, labels, n_frames, seconds: float | None):
    """seconds=None -> full clip (baseline)."""
    n = len(labels)
    conf = np.zeros((n, n), dtype=int)

    for label, paths in clips:
        if seconds is not None:
            keep = max(1, int(round(seconds * EXTRACT_FPS)))
            paths = paths[:keep]
        frames = sample_clip_frames(paths, n_frames, train=False)
        probs = model(mx.array(frames[None]))
        pred = int(np.array(probs.tolist())[0].argmax())
        conf[label2idx[label], pred] += 1

    per_class = {}
    for i, l in enumerate(labels):
        total = conf[i, :].sum()
        per_class[l] = conf[i, i] / max(total, 1)
    overall = np.trace(conf) / max(conf.sum(), 1)
    macro = float(np.mean(list(per_class.values())))
    return overall, macro, per_class, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="video_24f_balanced")
    parser.add_argument("--ckpt", default="best.npz")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-frames", type=int, default=24)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--seconds", type=float, nargs="+",
                        default=[5, 10, 20, 30, 45, 60])
    args = parser.parse_args()

    labels = load_labels(FRAMES_DIR)
    label2idx = {l: i for i, l in enumerate(labels)}
    clips = load_clips(args.split)

    # how much clip actually exists, for context on the truncation levels
    durations = sorted(len(p) / EXTRACT_FPS for _, p in clips)
    med = durations[len(durations) // 2]
    print(f"[+] {args.split}: {len(clips)} clips, median duration {med:.0f}s "
          f"(min {durations[0]:.0f}s, max {durations[-1]:.0f}s)")

    ckpt = resolve_ckpt_dir(args.ckpt_dir) / args.ckpt
    model = MLXVideoClassifier(num_classes=len(labels), num_frames_per_video=args.n_frames,
                               dim=args.dim, n_heads=args.n_heads)
    model.load_weights(str(ckpt))
    mx.eval(model.parameters())
    model.eval()
    print(f"[+] {ckpt}\n")

    header = f"{'available':>10s} {'overall':>8s} {'macro':>7s}  " + "".join(f"{l:>7s}" for l in labels)
    print(header)
    print("-" * len(header))

    rows = []
    for sec in list(args.seconds) + [None]:
        overall, macro, per_class, _ = evaluate(
            model, clips, label2idx, labels, args.n_frames, sec)
        name = "full" if sec is None else f"{sec:.0f}s"
        print(f"{name:>10s} {overall:8.3f} {macro:7.3f}  " +
              "".join(f"{per_class[l]:7.2f}" for l in labels))
        rows.append((name, overall, macro, per_class))

    # summarise the cat-vs-waste split, which is the actual decision
    cat = [l for l in ("cat1", "cat2") if l in labels]
    waste = [l for l in ("pee", "poo", "both") if l in labels]
    print(f"\n{'available':>10s} {'cat-ID avg':>11s} {'waste avg':>10s}")
    print("-" * 34)
    for name, _, _, pc in rows:
        c = np.mean([pc[l] for l in cat]) if cat else float("nan")
        w = np.mean([pc[l] for l in waste]) if waste else float("nan")
        print(f"{name:>10s} {c:11.3f} {w:10.3f}")


if __name__ == "__main__":
    main()
