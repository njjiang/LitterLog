#!/usr/bin/env python3
"""
Create a frozen, clip-level train/val/test split.

Frames are extracted at ~2fps from video clips, so frames from the same clip
are near-duplicates. Splitting at the frame level leaks near-identical images
across train/val (or train/test), which inflates accuracy without meaning
anything. This script splits at the CLIP level instead — every frame from a
given clip goes entirely into one split — and freezes the result to disk so
every training run uses the exact same split.

Usage:
    python split_dataset.py
    python split_dataset.py --val-frac 0.15 --test-frac 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

LITTERLOG_DIR = Path(__file__).resolve().parents[2] / "classifier"
FRAMES_DIR = LITTERLOG_DIR / "frames"
SPLIT_PATH = LITTERLOG_DIR / "split.json"

_FRAME_SUFFIX_RE = re.compile(r"_\d{4}\.jpg$")


def clip_id(path: Path) -> str:
    """Frames are named <clip-stem>_NNNN.jpg — strip the frame index to recover the source clip."""
    return _FRAME_SUFFIX_RE.sub("", path.name)


def load_dataset(frames_dir: Path) -> tuple[list[Path], list[str]]:
    labels = sorted(d.name for d in frames_dir.iterdir() if d.is_dir())
    paths, label_names = [], []
    for label in labels:
        for p in (frames_dir / label).glob("*.jpg"):
            paths.append(p)
            label_names.append(label)
    return paths, label_names


def clip_level_split(paths, label_names, val_frac, test_frac, seed):
    rng = random.Random(seed)

    clip_to_paths: dict[str, list[Path]] = collections.defaultdict(list)
    clip_to_label: dict[str, str] = {}
    for p, l in zip(paths, label_names):
        cid = clip_id(p)
        clip_to_paths[cid].append(p)
        clip_to_label[cid] = l

    clips_by_class = collections.defaultdict(list)
    for cid, l in clip_to_label.items():
        clips_by_class[l].append(cid)

    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    stats: dict[str, dict[str, int]] = {}

    for label, clips in sorted(clips_by_class.items()):
        rng.shuffle(clips)
        n = len(clips)
        n_val = max(1, round(n * val_frac)) if n > 2 else 0
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        # leave at least one clip for train if possible
        while n_val + n_test >= n and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            else:
                break

        val_clips = clips[:n_val]
        test_clips = clips[n_val:n_val + n_test]
        train_clips = clips[n_val + n_test:]

        for cid in train_clips:
            split["train"].extend(str(p.relative_to(FRAMES_DIR)) for p in clip_to_paths[cid])
        for cid in val_clips:
            split["val"].extend(str(p.relative_to(FRAMES_DIR)) for p in clip_to_paths[cid])
        for cid in test_clips:
            split["test"].extend(str(p.relative_to(FRAMES_DIR)) for p in clip_to_paths[cid])

        stats[label] = {
            "train_clips": len(train_clips), "val_clips": len(val_clips), "test_clips": len(test_clips),
            "train_frames": sum(len(clip_to_paths[c]) for c in train_clips),
            "val_frames": sum(len(clip_to_paths[c]) for c in val_clips),
            "test_frames": sum(len(clip_to_paths[c]) for c in test_clips),
        }

    return split, stats


def verify_no_leakage(split: dict[str, list[str]]) -> bool:
    def clips_in(name):
        return set(clip_id(Path(p)) for p in split[name])

    train_c, val_c, test_c = clips_in("train"), clips_in("val"), clips_in("test")
    overlaps = {
        "train∩val": train_c & val_c,
        "train∩test": train_c & test_c,
        "val∩test": val_c & test_c,
    }
    ok = all(len(v) == 0 for v in overlaps.values())
    for name, ov in overlaps.items():
        if ov:
            print(f"  [!] LEAKAGE: {name} overlap = {len(ov)} clips")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths, label_names = load_dataset(FRAMES_DIR)
    print(f"[+] {len(paths)} frames across {len(set(label_names))} classes")

    split, stats = clip_level_split(paths, label_names, args.val_frac, args.test_frac, args.seed)

    print(f"\n[+] Split sizes (frames): "
          f"train={len(split['train'])}  val={len(split['val'])}  test={len(split['test'])}")
    print("\n[+] Per-class breakdown (clips / frames):")
    for label, s in sorted(stats.items()):
        print(f"    {label:8s}  "
              f"train={s['train_clips']:3d}c/{s['train_frames']:5d}f  "
              f"val={s['val_clips']:3d}c/{s['val_frames']:5d}f  "
              f"test={s['test_clips']:3d}c/{s['test_frames']:5d}f")

    print("\n[+] Verifying no clip leakage across splits …")
    ok = verify_no_leakage(split)
    print("    OK — no overlap" if ok else "    FAILED — see above")
    if not ok:
        raise SystemExit(1)

    SPLIT_PATH.write_text(json.dumps(split, indent=2))
    print(f"\n[+] Frozen split written to {SPLIT_PATH}")


if __name__ == "__main__":
    main()
