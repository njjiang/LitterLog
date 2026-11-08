#!/usr/bin/env python3
"""
Clip-level data loading for the temporal (video) classifier.

Reuses the per-frame JPGs already written by extract_frames.py (which scales
them to 256px at extraction time) — no separate extraction or caching step.
Groups those frames back into clips (using the same clip_id() parsing that
built the frozen split.json) and samples a fixed number of frames per clip:
  - training: a random subset each call (fresh sample every epoch)
  - eval:     evenly-spaced, deterministic

This keeps every clip's label attached to a whole sequence of frames,
matching how the data was actually labeled (one label per clip).
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from split_dataset import clip_id
from train import CACHE_DIR, FRAMES_DIR, SPLIT_PATH, augment, load_labels


def _resolve(path: Path) -> Path:
    cached = CACHE_DIR / path.relative_to(FRAMES_DIR)
    return cached if cached.exists() else path


def load_clips(split_name: str) -> list[tuple[str, list[Path]]]:
    """Group frame paths by clip for one split ('train'/'val'/'test').
    Returns [(label, [frame paths for that clip, chronological order]), ...]
    — one entry per clip, not per frame."""
    raw = json.loads(SPLIT_PATH.read_text())
    rel_paths = raw[split_name]

    by_clip: dict[str, list[Path]] = collections.defaultdict(list)
    clip_label: dict[str, str] = {}
    for rp in rel_paths:
        full = FRAMES_DIR / rp
        cid = clip_id(full)
        by_clip[cid].append(_resolve(full))
        clip_label[cid] = Path(rp).parent.name

    clips = []
    for cid, paths in by_clip.items():
        paths.sort()  # filenames are zero-padded frame indices -> chronological order
        clips.append((clip_label[cid], paths))
    return clips


# Counts how often truncation augmentation actually fired, so the training log
# can show what the model really saw rather than just the configured probability.
TRUNC_STATS = {"clips": 0, "truncated": 0, "frac_sum": 0.0}


def reset_trunc_stats() -> None:
    TRUNC_STATS.update(clips=0, truncated=0, frac_sum=0.0)


def trunc_summary() -> dict:
    n, t = TRUNC_STATS["clips"], TRUNC_STATS["truncated"]
    return {
        "clips": n,
        "truncated": t,
        "rate": (t / n) if n else 0.0,
        # mean kept fraction over the clips that were actually truncated
        "mean_kept": (TRUNC_STATS["frac_sum"] / t) if t else 1.0,
    }


def truncate_prefix(paths: list[Path], prob: float, min_frac: float) -> list[Path]:
    """Randomly keep only a leading prefix of a clip's frames.

    At inference the server classifies partial clips (30s first pass, and live
    events that are still short), but training always saw whole clips — so the
    model never learned to judge a truncated one. This reproduces exactly that
    truncation: keep the first k frames, then sample from those, the same way
    `--max-seconds` cuts the frame list at inference.

    Only meaningful for training; eval must stay deterministic.
    """
    TRUNC_STATS["clips"] += 1
    if prob <= 0 or random.random() >= prob or len(paths) < 2:
        return paths
    frac = random.uniform(min_frac, 1.0)
    keep = max(1, int(round(len(paths) * frac)))
    TRUNC_STATS["truncated"] += 1
    TRUNC_STATS["frac_sum"] += keep / len(paths)
    return paths[:keep]


def sample_clip_frames(paths: list[Path], n_frames: int, train: bool, color_aug: bool = True,
                       trunc_prob: float = 0.0, trunc_min: float = 0.25) -> np.ndarray:
    """Fixed-N frame sampling: random subset for training (different each
    epoch), evenly-spaced + deterministic for eval. Returns [n_frames, H, W, C]."""
    if train:
        paths = truncate_prefix(paths, trunc_prob, trunc_min)
    n_available = len(paths)
    if train:
        if n_available >= n_frames:
            idx = sorted(random.sample(range(n_available), n_frames))
        else:
            idx = sorted(random.choices(range(n_available), k=n_frames))
    else:
        denom = max(n_frames - 1, 1)
        idx = [min(round(i * (n_available - 1) / denom), n_available - 1) for i in range(n_frames)]

    frames = []
    for i in idx:
        img = Image.open(paths[i]).convert("RGB")
        frames.append(augment(img, train, color_aug=color_aug))
    return np.stack(frames)


def _balanced_order(clips: list[tuple[str, list[Path]]], batch_size: int) -> list[int]:
    """Class-balanced ordering: draw round-robin across classes so every batch
    sees a spread of classes rather than whatever random shuffling produced.

    With 44 poo / 55 pee clips out of 487, plain shuffling leaves most
    batch_size=4 batches with zero rare-class examples, so their gradient
    arrives in sparse high-variance bursts — which showed up as pee/poo
    accuracy oscillating in anti-phase between epochs. Rare classes are
    cycled through repeatedly (oversampled) within an epoch.
    """
    by_class: dict[str, list[int]] = collections.defaultdict(list)
    for i, (label, _) in enumerate(clips):
        by_class[label].append(i)
    for indices in by_class.values():
        random.shuffle(indices)

    # one epoch = enough batches to cover the largest class once
    n_classes = len(by_class)
    epoch_len = max(len(v) for v in by_class.values()) * n_classes
    cursors = {label: 0 for label in by_class}
    class_cycle = list(by_class)
    random.shuffle(class_cycle)

    order: list[int] = []
    while len(order) < epoch_len:
        for label in class_cycle:
            indices = by_class[label]
            c = cursors[label]
            if c >= len(indices):          # exhausted: reshuffle and wrap around
                random.shuffle(indices)
                cursors[label] = c = 0
            order.append(indices[c])
            cursors[label] = c + 1
    return order


CLIP_LOAD_POOL = None


def batch_iter(clips: list[tuple[str, list[Path]]], label2idx: dict[str, int],
               n_frames: int, batch_size: int, train: bool = True, color_aug: bool = True,
               balanced: bool = False, trunc_prob: float = 0.0, trunc_min: float = 0.25):
    """Yields (X, y): X is mx.array [batch, n_frames, H, W, C], y is mx.array [batch].

    balanced=True (training only) uses class-balanced sampling with rare-class
    oversampling; otherwise clips are shuffled (train) or kept in order (eval).
    """
    if train and balanced:
        idx = _balanced_order(clips, batch_size)
    else:
        idx = list(range(len(clips)))
        if train:
            random.shuffle(idx)

    def load_clip(i):
        label, paths = clips[i]
        frames = sample_clip_frames(
            paths, n_frames, train, color_aug=color_aug,
            trunc_prob=trunc_prob, trunc_min=trunc_min)
        return frames, label2idx[label]

    for start in range(0, len(idx), batch_size):
        batch_idx = idx[start:start + batch_size]
        load = CLIP_LOAD_POOL.map if CLIP_LOAD_POOL is not None else map
        frames, labels = zip(*load(load_clip, batch_idx))
        yield mx.array(np.stack(frames)), mx.array(list(labels))


if __name__ == "__main__":
    # quick smoke test
    labels = load_labels(FRAMES_DIR)
    label2idx = {l: i for i, l in enumerate(labels)}
    train_clips = load_clips("train")
    print(f"[+] {len(train_clips)} train clips, classes: {labels}")
    for X, y in batch_iter(train_clips, label2idx, n_frames=12, batch_size=4, train=True):
        print(X.shape, y.shape, y.tolist())
        break
