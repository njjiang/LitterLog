#!/usr/bin/env python3
"""
Extract frames from labeled video clips for training.

For each .mp4 with a sidecar .txt label, extracts frames at --fps rate
and saves them to:
    frames/<label>/<clip-stem>_<frame-num>.jpg

Usage:
    python extract_frames.py
    python extract_frames.py --fps 2
    python extract_frames.py --workers 8
"""

from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA_DIRS = [
    Path(__file__).resolve().parent / "unifi_data",
    Path(__file__).resolve().parent / "unifi_data2",
]
FRAMES_DIR = Path(__file__).resolve().parent / "frames"


def get_duration(clip: Path) -> float:
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(clip),
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract(clip: Path, label: str, fps: float, skip: float = 0.0, end_trim: float = 0.0, size: int = 256,
            max_frames: int | None = None) -> tuple[str, int, str]:
    out_dir = FRAMES_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / f"{clip.stem}_%04d.jpg")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if skip > 0:
        cmd += ["-ss", str(skip)]
    cmd += ["-i", str(clip)]
    if end_trim > 0:
        duration = get_duration(clip) - skip - end_trim
        if duration > 0:
            cmd += ["-t", str(duration)]
    # Scale down at extraction time (not a separate caching pass) — avoids ever
    # writing full camera-resolution (2688x1512) frames to disk, which was
    # blowing through available disk space (26GB for a partial extraction).
    cmd += ["-vf", f"fps={fps},scale={size}:{size}", "-q:v", "4"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd.append(out_pattern)

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return label, 0, f"ERROR: {result.stderr.decode()[-100:]}"

    n = len(list(out_dir.glob(f"{clip.stem}_*.jpg")))
    return label, n, clip.name


def main():
    parser = argparse.ArgumentParser(description="Extract training frames from labeled clips")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default 1.0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel ffmpeg workers")
    parser.add_argument("--skip", type=float, default=1.0, help="Seconds to skip at start of each clip (default 1.0)")
    parser.add_argument("--trim", type=float, default=1.0, help="Seconds to trim from end of each clip (default 1.0)")
    parser.add_argument("--size", type=int, default=256, help="Frame output size, WxH square (default 256)")
    parser.add_argument("--dirs", type=str, nargs="+", default=None,
                        help="clip folders to scan (default: both unifi_data dirs)")
    parser.add_argument("--max-frames", type=int, help="Maximum frames per clip")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")

    data_dirs = [Path(d) for d in args.dirs] if args.dirs else DATA_DIRS
    clips = []
    for d in data_dirs:
        if not d.exists():
            print(f"[!] skipping missing {d}")
            continue
        found = sorted(d.rglob("*.mp4"))
        print(f"[+] {d}: {len(found)} clips")
        clips += found
    labeled = [
        (c, c.with_suffix(".txt").read_text().strip())
        for c in clips
        if c.with_suffix(".txt").exists() and c.with_suffix(".txt").read_text().strip()
    ]
    unlabeled_count = len(clips) - len(labeled)

    # Skip already-extracted clips
    to_extract = []
    skipped = []
    for clip, label in labeled:
        existing = list((FRAMES_DIR / label).glob(f"{clip.stem}_*.jpg")) if (FRAMES_DIR / label).exists() else []
        if existing:
            skipped.append((label, len(existing)))
        else:
            to_extract.append((clip, label))

    print(f"[+] {len(labeled)} labeled clips, {unlabeled_count} unlabeled (ignored)")
    print(f"[+] {len(skipped)} already extracted, {len(to_extract)} to process")
    print(f"[+] {args.workers} workers, {args.fps} fps, skip={args.skip}s trim={args.trim}s → {FRAMES_DIR}\n")

    label_counts: dict[str, int] = {}
    for label, n in skipped:
        label_counts[label] = label_counts.get(label, 0) + n

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract, clip, label, args.fps, args.skip, args.trim, args.size, args.max_frames): (clip, label)
            for clip, label in to_extract
        }
        for future in as_completed(futures):
            label, n, name = future.result()
            label_counts[label] = label_counts.get(label, 0) + n
            completed += 1
            status = f"{n} frames" if n > 0 else name  # show error inline
            print(f"  [{completed}/{len(to_extract)}] {label:6s}  {status}  {name}")

    print(f"\n[+] Done. Frame counts per class:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:8s}  {count}")
    print(f"\n    Total: {sum(label_counts.values())} frames in {FRAMES_DIR}")


if __name__ == "__main__":
    main()
