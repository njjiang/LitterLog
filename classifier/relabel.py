#!/usr/bin/env python3
"""
Re-label clips that already carry a given label — keypress-driven, same feel
as label_clips.py, but it *revisits* existing labels instead of skipping them.

label_clips.py only offers unlabeled clips, so there was no way to go back and
audit a class you no longer trust. This does that: filter to a label, step
through, overwrite where wrong.

Usage:
    python relabel.py --label none
    python relabel.py --label none --dir ./unifi_data2
    python relabel.py --label none --dirs A B      # several folders at once
    python relabel.py --label pee --dry-run        # just list what matches

Controls:
    1=cat1  2=cat2  p=pee  o=poo  b=both  n=none
    <space> keep as-is      r=replay      u=undo      q=quit
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import termios
import tty
from pathlib import Path

DEFAULT_DIRS = [
    Path(__file__).resolve().parent / "unifi_data",
    Path(__file__).resolve().parent / "unifi_data2",
]

KEYS = {
    "1": "cat1",
    "2": "cat2",
    "c": "cat1",
    "f": "cat2",
    "p": "pee",
    "o": "poo",
    "b": "both",
    "n": "none",
}


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def label_path(clip: Path) -> Path:
    return clip.with_suffix(".txt")


def get_label(clip: Path) -> str | None:
    p = label_path(clip)
    return p.read_text().strip() if p.exists() else None


def set_label(clip: Path, label: str) -> None:
    label_path(clip).write_text(label + "\n")


def open_clip(clip: Path) -> subprocess.Popen:
    return subprocess.Popen(["open", "-R", str(clip)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def close_clip(proc: subprocess.Popen | None) -> None:
    try:
        if proc:
            proc.kill()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="which existing label to revisit")
    ap.add_argument("--dir", type=str, default=None, help="single folder to scan")
    ap.add_argument("--dirs", type=str, nargs="+", default=None, help="several folders")
    ap.add_argument("--dry-run", action="store_true", help="list matches and exit")
    args = ap.parse_args()

    if args.dirs:
        dirs = [Path(d) for d in args.dirs]
    elif args.dir:
        dirs = [Path(args.dir)]
    else:
        dirs = [d for d in DEFAULT_DIRS if d.exists()]
    if not dirs:
        sys.exit("[!] no data folders found (NAS not mounted?)")

    clips: list[Path] = []
    for d in dirs:
        if not d.exists():
            print(f"[!] skipping missing {d}")
            continue
        clips += [c for c in sorted(d.rglob("*.mp4")) if get_label(c) == args.label]

    if not clips:
        sys.exit(f"[!] no clips currently labeled '{args.label}' under "
                 + ", ".join(str(d) for d in dirs))

    total = len(clips)
    print(f"[+] {total} clips labeled '{args.label}'")
    for d in dirs:
        print(f"    {d}")
    if args.dry_run:
        for c in clips:
            print(f"    {c.name}")
        return

    print("\n    Controls: 1=cat1  2=cat2  p=pee  o=poo  b=both  n=none")
    print("              <space>=keep  r=replay  u=undo  q=quit\n")

    original_term = termios.tcgetattr(sys.stdin.fileno())

    def cleanup(sig=None, frame=None):
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
        print("\n[+] Exited.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    history: list[tuple[Path, str]] = []   # (clip, previous label) for undo
    changed = kept = 0
    proc: subprocess.Popen | None = None

    i = 0
    while i < len(clips):
        clip = clips[i]
        current = get_label(clip)
        print(f"[{i+1}/{total}] {clip.name}  (currently {current})")

        proc = open_clip(clip)
        print("  ", end="", flush=True)

        while True:
            ch = getch()

            if ch == "q":
                close_clip(proc)
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
                print(f"\n[+] Stopped. {changed} changed, {kept} kept.")
                return

            if ch == "r":
                close_clip(proc)
                proc = open_clip(clip)
                print("replay … ", end="", flush=True)
                continue

            if ch == "u":
                if not history:
                    print("(nothing to undo) ", end="", flush=True)
                    continue
                prev_clip, prev_label = history.pop()
                set_label(prev_clip, prev_label)
                close_clip(proc)
                i = clips.index(prev_clip)
                print(f"undo -> restored {prev_label}")
                break

            if ch == " ":
                history.append((clip, current))
                close_clip(proc)
                kept += 1
                print("keep")
                i += 1
                break

            if ch in KEYS:
                new = KEYS[ch]
                history.append((clip, current))
                set_label(clip, new)
                close_clip(proc)
                if new != current:
                    changed += 1
                    print(f"{current} -> {new}")
                else:
                    kept += 1
                    print("keep")
                i += 1
                break

    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
    print(f"\n[+] Done. {changed} changed, {kept} kept, {total} reviewed.")


if __name__ == "__main__":
    main()
