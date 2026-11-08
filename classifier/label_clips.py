#!/usr/bin/env python3
"""
Label video clips with a keypress.
Labels are saved as sidecar .txt files next to each clip.

Usage:
    python label_clips.py

Controls:
    1  → cat1
    2  → cat2
    p  → pee
    o  → poo
    b  → both (pee + poo)
    n  → none (not a cat)
    s  → skip (leave unlabeled)
    r  → replay current clip
    u  → undo last label
    q  → quit
"""

from __future__ import annotations

import signal
import subprocess
import sys
import termios
import tty
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "unifi_data"

KEYS = {
    "1": "cat1",
    "2": "cat2",
    "c": "cat1",
    "f": "cat2",
    "p": "pee",
    "o": "poo",
    "b": "both",
    "n": "none",
    "s": None,   # skip — leave unlabeled
}


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def label_path(clip: Path) -> Path:
    return clip.with_suffix(".txt")


def get_label(clip: Path) -> str | None:
    p = label_path(clip)
    return p.read_text().strip() if p.exists() else None


def set_label(clip: Path, label: str | None) -> None:
    p = label_path(clip)
    if label is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(label + "\n")


def find_clips(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.mp4"))


def open_clip(clip: Path) -> subprocess.Popen:
    return subprocess.Popen(["open", "-R", str(clip)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def close_clip(proc: subprocess.Popen) -> None:
    try:
        proc.kill()
    except Exception:
        pass


def main():
    clips = find_clips(DATA_DIR)
    if not clips:
        print(f"[!] No .mp4 files found in {DATA_DIR}")
        sys.exit(1)

    unlabeled = [c for c in clips if get_label(c) is None]
    total = len(unlabeled)

    if total == 0:
        print("[+] All clips are already labeled!")
        sys.exit(0)

    print(f"[+] {len(clips)} clips total, {total} unlabeled")
    print("    Controls: 1=cat1  2=cat2  p=pee  o=poo  b=both  n=none  s=skip  r=replay  u=undo  q=quit\n")

    original_term = termios.tcgetattr(sys.stdin.fileno())

    def cleanup(sig=None, frame=None):
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
        print("\n[+] Exited.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    history: list[Path] = []
    proc: subprocess.Popen | None = None

    i = 0
    while i < len(unlabeled):
        clip = unlabeled[i]
        done = i + 1
        print(f"[{done}/{total}] {clip.relative_to(DATA_DIR)}")

        proc = open_clip(clip)
        print("  Waiting for label … ", end="", flush=True)

        while True:
            ch = getch()

            if ch == "q":
                close_clip(proc)
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
                print(f"\n[+] Done for now. {done - 1}/{total} labeled this session.")
                sys.exit(0)

            if ch == "r":
                close_clip(proc)
                proc = open_clip(clip)
                print("replay … ", end="", flush=True)
                continue

            if ch == "u":
                if history:
                    prev_clip = history.pop()
                    set_label(prev_clip, None)
                    close_clip(proc)
                    i = unlabeled.index(prev_clip)
                    print("undo")
                    break
                else:
                    print("(nothing to undo) ", end="", flush=True)
                    continue

            if ch in KEYS:
                label = KEYS[ch]
                set_label(clip, label)
                if label is not None:
                    history.append(clip)
                close_clip(proc)
                print(label if label is not None else "skip")
                i += 1
                break

    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original_term)
    print(f"\n[+] All {total} clips labeled!")


if __name__ == "__main__":
    main()
