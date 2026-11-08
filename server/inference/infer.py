#!/usr/bin/env python3
"""
Run the trained classifier on a single video clip.

Deliberately reuses the *training* frame-extraction params and the *training*
sampling/preprocessing functions (sample_clip_frames with train=False, which
applies augment(train=False)). Reimplementing them here would risk silent
train/inference skew — the model would see subtly different pixels than it was
trained on and lose accuracy for no visible reason.

Usage:
    python infer.py --video /path/to/clip.mp4
    python infer.py --video clip.mp4 --ckpt-dir video_24f_balanced --topk 3
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import imageio_ffmpeg
import av
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from model import MLXVideoClassifier
from video_dataset import sample_clip_frames
from storage_paths import directory

LITTERLOG_DIR = Path(__file__).resolve().parents[2] / "classifier"
CKPT_ROOT = LITTERLOG_DIR / "checkpoints"
LABELS_PATH = LITTERLOG_DIR / "labels.txt"


def resolve_ckpt_dir(name: str) -> Path:
    """Checkpoints live under litterlog/checkpoints/<name>. Older runs used
    litterlog/checkpoints_<name> at the top level, so fall back to that if a
    caller (or an old command line) still refers to one."""
    p = CKPT_ROOT / name
    if p.exists():
        return p
    legacy = LITTERLOG_DIR / (name if name.startswith("checkpoints") else f"checkpoints_{name}")
    return legacy if legacy.exists() else p

# Must match extract_frames.py defaults used to build the training set.
EXTRACT_FPS = 2.0
EXTRACT_SIZE = 256
EXTRACT_SKIP = 1.0
EXTRACT_TRIM = 1.0


@dataclass
class Prediction:
    label: str
    confidence: float
    probs: dict[str, float]
    n_frames_available: int
    best_frame: bytes | None = None      # highest CLS-attention frame (JPEG)
    attn_ratio: float | None = None      # peak attention / uniform; 1.0 == flat
    best_frame_seconds: float | None = None   # where in the clip that frame was

    def __str__(self) -> str:
        ranked = sorted(self.probs.items(), key=lambda kv: -kv[1])
        detail = "  ".join(f"{l}={p:.3f}" for l, p in ranked)
        return f"{self.label} ({self.confidence:.1%})  [{detail}]"


def load_labels() -> list[str]:
    """Class order is baked into the checkpoint's output layer, so read the
    labels file written at training time rather than re-deriving from dirs."""
    if not LABELS_PATH.exists():
        raise SystemExit(f"[!] {LABELS_PATH} not found — needed for class order")
    return [l.strip() for l in LABELS_PATH.read_text().splitlines() if l.strip()]


def get_duration(clip: Path) -> float:
    with av.open(str(clip)) as container:
        return float(container.duration / av.time_base) if container.duration is not None else 0.0


def extract_frame_at(clip: Path, seconds: float, width: int = 720) -> bytes | None:
    """Pull one frame from the source video at a timestamp, undistorted.

    Training frames are squashed to a square (scale=256:256), which is fine for
    the model but looks wrong to a human. Once attention has picked *which*
    frame matters, re-extract that moment from the original at its real aspect
    ratio for display.
    """
    import tempfile as _tf
    with _tf.TemporaryDirectory(prefix="litterlog_thumb_", dir=directory('tmp')) as tmp:
        out = Path(tmp) / "f.jpg"
        r = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(max(0.0, seconds)), "-i", str(clip),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3",
            str(out),
        ], capture_output=True)
        if r.returncode == 0 and out.exists():
            return out.read_bytes()
    return None


def frame_index_to_seconds(idx: int) -> float:
    """Frame i of the extracted sequence back to a timestamp in the source.
    Extraction starts at EXTRACT_SKIP and samples at EXTRACT_FPS."""
    return EXTRACT_SKIP + idx / EXTRACT_FPS


def extract_frames(clip: Path, out_dir: Path) -> list[Path]:
    """Same ffmpeg invocation as extract_frames.py so inference frames are
    pixel-identical in kind to training frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / "frame_%04d.jpg")

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error"]
    if EXTRACT_SKIP > 0:
        cmd += ["-ss", str(EXTRACT_SKIP)]
    cmd += ["-i", str(clip)]
    if EXTRACT_TRIM > 0:
        duration = get_duration(clip) - EXTRACT_SKIP - EXTRACT_TRIM
        if duration > 0:
            cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={EXTRACT_FPS},scale={EXTRACT_SIZE}:{EXTRACT_SIZE}",
            "-q:v", "4", out_pattern]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[-300:]}")

    return sorted(out_dir.glob("frame_*.jpg"))


class Classifier:
    """Loads the checkpoint once; call predict() per clip."""

    def __init__(self, ckpt_dir: str = "video_24f_balanced",
                 ckpt: str = "best.npz", n_frames: int = 24,
                 dim: int = 128, n_heads: int = 8):
        self.labels = load_labels()
        self.n_frames = n_frames

        ckpt_path = resolve_ckpt_dir(ckpt_dir) / ckpt
        if not ckpt_path.exists():
            raise SystemExit(f"[!] checkpoint not found: {ckpt_path}")

        self.model = MLXVideoClassifier(
            num_classes=len(self.labels),
            num_frames_per_video=n_frames,
            dim=dim,
            n_heads=n_heads,
        )
        self.model.load_weights(str(ckpt_path))
        mx.eval(self.model.parameters())
        self.model.eval()
        self.ckpt_path = ckpt_path

    def predict(self, video: Path, with_attention: bool = False,
                max_seconds: float | None = 60.0) -> Prediction:
        """max_seconds caps how much of the clip is used. Measured on the test
        split (eval_truncated.py): a 60s cap scores slightly *better* than the
        full clip (macro 0.929 vs 0.912) because the post-dump tail carries no
        signal — and it's faster. None disables the cap."""
        with tempfile.TemporaryDirectory(prefix="litterlog_infer_", dir=directory('tmp')) as tmp:
            frame_paths = extract_frames(video, Path(tmp))
            if not frame_paths:
                raise RuntimeError(f"no frames extracted from {video}")

            if max_seconds is not None:
                keep = max(1, int(round(max_seconds * EXTRACT_FPS)))
                frame_paths = frame_paths[:keep]

            # train=False -> evenly-spaced deterministic sampling + center crop,
            # exactly as used for val/test evaluation.
            frames = sample_clip_frames(frame_paths, self.n_frames, train=False)

            best_frame = None
            attn_ratio = None
            best_seconds = None
            if with_attention:
                from attention_viz import cls_attention
                probs, attn = cls_attention(self, frames)
                # map attention slot back to its source frame on disk
                n_avail = len(frame_paths)
                denom = max(self.n_frames - 1, 1)
                picked = [min(round(i * (n_avail - 1) / denom), n_avail - 1)
                          for i in range(self.n_frames)]
                # Attention says which moment mattered; go back to the source
                # for a presentable frame instead of the squashed 256x256 one.
                best_idx = picked[int(attn.argmax())]
                best_seconds = frame_index_to_seconds(best_idx)
                best_frame = extract_frame_at(video, best_seconds)
                if best_frame is None:                      # ffmpeg seek failed
                    best_frame = frame_paths[best_idx].read_bytes()
                attn_ratio = float(attn.max() * self.n_frames)
            else:
                X = mx.array(frames[None])  # [1, n_frames, H, W, C]
                probs = np.array(self.model(X).tolist())[0]

        idx = int(probs.argmax())
        return Prediction(
            label=self.labels[idx],
            confidence=float(probs[idx]),
            probs={l: float(p) for l, p in zip(self.labels, probs)},
            n_frames_available=len(frame_paths),
            best_frame=best_frame,
            attn_ratio=attn_ratio,
            best_frame_seconds=best_seconds,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--ckpt-dir", default="video_24f_balanced")
    parser.add_argument("--ckpt", default="best.npz")
    parser.add_argument("--n-frames", type=int, default=24)
    args = parser.parse_args()

    clf = Classifier(ckpt_dir=args.ckpt_dir, ckpt=args.ckpt, n_frames=args.n_frames)
    print(f"[+] model: {clf.ckpt_path}")
    print(f"[+] classes: {clf.labels}")

    pred = clf.predict(Path(args.video))
    print(f"\n{pred}")
    print(f"    ({pred.n_frames_available} frames available, sampled {clf.n_frames})")


if __name__ == "__main__":
    main()
