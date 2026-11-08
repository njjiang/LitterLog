#!/usr/bin/env python3
"""
Extract per-frame CLS attention weights and dump the top frames.

Replays MLXVideoClassifier's forward pass from outside using the model's own
submodules, so model.py needs no changes and there is no risk of the training
path and this diagnostic path drifting apart.

The CLS token is the pooled representation the classifier reads, so
attention_scores[:, :, 0, 1:] — CLS query attending to each frame key — is a
direct readout of which frames drove the prediction.

Usage:
    python attention_viz.py --video clip.mp4 --out /tmp/attn
    python attention_viz.py --split test --per-class 2 --out /tmp/attn
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from infer import Classifier, extract_frames
from video_dataset import sample_clip_frames
from storage_paths import directory

LITTERLOG_DIR = Path(__file__).resolve().parents[2] / "classifier"
FRAMES_DIR = LITTERLOG_DIR / "frames"


def cls_attention(clf: Classifier, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (probs, per_frame_attention).

    per_frame_attention[i] = how much the CLS token attended to frame i,
    averaged over heads. Excludes CLS's attention to itself.
    """
    model = clf.model
    X = mx.array(frames[None])                      # [1, t, H, W, C]
    b, t = X.shape[0], X.shape[1]

    # -- replay MLXVideoClassifier.__call__ --
    X_bt = X.reshape((b * t, *X.shape[2:]))
    frame_activations = model.W(model.resnet.get_features(X_bt))
    seq = mx.concat([mx.tile(model.cls, [b, 1, 1]),
                     frame_activations.reshape(b, t, -1)], axis=1)   # [b, t+1, dim]

    # -- replay MultiHeadAttention.__call__, keeping the scores --
    attn = model.attention
    dim = seq.shape[-1]
    Z = attn.layernorm(seq)
    qkv = attn.QKV(Z)
    query, key, value = mx.split(qkv, 3, axis=-1)
    shape = (b, t + 1, attn.n_heads, attn.head_dim)
    query, key, value = query.reshape(shape), key.reshape(shape), value.reshape(shape)

    scores = mx.softmax(
        mx.einsum("bqnd,bknd->bnqk", query, key) / mx.sqrt(attn.head_dim), axis=-1)

    attended = mx.einsum("bnqk,bknd->bnqd", scores, value)
    out = attn.out(attended.transpose(0, 2, 1, 3).reshape(b, t + 1, dim))
    probs = np.array(model.classifier(out[:, 0, :]).tolist())[0]
    probs = np.exp(probs - probs.max()); probs = probs / probs.sum()

    # CLS query (position 0) -> frame keys (positions 1..t), mean over heads
    cls_attn = np.array(scores[:, :, 0, 1:].tolist())[0].mean(axis=0)
    return probs, cls_attn


def clip_frames_from_dataset(clip_id: str) -> list[Path]:
    for label_dir in FRAMES_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        hits = sorted(label_dir.glob(f"{clip_id}_*.jpg"))
        if hits:
            return hits
    return []


def analyse(clf: Classifier, frame_paths: list[Path], out_dir: Path,
            name: str, true_label: str | None, topk: int) -> dict:
    frames = sample_clip_frames(frame_paths, clf.n_frames, train=False)
    probs, attn = cls_attention(clf, frames)

    pred_idx = int(probs.argmax())
    pred_label = clf.labels[pred_idx]

    # which source frames the sampler actually picked (mirrors train=False path)
    n_avail = len(frame_paths)
    denom = max(clf.n_frames - 1, 1)
    picked = [min(round(i * (n_avail - 1) / denom), n_avail - 1) for i in range(clf.n_frames)]

    order = np.argsort(-attn)
    clip_out = out_dir / name
    clip_out.mkdir(parents=True, exist_ok=True)

    for rank, slot in enumerate(order[:topk]):
        src = frame_paths[picked[slot]]
        dst = clip_out / f"top{rank+1}_slot{slot:02d}_w{attn[slot]:.4f}{src.suffix}"
        shutil.copy(src, dst)

    # lowest-attention frame too, as a contrast check
    lo = order[-1]
    shutil.copy(frame_paths[picked[lo]],
                clip_out / f"lowest_slot{lo:02d}_w{attn[lo]:.4f}{frame_paths[picked[lo]].suffix}")

    return {
        "name": name,
        "true": true_label,
        "pred": pred_label,
        "confidence": float(probs[pred_idx]),
        "attn": [float(a) for a in attn],
        "attn_max": float(attn.max()),
        "attn_min": float(attn.min()),
        "attn_uniform": 1.0 / clf.n_frames,
        "top_slots": [int(s) for s in order[:topk]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="single video file")
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--ckpt-dir", default="video_24f_balanced")
    parser.add_argument("--n-frames", type=int, default=24)
    args = parser.parse_args()

    clf = Classifier(ckpt_dir=args.ckpt_dir, n_frames=args.n_frames)
    out_dir = Path(args.out) if args.out else directory('attention-visualizations') / 'latest'
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    results = []

    if args.video:
        with tempfile.TemporaryDirectory(prefix='litterlog-attention-', dir=directory('tmp')) as tmp:
            fp = extract_frames(Path(args.video), Path(tmp))
            results.append(analyse(clf, fp, out_dir, Path(args.video).stem, None, args.topk))
    else:
        split = json.loads((LITTERLOG_DIR / "split.json").read_text())[args.split or "test"]
        by_class: dict[str, list[str]] = {}
        for rp in split:
            label = Path(rp).parent.name
            cid = re.sub(r"_\d{4}\.jpg$", "", Path(rp).name)
            by_class.setdefault(label, [])
            if cid not in by_class[label]:
                by_class[label].append(cid)

        for label, cids in sorted(by_class.items()):
            for cid in cids[:args.per_class]:
                fp = clip_frames_from_dataset(cid)
                if fp:
                    results.append(analyse(clf, fp, out_dir, f"{label}__{cid[:24]}", label, args.topk))

    print(f"\n{'clip':32s} {'true':6s} {'pred':6s} {'conf':>6s} {'attn max':>9s} {'vs uniform':>11s}")
    print("-" * 80)
    for r in results:
        ratio = r["attn_max"] / r["attn_uniform"]
        mark = "" if r["true"] is None else ("✓" if r["true"] == r["pred"] else "✗")
        print(f"{r['name'][:32]:32s} {str(r['true'] or '-'):6s} {r['pred']:6s} "
              f"{r['confidence']:6.1%} {r['attn_max']:9.4f} {ratio:10.2f}x {mark}")

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[+] frames + results.json in {out_dir}")
    print(f"    uniform weight would be {1.0/args.n_frames:.4f} "
          f"— ratios near 1.0x mean attention is flat (not selective)")


if __name__ == "__main__":
    main()
