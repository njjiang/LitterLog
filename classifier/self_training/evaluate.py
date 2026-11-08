#!/usr/bin/env python3
"""Evaluate a checkpoint on local first-minute clip caches."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, help='Evaluate one checkpoint, otherwise rank all saved checkpoints by validation')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parents[1]
    outputs = Path(os.environ.get('LITTERLOG_EVAL_OUTPUTS', Path(__file__).resolve().parents[1] / 'outputs'))
    run = args.run_dir.resolve()
    checkpoint = args.checkpoint
    if checkpoint is None and not args.single:
        candidates = sorted(p for p in (run / 'checkpoints').rglob('*.npz') if not p.name.startswith('optimizer_'))
        if not candidates:
            parser.error('No saved checkpoint files found under this run.')
        scores = []
        fingerprints = {}
        for candidate in candidates:
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest in fingerprints:
                scores.append(dict(checkpoint=str(candidate), duplicate_of=fingerprints[digest],
                                   checkpoint_sha256=digest))
                continue
            fingerprints[digest] = str(candidate)
            command = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run),
                       '--checkpoint', str(candidate), '--single',
                       '--batch-size', str(args.batch_size), '--workers', str(args.workers)]
            subprocess.run(command, check=True)
            report_path = run / ('validation_' + digest[:16] + '.json')
            report = json.loads(report_path.read_text())
            validation = report['cohorts']['val']
            scores.append(dict(checkpoint=str(candidate), checkpoint_sha256=digest,
                               validation=validation, report=str(report_path)))
        ranked = sorted([r for r in scores if 'validation' in r],
                        key=lambda r: (-r['validation']['balanced_accuracy'],
                                       -r['validation']['accuracy'], r['checkpoint']))
        winner = ranked[0]
        summary = dict(selection_metric='validation balanced accuracy, then validation accuracy',
                       selected_checkpoint=winner['checkpoint'], ranking=ranked,
                       duplicates=[r for r in scores if 'duplicate_of' in r])
        (run / 'checkpoint_ranking.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('Selected ' + winner['checkpoint'], flush=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run-dir', str(run),
                        '--checkpoint', winner['checkpoint'], '--batch-size', str(args.batch_size),
                        '--workers', str(args.workers)], check=True)
        return
    if checkpoint is None:
        candidates = sorted(run.glob('checkpoints/*/best.npz')) or sorted(run.glob('checkpoints/*/last.npz'))
        if len(candidates) != 1:
            parser.error('Specify --checkpoint because the run has zero or multiple candidate checkpoints.')
        checkpoint = candidates[0]
    checkpoint = checkpoint.resolve()
    fingerprint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source = model_dir / 'combined_training_yrr_uepf'
    split = json.loads((source / 'split.json').read_text())
    cohorts = {}
    for name in ('val', 'test'):
        clips = defaultdict(list)
        for relative in split[name]:
            clips[Path(relative).stem.rsplit('_', 1)[0]].append(relative)
        cohorts[name] = [dict(clip=clip, label=Path(paths[0]).parent.name,
                             frames=[str(source / 'frames' / p) for p in sorted(paths)[:120]])
                         for clip, paths in clips.items()]
    known = json.loads((outputs / 'trained_checkpoint_eval/manifest.json').read_text())['rows']
    known = {r['clip']: r for r in known}
    corrections = {}
    for marker in (outputs / 'corrected_clips').rglob('*.txt'):
        row = dict(known[marker.stem])
        row['label'] = marker.read_text().strip()
        corrections[marker.stem] = row
    cohorts['reviewed_corrections'] = list(corrections.values())
    losses = json.loads((outputs / 'trained_checkpoint_eval/lost_attention/attention.json').read_text())['records']
    cohorts['five_regressions'] = [known[r['clip']] for r in losses]
    if args.single:
        cohorts = {'val': cohorts['val']}
    training_ids = None
    manifest_path = run / 'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        training_ids = {r['clip'] for r in manifest['rows']}
    sys.path.insert(0, str(model_dir))
    import numpy as np
    import mlx.core as mx
    import model
    from infer import Classifier
    from video_dataset import sample_clip_frames
    create = model.create_model
    def initialize(*a, **kw):
        kw['weights'] = False
        return create(*a, **kw)
    model.create_model = initialize
    try:
        classifier = Classifier(ckpt_dir=str(checkpoint.parent), ckpt=checkpoint.name)
    finally:
        model.create_model = create
    classifier.model.load_weights(str(checkpoint), strict=True)
    classifier.model.eval()
    labels = classifier.labels
    results = dict(checkpoint=str(checkpoint), checkpoint_sha256=fingerprint,
                   window='24 uniformly sampled frames from first120 cached frames at2fps',
                   cohorts={})
    def sample(row):
        return sample_clip_frames([Path(p) for p in row['frames'][:120]], 24, train=False)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, rows in cohorts.items():
            records = []
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                frames = np.stack(list(pool.map(sample, batch)))
                probabilities = np.asarray(classifier.model(mx.array(frames)))
                for row, probs in zip(batch, probabilities):
                    records.append(dict(clip=row['clip'], label=row['label'],
                        prediction=labels[int(probs.argmax())],
                        in_training=None if training_ids is None else row['clip'] in training_ids,
                        probabilities=dict(zip(labels, map(float, probs)))))
            counts = {}
            for label in sorted({r['label'] for r in records}):
                selected = [r for r in records if r['label'] == label]
                correct = sum(r['prediction'] == label for r in selected)
                counts[label] = dict(correct=correct, total=len(selected), recall=correct/len(selected))
            correct = sum(r['label'] == r['prediction'] for r in records)
            score = dict(correct=correct, total=len(records), accuracy=correct/len(records),
                balanced_accuracy=float(np.mean([c['recall'] for c in counts.values()])),
                training_overlap=None if training_ids is None else sum(r['in_training'] for r in records),
                by_class=counts, predictions=records)
            results['cohorts'][name] = score
            print(f"{name}: {correct}/{len(records)} ({score['accuracy']:.1%}), balanced accuracy {score['balanced_accuracy']:.1%}, training overlap {score['training_overlap']}", flush=True)
    if fingerprint != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
        raise RuntimeError('Checkpoint changed during evaluation. Evaluate a completed checkpoint.')
    destination = (run / ('validation_' + fingerprint[:16] + '.json') if args.single
                   else run / 'selected_checkpoint_evaluation.json')
    destination.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Saved {destination}')


if __name__ == '__main__':
    main()
