#!/usr/bin/env python3
"""Combine cached human labels and saved unifi_data2 predictions, then fine-tune."""
import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get('LITTERLOG_EVAL_OUTPUTS', Path(__file__).resolve().parents[1] / 'outputs'))
MODEL = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(__file__).resolve().parents[1]
LABELS = {'cat1', 'cat2', 'none', 'pee', 'poo'}


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=MODEL)
    p.add_argument('--teacher', type=Path, default=MODEL / 'checkpoints/video_24f_balanced/best.npz')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--old', type=Path, default=DATA_ROOT / 'unifi_data')
    p.add_argument('--new', type=Path, default=DATA_ROOT / 'unifi_data2')
    p.add_argument('--events', type=Path, default=MODEL / 'events')
    p.add_argument('--inventory', action='store_true', help='Count local cached labels and saved predictions without accessing video roots')
    p.add_argument('--prepared', action='store_true', help='Train using this run directory’s prepared manifest')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--reuse-data', type=Path, help='Reuse an extracted run and restore its original held-out split')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--wandb-mode', choices=('online', 'offline', 'disabled'), default='offline')
    p.add_argument('--wandb-project', default='litterlog_video')
    args = p.parse_args()
    args.run_dir = args.run_dir.resolve()
    dataset = args.model_dir / 'combined_training_yrr_uepf'
    if args.resume or args.prepared:
        manifest = read(args.run_dir / 'manifest.json')
        if not manifest.get('validation') or not manifest.get('test'):
            p.error('Resume requires a run prepared with preserved validation/test sets.')
        from training import train
        train(args, manifest)
        return
    if args.reuse_data:
        if (args.run_dir / 'manifest.json').exists():
            p.error('Use a fresh --run-dir or --resume.')
        original = {}
        for split, paths in read(dataset / 'split.json').items():
            for name in paths:
                original[Path(name).stem.rsplit('_', 1)[0]] = split
        previous = read(args.reuse_data / 'manifest.json')
        all_rows = previous['rows'] + previous.get('validation', []) + previous.get('test', [])
        partitions = {key: [] for key in ('train', 'val', 'test')}
        seen = set()
        for row in all_rows:
            if row['clip'] in seen:
                raise ValueError('Duplicate cached clip ' + row['clip'])
            seen.add(row['clip'])
            if not all(Path(f).is_file() for f in row['frames']):
                raise ValueError('Missing cached frames for ' + row['clip'])
            partitions[original.get(row['clip'], 'train')].append(row)
        for key in ('val', 'test'):
            expected = {clip for clip, split in original.items() if split == key}
            actual = {row['clip'] for row in partitions[key]}
            if actual != expected:
                raise ValueError(f'Incomplete original {key} clips in reused data.')
        manifest = dict(rows=partitions['train'], validation=partitions['val'],
                        test=partitions['test'], retest=previous['retest'],
                        source_split=str(dataset / 'split.json'))
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print({key: len(value) for key, value in partitions.items()}, flush=True)
        if not args.prepare_only:
            from training import train
            train(args, manifest)
        return
    original_splits = {}
    for split, paths in read(dataset / 'split.json').items():
        for name in paths:
            clip = Path(name).stem.rsplit('_', 1)[0]
            previous = original_splits.setdefault(clip, split)
            if previous != split:
                raise ValueError(f'Clip occurs in multiple original splits: {clip}')
    rows = {}
    grouped = defaultdict(list)
    for paths in read(dataset / 'split.json').values():
        for name in paths:
            path = Path(name)
            grouped[path.stem.rsplit('_', 1)[0]].append(path)
    for clip, paths in grouped.items():
        rows[clip] = dict(clip=clip, label=paths[0].parent.name, label_source='human',
                          weight=1., frames=[str(dataset / 'frames' / x) for x in sorted(paths)[:120]])
    eval_dir = OUTPUTS / 'trained_checkpoint_eval'
    for r in read(eval_dir / 'manifest.json')['rows']:
        rows.setdefault(r['clip'], dict(clip=r['clip'], label=r['label'], label_source='human',
                                        weight=1., frames=r['frames'][:120]))
    corrections = {x.stem: x.read_text().strip() for x in (OUTPUTS / 'corrected_clips').rglob('*.txt')}
    predictions, reviewed = {}, {}
    for path in sorted(args.events.glob('*.json')):
        r = read(path)
        if 'unifi_data2' not in Path(r.get('clip_path') or '').parts:
            continue
        clip = Path(r['clip_name']).stem
        if r.get('pred_label') in LABELS:
            predictions[clip] = r
        if r.get('true_label') in LABELS and r.get('labeled_at'):
            reviewed[clip] = r['true_label']
    reviewed.update(corrections)
    print(f'{len(rows)} cached labeled clips, {len(predictions)} saved predictions, {len(corrections)} local corrections', flush=True)
    if args.inventory:
        print('Preserved split counts', dict(Counter(original_splits.values())))
        return
    if not predictions:
        p.error('No saved unifi_data2 predictions found in --events')
    if (args.run_dir / 'checkpoints').exists():
        p.error('Choose a fresh --run-dir for training')
    videos = {}
    for root in (args.old, args.new):
        # Check access explicitly so a missing mount cannot silently yield an incomplete run.
        list(root.iterdir())
        for video in sorted(root.rglob('*.mp4')):
            if video.stem in videos:
                raise ValueError(f'Duplicate clip ID: {video.stem}')
            videos[video.stem] = video
            existing = rows.get(video.stem)
            try:
                sidecar = video.with_suffix('.txt').read_text().strip()
            except FileNotFoundError:
                sidecar = ''
            if sidecar and sidecar not in LABELS:
                raise ValueError(f'Unknown label in {video.with_suffix(".txt")}: {sidecar!r}')
            human = reviewed.get(video.stem) or sidecar or (existing and existing['label'])
            prediction = predictions.get(video.stem) if root == args.new else None
            if not human and not prediction:
                continue
            rows[video.stem] = dict(clip=video.stem, label=human or prediction['pred_label'],
                label_source='human' if human else 'pseudo', weight=1. if human else .25,
                frames=existing['frames'] if existing else [],
                prediction_confidence=prediction.get('confidence') if not human else None)
    for clip, label in reviewed.items():
        if clip in rows:
            rows[clip].update(label=label, label_source='human', weight=1.)
    missing = set(predictions) - set(videos) - set(rows)
    if missing:
        raise SystemExit(f'{len(missing)} saved predictions have no matching video or cached clip. Check --new.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    os.environ['PATH'] = '/opt/homebrew/bin:' + os.environ['PATH']

    def extract(row):
        if row['frames'] and all(Path(x).is_file() for x in row['frames']):
            return
        video = videos[row['clip']]
        cache = args.run_dir / 'frames' / row['clip']
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache / '.complete'
        signature = f'{video.resolve()}:{video.stat().st_size}:{video.stat().st_mtime_ns}'
        if marker.is_file() and marker.read_text() == signature:
            row['frames'] = [str(x) for x in sorted(cache.glob('*.jpg'))]
            if row['frames']:
                return
        marker.unlink(missing_ok=True)
        for stale in cache.glob('*.jpg'):
            stale.unlink()
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', '1', '-i', str(video),
                '-vf', 'fps=2,scale=256:256', '-frames:v', '120', '-q:v', '4',
                str(Path(temp) / '%04d.jpg')], check=True)
            frames = sorted(Path(temp).glob('*.jpg'))
            if not frames:
                raise ValueError(f'No frames decoded from {video}')
            row['frames'] = [str(x.rename(cache / x.name)) for x in frames]
            marker.write_text(signature)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(extract, rows.values()), 1):
            if i % 100 == 0:
                print(f'Frames ready for {i}/{len(rows)} clips', flush=True)
    losses = read(eval_dir / 'lost_attention/attention.json')['records']
    partitions = {name: [] for name in ('train', 'val', 'test')}
    for clip, row in rows.items():
        partitions[original_splits.get(clip, 'train')].append(row)
    ids = {name: {r['clip'] for r in items} for name, items in partitions.items()}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test'])
    manifest = dict(rows=partitions['train'], validation=partitions['val'], test=partitions['test'],
                    source_split=str(dataset / 'split.json'), retest={
        'lost_five': [rows[x['clip']] for x in losses],
        'reviewed_corrections': [rows[x] for x in sorted(corrections) if x in rows]})
    counts = Counter(row['label_source'] for row in rows.values())
    if not counts['pseudo']:
        raise SystemExit('No pseudo-labeled clips found. Check the new data and prediction paths.')
    (args.run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(dict(counts), flush=True)
    if not args.prepare_only:
        from training import train
        train(args, manifest)


if __name__ == '__main__':
    main()
