"""Fine-tune the existing classifier and score the requested training clips."""
import json
import random
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


def train(args, manifest):
    sys.path.insert(0, str(args.model_dir))
    import model
    from mlx.utils import tree_flatten, tree_unflatten
    from infer import Classifier
    from video_dataset import sample_clip_frames

    random.seed(args.seed)
    mx.random.seed(args.seed)
    create = model.create_model
    def initialize(*a, **kw):
        kw['weights'] = False
        return create(*a, **kw)
    model.create_model = initialize
    try:
        classifier = Classifier(ckpt_dir=str(args.teacher.resolve().parent), ckpt=args.teacher.name)
    finally:
        model.create_model = create
    net = classifier.model
    net.load_weights(str(args.teacher), strict=True)
    labels = classifier.labels
    rows = list(manifest['rows'])

    def loss(net, x, target, weight):
        probs = net(x)
        selected = mx.take_along_axis(probs, target[:, None], axis=1)
        return (-mx.log(selected[:, 0] + 1e-8) * weight).mean()

    gradient = nn.value_and_grad(net, loss)
    optimizer = optim.Adam(learning_rate=args.lr)

    def frames(row, training):
        return sample_clip_frames([Path(x) for x in row['frames']], 24, train=training)

    def evaluate(pool, cohorts):
        net.eval()
        result = {}
        for cohort, selected in cohorts.items():
            records = []
            for start in range(0, len(selected), args.batch_size):
                batch = selected[start:start + args.batch_size]
                x = mx.array(np.stack(list(pool.map(lambda row: frames(row, False), batch))))
                for row, probs in zip(batch, np.asarray(net(x))):
                    idx = int(probs.argmax())
                    records.append(dict(clip=row['clip'], label=row['label'],
                                        prediction=labels[idx], confidence=float(probs[idx])))
            correct = sum(r['label'] == r['prediction'] for r in records)
            present = sorted({r['label'] for r in records})
            recalls = [sum(r['label'] == label and r['prediction'] == label for r in records) /
                       sum(r['label'] == label for r in records) for label in present]
            result[cohort] = dict(correct=correct, total=len(records),
                                  accuracy=correct / len(records) if records else 0.,
                                  balanced_accuracy=float(np.mean(recalls)) if recalls else 0.,
                                  records=records)
        return result

    logger = None
    if args.wandb_mode != 'disabled':
        try:
            import wandb
            logger = wandb.init(project=args.wandb_project, mode=args.wandb_mode,
                                dir=str(args.run_dir))
        except Exception as exc:
            print(f'W&B unavailable: {exc}', flush=True)

    destination = args.run_dir / 'checkpoints/all_data'
    destination.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'labels.txt').write_text('\n'.join(labels) + '\n')
    history = []
    best_score = -1.
    step = 0
    start_epoch = 1
    state_path = args.run_dir / 'train_state.json'
    manifest_hash = hashlib.sha256((args.run_dir / 'manifest.json').read_bytes()).hexdigest()
    if getattr(args, 'resume', False):
        state = json.loads(state_path.read_text())
        if state['manifest_sha256'] != manifest_hash:
            raise ValueError('Training manifest changed since checkpoint.')
        net.load_weights(str(destination / state['weights']), strict=True)
        optimizer.state = tree_unflatten(list(mx.load(str(destination / state['optimizer'])).items()))
        mx.eval(net.parameters(), optimizer.state)
        start_epoch = state['epoch'] + 1
        step = state['step']
        best_score = state['best_score']
        history = json.loads((args.run_dir / 'validation_history.json').read_text())
    elif state_path.exists():
        raise ValueError('Existing training state. Use --resume or a new run directory.')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        before = evaluate(pool, manifest['retest'])
        for epoch in range(start_epoch, args.epochs + 1):
            net.train()
            random.seed(args.seed + epoch)
            mx.random.seed(args.seed + epoch)
            rows.sort(key=lambda row: row['clip'])
            random.shuffle(rows)
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start + args.batch_size]
                x = mx.array(np.stack(list(pool.map(lambda row: frames(row, True), batch))))
                y = mx.array([labels.index(row['label']) for row in batch])
                weights = mx.array([row['weight'] for row in batch])
                value, grads = gradient(net, x, y, weights)
                optimizer.update(net, grads)
                mx.eval(net.parameters(), optimizer.state, value)
                step += 1
                if step % 10 == 0 or step == 1:
                    print(f'Epoch {epoch}/{args.epochs}, step {step}, loss {value.item():.4f}', flush=True)
                if logger:
                    logger.log({'train/loss': value.item(), 'epoch': epoch}, step=step)
                if args.max_steps and step >= args.max_steps:
                    break
            validation = evaluate(pool, {'validation': manifest['validation']})['validation']
            score = validation['balanced_accuracy']
            if score > best_score:
                best_score = score
                temporary_best = args.run_dir / '.saving_best.npz'
                net.save_weights(str(temporary_best))
                temporary_best.replace(destination / 'best.npz')
            history.append(dict(epoch=epoch, validation=validation))
            (args.run_dir / 'validation_history.json').write_text(json.dumps(history, indent=2) + '\n')
            print(f"Epoch {epoch} validation {validation['correct']}/{validation['total']}, balanced accuracy {score:.3f}", flush=True)
            if logger:
                logger.log({'val/accuracy': validation['accuracy'], 'val/balanced_accuracy': score, 'epoch': epoch}, step=step)
            epoch_directory = args.run_dir / 'checkpoints/all_data'
            epoch_directory.mkdir(parents=True, exist_ok=True)
            (args.run_dir / 'labels.txt').write_text('\n'.join(labels) + '\n')
            temporary = args.run_dir / '.saving_epoch.npz'
            net.save_weights(str(temporary))
            temporary.replace(epoch_directory / f'epoch_{epoch:03d}.npz')
            temporary = args.run_dir / '.saving_last.npz'
            net.save_weights(str(temporary))
            temporary.replace(epoch_directory / 'last.npz')
            optimizer_name = f'optimizer_{epoch:03d}.npz'
            temporary = args.run_dir / '.saving_optimizer.npz'
            mx.savez(str(temporary), **dict(tree_flatten(optimizer.state)))
            temporary.replace(epoch_directory / optimizer_name)
            state = dict(epoch=epoch, step=step, best_score=best_score,
                         weights=f'epoch_{epoch:03d}.npz', optimizer=optimizer_name,
                         manifest_sha256=manifest_hash)
            temporary_state = args.run_dir / '.train_state.tmp'
            temporary_state.write_text(json.dumps(state, indent=2) + '\n')
            temporary_state.replace(state_path)
            print(f'Saved epoch_{epoch:03d}.npz and resumable optimizer state', flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        net.load_weights(str(destination / 'best.npz'), strict=True)
        after = evaluate(pool, manifest['retest'])
        test = evaluate(pool, {'test': manifest['test']})['test']
        (args.run_dir / 'test_results.json').write_text(json.dumps(test, indent=2) + '\n')
        print(f"Best checkpoint test {test['correct']}/{test['total']}, balanced accuracy {test['balanced_accuracy']:.3f}", flush=True)
    (args.run_dir / 'labels.txt').write_text('\n'.join(labels) + '\n')
    report = dict(scope='Diagnostic recheck with per-clip split recorded in manifest', before=before, after=after,
                  smoke_run=bool(args.max_steps), steps=step)
    (args.run_dir / 'training_recheck.json').write_text(json.dumps(report, indent=2) + '\n')
    for cohort, score in after.items():
        print(f"{cohort}: {before[cohort]['correct']}/{score['total']} → {score['correct']}/{score['total']} (training recheck)")
    if logger:
        logger.finish()
