"""Repair attention thumbnails; --watch also handles new worker clips.

This process reads predictions but never changes labels, cat assignments, or
manual corrections. Without --watch it performs one bounded scan and exits.
"""
import argparse
import contextlib
import fcntl
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

from clip_storage import StorageUnavailable
from inference_clips import saved_clip
from thumbnail_store import (complete_thumbnail, publish_thumbnail,
                             purge_legacy_thumbnail_partials)
from protect_service import ProtectService
from backend_paths import state_root, model_directory

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'inference'))


def pending_recordings(database, folder):
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True, timeout=10) as connection:
        rows = connection.execute(
            "SELECT r.event_id,r.camera_id,r.start_ms,r.end_ms FROM recordings r "
            "JOIN events e ON r.event_id=e.event_id WHERE e.label != 'recording' "
            "AND r.end_ms>r.start_ms ORDER BY r.start_ms DESC").fetchall()
    return [row for row in rows if not complete_thumbnail(folder, row[0])]


class ThumbnailRepair:
    def __init__(self, database, folder, model_factory, protect, checkpoint='video_24f_balanced', interval=30, limit=0):
        self.database = Path(database)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.model_factory = model_factory
        self.model = None
        self.protect = protect
        self.checkpoint = checkpoint
        self.interval = interval
        self.limit = limit
        self.failures = {}
        self.last_pending = None

    def scan(self, now=None):
        explicit_now = now is not None
        now = time.monotonic() if now is None else now
        purge_legacy_thumbnail_partials(self.folder)
        pending = pending_recordings(self.database, self.folder)
        present = {row[0] for row in pending}
        self.failures = {key: value for key, value in self.failures.items() if key in present}
        eligible = [row for row in pending if self.failures.get(row[0], (0, 0))[1] <= now]
        if self.limit:
            eligible = eligible[:self.limit]
        if len(pending) != self.last_pending:
            print(f'Attention thumbnails pending: {len(pending)}', flush=True)
            self.last_pending = len(pending)
        result = dict(pending=len(pending), saved=0, failed=0, deferred=len(pending) - len(eligible))
        if not eligible:
            return result
        if self.model is None:
            print('Loading CPU model for attention thumbnails…', flush=True)
            self.model = self.model_factory()
        for eid, camera, start, end in eligible:
            try:
                sample_end = min(end, start + 62000)
                clip = saved_clip(eid, camera, start, sample_end,
                                  lambda destination: self.protect.export(eid, camera, start, sample_end, destination))
                prediction = self.model.predict(clip, max_seconds=60, with_attention=True)
                if publish_thumbnail(self.folder, eid, prediction, self.checkpoint):
                    result['saved'] += 1
                    print('Attention thumbnail saved.', flush=True)
                self.failures.pop(eid, None)
            except Exception as error:
                attempts = self.failures.get(eid, (0, 0))[0] + 1
                delay = min(900, self.interval * 2 ** min(attempts - 1, 5))
                failed_at = now if explicit_now else time.monotonic()
                self.failures[eid] = (attempts, failed_at + delay)
                result['failed'] += 1
                if isinstance(error, StorageUnavailable):
                    print('Clip storage unavailable; check configured media root.', flush=True)
                else:
                    print(f'Attention thumbnail failed ({type(error).__name__}); retry deferred for {delay:g}s.', flush=True)
        return result


@contextlib.contextmanager
def worker_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.thumbnail-worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true', help='Keep repairing thumbnails for new events.')
    parser.add_argument('--interval', type=float, default=30, help='Seconds between scans; minimum 5.')
    parser.add_argument('--limit', type=int, default=0, help='Maximum eligible clips per scan; 0 means all.')
    parser.add_argument('--database', type=Path, default=Path(os.environ.get('LITTERLOG_DB', str(state_root(ROOT) / 'events.sqlite3'))))
    parser.add_argument('--model-dir', type=Path, default=model_directory())
    parser.add_argument('--checkpoint', default='video_24f_balanced')
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 5:
        parser.error('--interval must be at least 5 seconds')
    if args.limit < 0:
        parser.error('--limit must be nonnegative')

    def model_factory():
        from cpu_infer import Classifier
        return Classifier(args.model_dir, checkpoint=args.checkpoint)

    folder = state_root(ROOT) / 'attention-cache'
    repair = ThumbnailRepair(args.database, folder, model_factory, ProtectService(), args.checkpoint, args.interval, args.limit)
    try:
        with worker_lock(folder):
            while True:
                try:
                    result = repair.scan()
                except Exception as error:
                    print(f'Thumbnail scan unavailable ({type(error).__name__}). Check the database, checkpoint, and storage.', flush=True)
                    if not args.watch:
                        return 1
                else:
                    if not args.watch:
                        print(f"Thumbnail scan finished: {result['saved']} saved, {result['failed']} failed, {result['deferred']} deferred.", flush=True)
                        return int(result['failed'] > 0)
                time.sleep(args.interval)
    except BlockingIOError:
        print('The thumbnail repair worker is already running.', flush=True)
        return 0
    except KeyboardInterrupt:
        print('Thumbnail repair stopped.', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
