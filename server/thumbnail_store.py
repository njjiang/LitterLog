"""Atomic publication of model attention images shared by inference and repair."""
from calendar import monthrange
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from PIL import Image


MAX_THUMBNAIL_BYTES = 2 * 1024**2
MAX_METADATA_BYTES = 16 * 1024


def retention_cutoff(now=None, months=2):
    if type(months) is not int or months < 1:
        raise ValueError('History retention must be a positive number of months.')
    now = now if now is not None else datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('Timestamp must include a timezone.')
    now = now.astimezone(timezone.utc)
    year, month_index = divmod(now.year * 12 + now.month - 1 - months, 12)
    month = month_index + 1
    return now.replace(year=year, month=month, day=min(now.day, monthrange(year, month)[1]))


def name_for(event_id):
    return hashlib.sha256(event_id.encode()).hexdigest()


def thumbnail_metadata(folder, name):
    try:
        metadata = json.loads((folder / (name + '.json')).read_text())
        if not isinstance(metadata, dict):
            return None
        seconds = metadata.get('seconds')
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            return None
        return metadata
    except (OSError, ValueError, TypeError):
        return None


def validate_jpeg(frame):
    if not frame:
        raise ValueError('Attention frame is empty')
    try:
        with Image.open(io.BytesIO(frame)) as image:
            if image.format != 'JPEG':
                raise ValueError('Attention frame must be JPEG')
            image.verify()
    except (OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError('Attention frame must be a valid JPEG') from error
    return frame


def local_thumbnail_bytes(folder, name):
    path = folder / (name + '.jpg')
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open('rb') as source:
            frame = source.read(MAX_THUMBNAIL_BYTES + 1)
        if len(frame) <= MAX_THUMBNAIL_BYTES:
            return validate_jpeg(frame)
    except (OSError, ValueError):
        return None


def complete_thumbnail(folder, event_id):
    name = name_for(event_id)
    metadata = thumbnail_metadata(folder, name)
    if metadata is None:
        return False
    try:
        with Image.open(folder / (name + '.jpg')) as image:
            image.verify()
        return True
    except (OSError, ValueError, TypeError, SyntaxError, Image.DecompressionBombError):
        return False


def atomic_write(destination, data):
    fd, name = tempfile.mkstemp(prefix='.litterlog-thumbnail-', suffix='.partial', dir=destination.parent)
    partial = Path(name)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


@contextmanager
def thumbnail_publication_lock(folder):
    with (folder / '.thumbnail-publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def purge_legacy_thumbnail_partials(folder, *, now=None, months=2):
    """Expire only recognized abandoned atomic-write files, never live locks."""
    cutoff = retention_cutoff(now=now, months=months).timestamp()
    removed = 0
    with thumbnail_publication_lock(folder):
        for path in folder.glob('.litterlog-thumbnail-*.partial'):
            if (not re.fullmatch(r'\.litterlog-thumbnail-[A-Za-z0-9_-]+\.partial', path.name)
                    or path.is_symlink() or not path.is_file()):
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
    return removed


def publish_thumbnail(folder, event_id, prediction, checkpoint):
    frame = prediction.best_frame
    seconds = prediction.best_frame_seconds
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Attention frame unavailable')
    validate_jpeg(frame)
    name = name_for(event_id)
    metadata = {'seconds': seconds, 'attn_ratio': prediction.attn_ratio, 'checkpoint': checkpoint}
    if complete_thumbnail(folder, event_id):
        return False
    atomic_write(folder / (name + '.json'), json.dumps(metadata, allow_nan=False).encode())
    atomic_write(folder / (name + '.jpg'), frame)
    return True
