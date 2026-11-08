"""Persistent classifier source clips, separate from full playback exports."""
import hashlib
import os
import tempfile
from pathlib import Path

import av

from clip_storage import StorageUnavailable, directory

MAX_SCRATCH_CLIP_BYTES = 96 * 1024**2


def valid_video(path):
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        with av.open(str(path)) as container:
            return bool(container.streams.video)
    except (OSError, ValueError, av.FFmpegError):
        return False


def clip_name(event_id, camera_id, start_ms, end_ms):
    if not isinstance(event_id, str) or not event_id or not isinstance(camera_id, str) or not camera_id:
        raise ValueError('A recording and camera ID are required')
    if type(start_ms) is not int or type(end_ms) is not int or not 0 < start_ms < end_ms:
        raise ValueError('A completed source interval is required')
    key = hashlib.sha256((event_id + '\0' + camera_id).encode()).hexdigest()
    return f'{key}-{start_ms}-{end_ms}.mp4'


def saved_clip(event_id, camera_id, start_ms, end_ms, download):
    """Reuse an exact source interval or atomically save a new clip download."""
    name = clip_name(event_id, camera_id, start_ms, end_ms)
    folder = directory('inference-clips')
    destination = folder / name
    if valid_video(destination):
        return destination
    fd, name = tempfile.mkstemp(prefix='.litterlog-inference-', suffix='.partial', dir=folder)
    os.close(fd)
    partial = Path(name)
    try:
        download(partial)
        if directory('inference-clips') != folder:
            raise StorageUnavailable('Clip storage changed during download; please retry.')
        if not valid_video(partial):
            raise ValueError('Protect did not return a readable video')
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def downloaded_clip(event_id, camera_id, start_ms, end_ms, download_bytes):
    """Adapt the classifier's existing byte-returning Protect client."""
    def write(destination):
        data = download_bytes()
        if not data:
            raise ValueError('Protect returned an empty clip')
        if directory('inference-clips') != destination.parent:
            raise StorageUnavailable('Clip storage changed during download; please retry.')
        destination.write_bytes(data)
    return saved_clip(event_id, camera_id, start_ms, end_ms, write)
