"""Shared local clip storage directory resolution and validation."""
import os
from pathlib import Path
from backend_paths import state_root


ROOT = Path(__file__).resolve().parent
CONFIG_FILE = state_root(ROOT) / 'media-root.txt'


class StorageUnavailable(OSError):
    """Configured clip storage cannot be used."""


def _resolve(path):
    try:
        return path.resolve()
    except (OSError, RuntimeError) as error:
        raise StorageUnavailable('The configured clip-storage folder could not be resolved.') from error


def configured_root():
    """Read the selected media root path."""
    configured = os.environ.get('LITTERLOG_MEDIA_ROOT')
    if configured is None:
        try:
            configured = CONFIG_FILE.read_text().strip()
        except FileNotFoundError:
            return state_root(ROOT)
        except OSError as error:
            raise StorageUnavailable('The saved clip-storage setting could not be read.') from error
    if not configured.strip():
        raise StorageUnavailable('The clip-storage folder is empty. Set LITTERLOG_MEDIA_ROOT or media-root.txt.')
    path = Path(configured.strip()).expanduser()
    if not path.is_absolute():
        raise StorageUnavailable('The clip-storage folder must be an absolute path.')
    return path


def ensure_available(path):
    """Validate a stored path without creating it; return its resolved path."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        raise StorageUnavailable('The clip-storage folder must be an absolute path.')
    return _resolve(path)


def root():
    """Return the configured, validated root without creating any directories."""
    return ensure_available(configured_root())


def scratch_parent():
    """Return the temporary scratch directory inside configured storage."""
    return directory('tmp')


def directory(name):
    """Create/return a child directory inside the configured storage root."""
    child = Path(name)
    if not str(name) or child.is_absolute() or '..' in child.parts or child == Path('.'):
        raise ValueError('Clip-storage directory must be a relative child path.')
    base = root()
    destination = _resolve(base / child)
    if not destination.is_relative_to(base):
        raise StorageUnavailable('The clip-storage subfolder points outside the configured storage.')
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise StorageUnavailable('The configured clip-storage folder could not be opened for writing.') from error
    return destination
