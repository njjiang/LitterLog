"""Keep writable service state separate from packaged application code."""
import os
from pathlib import Path


LEGACY_MODEL_DIR = Path(__file__).resolve().parent.parent / 'classifier'


def _configured_directory(name, default):
    value = os.environ.get(name)
    if value is None:
        return Path(default)
    if not value.strip():
        raise ValueError(f'{name} must name an absolute directory.')
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f'{name} must name an absolute directory.')
    return path


def state_root(default):
    """Use the desktop backend directory unless a persistent volume is selected.

    Resolve on use so callers retain their existing local-root overrides. This
    does not create folders or change media/NAS availability checks.
    """
    return _configured_directory('LITTERLOG_STATE_DIR', default)


def model_directory():
    return _configured_directory('LITTERLOG_MODEL_DIR', LEGACY_MODEL_DIR)
