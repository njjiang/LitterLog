"""Use the app's configured storage even for direct inference CLI runs."""
import sys
from pathlib import Path

backend = str(Path(__file__).resolve().parents[1])
if backend not in sys.path:
    sys.path.insert(0, backend)

from clip_storage import directory, scratch_parent
