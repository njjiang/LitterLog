"""Keep only recent, size-bounded runtime output in the LitterLog state volume."""

from contextlib import ExitStack, contextmanager, suppress
from datetime import date, datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from history_archive import retention_cutoff


ROLES = ('api', 'worker', 'thumbnails')
MAX_BYTES = 30 * 1024**2
SEGMENT_BYTES = 1024**2
READ_BYTES = 64 * 1024
PRUNE_INTERVAL = 60
SEGMENT_NAME = re.compile(r'(\d{4}-\d{2}-\d{2})\.(\d{6})\.log', re.ASCII)
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_FLAGS = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK


@contextmanager
def directory(path, *, parent=None):
    if parent is not None:
        with suppress(FileExistsError):
            os.mkdir(path, 0o700, dir_fd=parent)
    fd = os.open(path, DIRECTORY_FLAGS, dir_fd=parent)
    try:
        yield fd
    finally:
        os.close(fd)


class RetainedLogs:
    def __init__(self, fd, *, max_bytes=MAX_BYTES, segment_bytes=SEGMENT_BYTES):
        if not 0 < segment_bytes <= max_bytes:
            raise ValueError('Log segment size must fit in the retention limit.')
        self.fd = fd
        self.max_bytes = max_bytes
        self.segment_bytes = segment_bytes

    @classmethod
    @contextmanager
    def open(cls, state, role, **limits):
        if role not in ROLES:
            raise ValueError('Choose api, worker, or thumbnails.')
        with ExitStack() as stack:
            root = stack.enter_context(directory(state))
            logs = stack.enter_context(directory('logs', parent=root))
            fd = stack.enter_context(directory(role, parent=logs))
            # Each role has one runtime writer. A replacement starts after it exits.
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield cls(fd, **limits)

    def segments(self):
        segments = []
        for name in sorted(os.listdir(self.fd)):
            match = SEGMENT_NAME.fullmatch(name)
            if match is None:
                continue
            try:
                day = date.fromisoformat(match[1])
                info = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            except (FileNotFoundError, ValueError):
                continue
            if stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                segments.append((name, day, int(match[2]), info.st_size))
        return segments

    def remove(self, name):
        with suppress(FileNotFoundError):
            os.unlink(name, dir_fd=self.fd)

    def trim(self, segments, *, reserve=0, keep=None):
        size = sum(segment[3] for segment in segments)
        kept = []
        for segment in segments:
            name, _, _, length = segment
            if name != keep and size + reserve > self.max_bytes:
                self.remove(name)
                size -= length
            else:
                kept.append(segment)
        return kept

    def prune(self, now=None):
        cutoff = retention_cutoff(now).date()
        recent = []
        for segment in self.segments():
            name, day, _, _ = segment
            if day < cutoff:
                self.remove(name)
            else:
                recent.append(segment)
        return self.trim(recent)

    def create_segment(self, today, start):
        for order in range(start, 1_000_000):
            name = f'{today.isoformat()}.{order:06d}.log'
            try:
                return name, os.open(name, FILE_FLAGS | os.O_CREAT | os.O_EXCL,
                                     0o600, dir_fd=self.fd)
            except FileExistsError:
                continue
        raise OSError('No available LitterLog log segment names for today.')

    def append(self, data, now=None):
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        today = now.date()
        data = memoryview(data)
        while data:
            segments = self.prune(now)
            latest = next((segment for segment in reversed(segments) if segment[1] == today), None)
            if latest is not None and latest[3] < self.segment_bytes:
                name, _, _, length = latest
                fd = os.open(name, FILE_FLAGS, dir_fd=self.fd)
            else:
                name, fd = self.create_segment(today, latest[2] + 1 if latest else 0)
                length = 0
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != length:
                    raise OSError('LitterLog log segment changed while opening it.')
                chunk = data[:self.segment_bytes - length]
                self.trim(segments, reserve=len(chunk), keep=name)
                while chunk:
                    written = os.write(fd, chunk)
                    chunk = chunk[written:]
                    data = data[written:]
            finally:
                os.close(fd)


def run_child(command, logs, *, prune_interval=PRUNE_INTERVAL):
    """Drain merged output and deliver Docker stop signals to the service group."""
    child = None
    pending = []
    previous = {}
    logging_failed = False

    def forward(number, _frame):
        if child is None:
            pending.append(number)
        else:
            with suppress(ProcessLookupError):
                os.killpg(child.pid, number)

    def maintain(operation, *args):
        nonlocal logging_failed
        try:
            operation(*args)
        except OSError:
            if not logging_failed:
                print('LitterLog runtime logs could not be saved.', file=sys.stderr, flush=True)
            logging_failed = True
        else:
            logging_failed = False

    try:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            previous[number] = signal.signal(number, forward)
        logs.prune()
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        for number in pending:
            forward(number, None)
        with child.stdout as output, selectors.DefaultSelector() as selector:
            selector.register(output, selectors.EVENT_READ)
            next_prune = time.monotonic() + prune_interval
            drain_until = None
            while True:
                now = time.monotonic()
                if now >= next_prune:
                    maintain(logs.prune)
                    next_prune = now + prune_interval
                reading = bool(selector.get_map())
                if child.poll() is not None:
                    if not reading:
                        break
                    if drain_until is None:
                        drain_until = now + 1
                    if now >= drain_until:
                        break
                wait = min(next_prune - now, 1)
                if not reading:
                    time.sleep(wait)
                    continue
                if selector.select(wait):
                    chunk = os.read(output.fileno(), READ_BYTES)
                    if chunk:
                        maintain(logs.append, chunk)
                    else:
                        selector.unregister(output)
                elif drain_until is not None:
                    break
        code = child.wait()
        return 128 - code if code < 0 else code
    finally:
        if child is not None and child.poll() is None:
            forward(signal.SIGTERM, None)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                forward(signal.SIGKILL, None)
                child.wait()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    role = args.pop(0) if args else 'api'
    state = os.environ.get('LITTERLOG_STATE_DIR')
    if not state:
        raise SystemExit('LITTERLOG_STATE_DIR must be configured.')
    entrypoint = Path(__file__).with_name('container_entrypoint.py').resolve()
    try:
        with RetainedLogs.open(state, role) as logs:
            return run_child([sys.executable, '-u', str(entrypoint), role, *args], logs)
    except OSError:
        raise SystemExit('LitterLog runtime log directory could not be used.') from None
    except ValueError as error:
        raise SystemExit(str(error)) from None


if __name__ == '__main__':
    raise SystemExit(main())
