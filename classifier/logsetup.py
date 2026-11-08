#!/usr/bin/env python3
"""
Timestamped, unbuffered console output.

Wraps stdout/stderr rather than converting every print() call site, so existing
logging keeps working and gains timestamps for free.

Unbuffering matters as much as the timestamps: when these servers run under
nohup, Python block-buffers stdout, and the log file sits empty for ages —
which made a live server look silent when it was actually working.
"""

from __future__ import annotations

import sys
import time


class _Stamped:
    def __init__(self, stream):
        self._s = stream
        self._at_line_start = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        stamp = time.strftime("%Y-%m-%d %H:%M:%S ")
        out = []
        for ch in text:
            if self._at_line_start and ch != "\n":
                out.append(stamp)
                self._at_line_start = False
            out.append(ch)
            if ch == "\n":
                self._at_line_start = True
        n = self._s.write("".join(out))
        self._s.flush()          # unbuffered: nohup logs stay current
        return n

    def flush(self):
        self._s.flush()

    def isatty(self):
        return self._s.isatty()

    def fileno(self):
        return self._s.fileno()

    def __getattr__(self, name):
        return getattr(self._s, name)


_installed = False


def install() -> None:
    global _installed
    if _installed:
        return
    sys.stdout = _Stamped(sys.stdout)
    sys.stderr = _Stamped(sys.stderr)
    _installed = True
