"""The append-only log, and what writing to it actually costs.

One ``os.write`` of one line to one file descriptor held open with ``O_APPEND``.
No buffering layer, no fsync, no formatting beyond ``json.dumps`` — because the
brief's requirement is that a *crash* cannot lose the row, and a completed
``write(2)`` puts the bytes in the kernel's page cache, which survives the process
dying. It does not survive the machine losing power; ``sync_every`` exists for a
caller who wants that trade and the benchmark measures what it costs.

A logging failure must not fail the request: ``OSError`` is counted, warned about
once per distinct message, and swallowed. A full disk is a reason to lose a log
row, not to stop answering the operator.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, Mapping

from ..data import REPO_ROOT
from .records import EffectRecord, RequestRecord, to_line

LOG_NAME = "requests.jsonl"
EFFECT_NAME = "effects.jsonl"
FILE_MODE = 0o600


def collect_root(explicit: str | Path | None = None) -> Path:
    """Where personal data lives: ``$LOCALALEXA_COLLECT``, else ``<repo>/var/collect``.

    Deliberately outside ``data/``: ``data/`` is shipped with the repo and holds
    no utterances the router has actually seen.
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("LOCALALEXA_COLLECT", "").strip()
    return Path(env) if env else REPO_ROOT / "var" / "collect"


def new_session() -> str:
    """A session id. One process, one session, unless the caller says otherwise."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]


class RequestLog:
    """Append-only JSONL writer for request and effect records."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        session: str | None = None,
        clock=time.time,
        sync_every: int = 0,
    ) -> None:
        if path is not None and str(path).endswith(".jsonl"):
            self.path = Path(path)
            self.root = self.path.parent
        else:
            self.root = collect_root(path)
            self.path = self.root / LOG_NAME
        self.effect_path = self.path.parent / EFFECT_NAME
        self.session = session or new_session()
        self._clock = clock
        self._sync_every = sync_every
        self._writes = 0
        self.errors = 0
        self._warned: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_MODE)
        self._effect_fd = os.open(
            self.effect_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, FILE_MODE
        )

    def append(self, record: RequestRecord) -> None:
        """Write one request row. The only thing the request path does."""
        self.write_line(self._fd, to_line(record.as_dict()))

    def append_effect(self, effect: EffectRecord) -> None:
        self.write_line(self._effect_fd, to_line(effect.as_dict()))

    def write_line(self, fd: int, line: str) -> None:
        """The raw append underneath ``append``: one ``write``, no formatting."""
        try:
            os.write(fd, line.encode("utf-8"))
            self._writes += 1
            if self._sync_every and self._writes % self._sync_every == 0:
                os.fsync(fd)
        except OSError as exc:
            self.errors += 1
            message = f"{type(exc).__name__}: {exc}"
            if message not in self._warned:
                self._warned.add(message)
                print(f"localalexa.collect: log write failed ({message}); continuing", file=sys.stderr)

    def rows(self) -> list[dict]:
        return list(self._iter(self.path))

    def effects(self) -> list[dict]:
        return list(self._iter(self.effect_path))

    def read_from(self, offset: int = 0) -> tuple[list[dict], int]:
        """Rows written since ``offset``, and the offset to pass next time.

        The collector follows the log with this instead of holding the only copy
        of what happened: a second process appending to the same file is picked
        up, and a restart resumes from nothing but the file. Only complete lines
        advance the offset, so a row half-written by a crash is read once the
        write finishes rather than being parsed as garbage.
        """
        return self._read_from(self.path, offset)

    def effects_from(self, offset: int = 0) -> tuple[list[dict], int]:
        return self._read_from(self.effect_path, offset)

    @staticmethod
    def _read_from(path: Path, offset: int) -> tuple[list[dict], int]:
        if not path.is_file():
            return [], offset
        size = path.stat().st_size
        if offset > size:  # the file was rotated or truncated under us
            offset = 0
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()
        end = chunk.rfind(b"\n")
        if end == -1:
            return [], offset
        body = chunk[: end + 1]
        rows = []
        for line in body.splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows, offset + len(body)

    @staticmethod
    def _iter(path: Path) -> Iterator[dict]:
        """Yield parsed rows, skipping any line a crash left half-written."""
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def close(self) -> None:
        for fd in (self._fd, self._effect_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> "RequestLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"RequestLog({str(self.path)!r}, session={self.session!r})"


def write_jsonl(path: str | Path, rows: list[Mapping]) -> Path:
    """Rewrite a derived artefact atomically: temp file, then rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(to_line(row))
    os.replace(tmp, target)
    os.chmod(target, FILE_MODE)
    return target
