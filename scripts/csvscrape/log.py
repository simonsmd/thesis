"""Append-only JSONL logs: the source of truth for everything scraped.

There is deliberately no database in the ingest path. Every fact recorded here
is immutable (a repo's tree at a point in time, a resolved LFS pointer, a fetch
result), so the only durability requirement is "append a line and don't lose
it". Mutable state -- which repos are still pending -- is derived by set
difference at startup instead of being stored.

Shards are written straight into zstd rather than as plain JSONL that a later
pass recompresses. A frame boundary is forced every FLUSH_RECORDS records, so
the file on disk is a sequence of complete frames: bounded writer memory, one
write instead of two, and a killed process costs at most the records since the
last frame instead of the whole shard.

Crash semantics: an unclean exit loses everything written since the last frame
flush, and can leave a torn final frame. Readers decode what they can and drop
a trailing partial line, so a torn shard costs its tail, never the whole file.
Callers that need a specific record durable before writing elsewhere call
`flush()`, which closes the current frame. Retries may produce duplicate
records; readers dedupe rather than the writer preventing it -- `rec_id` gives
every record a stable identity for exactly that.
"""
from __future__ import annotations

import errno
import fcntl
import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import zstandard

from . import config

if TYPE_CHECKING:
    import polars as pl

# An open shard carries the suffix and is renamed on close. Readers that
# tolerate a torn tail (ours) see both; Polars only ever sees sealed files,
# because a zstd stream caught mid-frame is not merely truncated -- it raises
# `OSError: incomplete frame` and takes the whole scan down.
ACTIVE_SUFFIX = ".active"

_PART_RE = re.compile(r"part-(\d{5})\.jsonl(\.zst)?(\.active)?$")

# Records between forced frame boundaries. Every flush costs compression ratio
# (the window restarts), so this trades a little size for a bounded crash
# window: at ~200 records a killed scan loses seconds of work, not a shard.
FLUSH_RECORDS = 200


def log_dir(name: str) -> Path:
    d = config.DATA_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parts(name: str, sealed_only: bool = False) -> list[Path]:
    """Shard files, oldest first. `sealed_only` excludes the shard being written."""
    found = [p for p in log_dir(name).iterdir() if _PART_RE.search(p.name)]
    if sealed_only:
        found = [p for p in found if not p.name.endswith(ACTIVE_SUFFIX)]
    return sorted(found)


def _next_index(name: str) -> int:
    indices = [int(_PART_RE.search(p.name).group(1)) for p in _parts(name)]
    return max(indices, default=-1) + 1


def _read_plain(path: Path) -> Iterator[bytes]:
    """Legacy shards from before writes went straight to zstd."""
    with path.open("rb") as fh:
        for line in fh:
            if not line.endswith(b"\n"):
                break  # torn final line from an unclean exit
            yield line


def _read_zst(path: Path) -> Iterator[bytes]:
    """Decode every frame in a shard, discarding a torn trailing line.

    `read_across_frames` is required, not optional: without it the reader stops
    at the first frame boundary and silently returns only the records written
    before the first flush.
    """
    dctx = zstandard.ZstdDecompressor()
    buf = b""
    with path.open("rb") as raw:
        reader = dctx.stream_reader(raw, read_across_frames=True)
        while True:
            try:
                chunk = reader.read(1 << 20)
            except zstandard.ZstdError:
                break  # torn final frame; keep whatever decoded cleanly
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                if line:
                    yield line
    # Anything left in `buf` has no terminating newline, so it is a torn record
    # rather than a short one. Dropping it is the same rule `_read_plain` uses.


def _is_zst(path: Path) -> bool:
    """Compression is decided by the .jsonl.zst stem, not the final suffix,
    which is `.active` on an open shard."""
    return ".jsonl.zst" in path.name


def raw_lines(name: str) -> Iterator[bytes]:
    """Undecoded records, oldest shard first."""
    for part in _parts(name):
        reader = _read_zst if _is_zst(part) else _read_plain
        try:
            yield from reader(part)
        except zstandard.ZstdError:
            continue  # a shard damaged beyond the torn-line case; skip it


def read(name: str) -> Iterator[dict[str, Any]]:
    """Every record ever written to this log, oldest shard first."""
    for line in raw_lines(name):
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _scalar_pattern(field: str) -> re.Pattern[bytes]:
    """Matches `"field": <scalar>` in a raw record, capturing the value."""
    return re.compile(
        rb'"' + re.escape(field.encode()) + rb'":\s*'
        rb'(-?\d+(?:\.\d+)?|"(?:[^"\\]|\\.)*"|true|false|null)'
    )


def scan_field(name: str, field: str) -> Iterator[Any]:
    """One top-level scalar out of each record, without parsing the rest.

    `json.loads` on a trees record materialises its path/sha/size arrays --
    up to MAX_BLOBS_PER_REPO strings apiece -- when the caller wants a single
    integer, and startup questions like "which repos are scanned" would
    otherwise decode the entire log.

    Only valid for a field that appears once, at the top level, with a scalar
    value; `repo_id` and `content_key` qualify, a nested key would not.
    """
    pattern = _scalar_pattern(field)
    for line in raw_lines(name):
        match = pattern.search(line)
        if match:
            yield json.loads(match.group(1))


def sealed(name: str) -> list[str]:
    """Shards safe for a Parquet/Arrow reader: closed, whole-frame, complete.

    Deliberately excludes the shard currently being appended to. A reader that
    stops at a torn tail could include it; Polars cannot -- it raises on an
    incomplete zstd frame and fails the entire scan. Records in the open shard
    become visible on the next rotation, which is what `compact` picks up.
    """
    return [str(p) for p in _parts(name, sealed_only=True)]


def scan_fields(name: str, fields: tuple[str, ...]) -> Iterator[tuple[Any, ...]]:
    """Several top-level scalars from each record, without parsing the rest.

    See `scan_field`. Fields absent from a record come back as None.
    """
    patterns = [_scalar_pattern(field) for field in fields]
    for line in raw_lines(name):
        out = []
        for pattern in patterns:
            match = pattern.search(line)
            out.append(json.loads(match.group(1)) if match else None)
        yield tuple(out)


def newest_per(frame: "pl.LazyFrame", subset: list[str]) -> "pl.LazyFrame":
    """Keep the newest record per key, ordered by recorded fields.

    `unique(keep="last")` alone means last *in scan order*, which is a property
    of how the files happened to be read rather than a fact in the data.

    `ts` is written by every producer and `rec_id` by the writer, so sorting on
    the pair is exact -- rec_id is monotonic within a shard and breaks
    same-second ties. Records without them fall back to scan order.
    """
    names = frame.collect_schema().names()
    order = [c for c in ("ts", "rec_id") if c in names]
    if order:
        frame = frame.sort(order)
    return frame.unique(subset=subset, keep="last", maintain_order=True)


def _is_live(path: Path) -> bool:
    """Whether a live `Writer` somewhere still holds this shard open.

    An advisory lock, taken by `Writer._open` and released only when the file
    descriptor goes away, is the one signal that survives the case that matters:
    the holder is a *different process*, so no in-process bookkeeping can see
    it, and it may die at any moment without getting to clean up. The kernel
    drops the lock on exit either way, so a crashed writer's shard is still
    sealed by the next run and a live writer's is never touched.

    Anything that cannot report an answer -- a filesystem without flock, a
    file that vanished between the listing and the open -- is reported as not
    live. A shard is much likelier to be an orphan than to be under a writer
    this process cannot see, and treating a stale file as live would leave it
    unreadable forever.
    """
    try:
        handle = path.open("rb")
    except OSError:
        return False
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        # Only contention means a writer holds it. Anything else (ENOLCK on a
        # filesystem without locks) is "cannot tell".
        return exc.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)
    finally:
        handle.close()       # drops the probe's own lock immediately
    return False


def seal_orphans(name: str) -> int:
    """Make shards left open by a killed process readable again.

    A crash leaves an `.active` file whose last frame is torn. Renaming it as
    is would hand Polars the very thing it cannot read, so it is decoded with
    the tolerant reader and rewritten as whole frames -- the torn tail is
    dropped, which is the same record loss the crash already caused.

    Shards a live writer still holds (see `_is_live`) are skipped: sealing one
    would drop its records since the last frame boundary and unlink the file
    out from under the writer.
    """
    sealed_count = 0
    for path in log_dir(name).iterdir():
        if not path.name.endswith(ACTIVE_SUFFIX):
            continue
        if _is_live(path):
            continue
        final = path.with_name(path.name[: -len(ACTIVE_SUFFIX)])
        lines = [line for line in _read_zst(path)]
        if lines:
            tmp = path.with_name(path.name + ".sealing")
            cctx = zstandard.ZstdCompressor(level=10)
            with tmp.open("wb") as raw, cctx.stream_writer(raw) as out:
                for line in lines:
                    out.write(line + b"\n")
            tmp.replace(final)
            sealed_count += 1
        path.unlink(missing_ok=True)
    return sealed_count


class Writer:
    """Thread-safe appender that rotates and compresses shards as it goes."""

    def __init__(self, name: str, rotate: int = config.ROTATE_RECORDS,
                 flush_records: int = FLUSH_RECORDS):
        self.name = name
        self.rotate = rotate
        self.flush_records = flush_records
        self._lock = threading.Lock()
        seal_orphans(name)
        self._index = _next_index(name)
        self._cctx = zstandard.ZstdCompressor(level=10)
        self._count = 0
        self._since_flush = 0
        self._seq = 0
        self._closed = False
        self._open()

    def _open(self) -> None:
        # Created exclusively, stepping past any index that is taken -- by a
        # sealed shard, or by another process's open one. `_roll` cannot see
        # what other writers have created, and opening an existing shard would
        # truncate it (or the rename on close would overwrite it).
        while True:
            self._final = log_dir(self.name) / f"part-{self._index:05d}.jsonl.zst"
            self._path = self._final.with_name(self._final.name + ACTIVE_SUFFIX)
            if not self._final.exists():
                try:
                    self._raw = self._path.open("xb")
                    break
                except FileExistsError:
                    pass
            self._index += 1
        # Held for the life of the shard and released by the kernel when this
        # descriptor goes away, however the process ends. `seal_orphans` in
        # any other run tests it before touching a shard, which is what stops
        # a concurrent writer from sealing this one out from under us. A
        # filesystem without flock support goes unlocked.
        try:
            fcntl.flock(self._raw, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            pass
        self._out = self._cctx.stream_writer(self._raw)
        self._count = 0
        self._since_flush = 0
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            # Stable identity for a record, assigned once and never reused:
            # readers dedupe retried work on this instead of on row position,
            # which is a property of how a frame happens to be scanned.
            record = {"rec_id": f"{self._index:05d}:{self._seq:08d}", **record}
            self._seq += 1
            self._out.write((json.dumps(record, separators=(",", ":")) + "\n").encode())
            self._count += 1
            self._since_flush += 1
            if self._count >= self.rotate:
                self._roll()
            elif self._since_flush >= self.flush_records:
                self._out.flush(zstandard.FLUSH_FRAME)
                self._since_flush = 0

    def _roll(self) -> None:
        self._close_shard()
        self._index += 1
        self._seq = 0
        self._open()

    def _close_shard(self) -> None:
        if self._closed:
            # The file has already been renamed; a second rename would raise
            # out of `__exit__` and mask whatever the caller was reporting.
            return
        self._closed = True
        self._out.close()
        self._raw.close()
        if self._count:
            # Sealed only now: every frame is complete, so a Parquet reader can
            # take it. Rename is atomic, so a reader never sees a half-file.
            self._path.replace(self._final)
        if not self._count:
            # Polars' NDJSON reader refuses an empty file ("cannot infer types
            # on empty reader"), which would break every scan over this log.
            self._path.unlink(missing_ok=True)

    def flush(self) -> None:
        """Close the current frame, making every record so far readable."""
        with self._lock:
            self._out.flush(zstandard.FLUSH_FRAME)
            self._raw.flush()
            self._since_flush = 0

    def close(self) -> None:
        with self._lock:
            self._close_shard()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
