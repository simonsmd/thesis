"""Packed, content-addressed storage for cached file heads.

One file per cached blob is not viable at the scale this corpus reaches: a
pool of millions of files would be millions of inodes. Blobs are therefore
appended into large shards, each compressed individually so a single blob is
still randomly readable by (shard, offset, length) without decompressing its
neighbours.

The store is optional and lives under its own root, so it can sit on a
different filesystem from the logs it belongs to.
"""
from __future__ import annotations

import threading
from pathlib import Path

import zstandard

SHARD_BYTES = 512 * 1024 * 1024  # roll a shard at ~512 MB


class BlobStore:
    def __init__(self, root: Path, shard_bytes: int = SHARD_BYTES, level: int = 10):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.level = level
        # zstandard's compressor/decompressor objects are NOT safe for
        # concurrent use -- sharing one across worker threads segfaults the
        # interpreter rather than raising. One context per thread.
        self._local = threading.local()
        self._lock = threading.Lock()
        self._index = self._next_index()
        self._fh = None
        self._offset = 0

    def _path(self, index: int) -> Path:
        return self.root / f"blobs-{index:05d}.bin"

    def _next_index(self) -> int:
        existing = sorted(self.root.glob("blobs-*.bin"))
        if not existing:
            return 0
        last = int(existing[-1].stem.split("-")[1])
        # Reuse the last shard if it still has room, else start a new one.
        return last if existing[-1].stat().st_size < self.shard_bytes else last + 1

    def _open(self) -> None:
        if self._fh is None:
            path = self._path(self._index)
            self._offset = path.stat().st_size if path.exists() else 0
            self._fh = path.open("ab")

    @property
    def _cctx(self) -> zstandard.ZstdCompressor:
        ctx = getattr(self._local, "cctx", None)
        if ctx is None:
            ctx = self._local.cctx = zstandard.ZstdCompressor(level=self.level)
        return ctx

    @property
    def _dctx(self) -> zstandard.ZstdDecompressor:
        ctx = getattr(self._local, "dctx", None)
        if ctx is None:
            ctx = self._local.dctx = zstandard.ZstdDecompressor()
        return ctx

    def put(self, data: bytes) -> dict[str, int]:
        """Append a blob. Returns the locator to store in the metadata log."""
        frame = self._cctx.compress(data)
        with self._lock:
            self._open()
            if self._offset >= self.shard_bytes:
                self._fh.close()
                self._index += 1
                self._fh = None
                self._open()
            offset = self._offset
            self._fh.write(frame)
            self._offset += len(frame)
            return {"shard": self._index, "off": offset, "len": len(frame)}

    def get(self, locator: dict[str, int]) -> bytes:
        path = self._path(locator["shard"])
        with path.open("rb") as fh:
            fh.seek(locator["off"])
            return self._dctx.decompress(fh.read(locator["len"]))

    def flush(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "BlobStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def bytes_written(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("blobs-*.bin"))
