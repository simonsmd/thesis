"""Run several dialect sniffers and keep each one's verdict verbatim.

No normalisation into a common schema. The tools model different things --
CleverCSV returns delimiter/quote/escape, DuckDB adds terminator, comment,
skip-rows and header, `custom` adds quote style and the escape flags -- and
folding them together at collection time destroys information that cannot be
recovered. Merging is cheap to do later against stored output; re-running a
sniffer over millions of files is not. (`dialects` does the merging for a
downloaded dataset.)

Each sniffer writes under its own key, so a record looks like:

    {"custom": {...}, "clevercsv": {...}, "duckdb": {...},
     "charset_normalizer": {...}, "chardet": {...}}

Two DuckDB quirks worth knowing:

  * It must be given `ignore_errors=true`. Its defaults are tuned for
    *loading* data, so the truncated final row every head sample has makes it
    fall back to reporting `|` with one column.
  * It still answers `|` on some files that contain no pipe byte -- a give-up
    sentinel rather than a detection. That verdict is recorded verbatim all
    the same, with a `head_has_pipe` flag beside it so the sentinel case can
    be filtered out later.
"""
from __future__ import annotations

import os
import signal
import tempfile
import threading
import warnings
from pathlib import Path
from typing import Any

from . import config

warnings.filterwarnings("ignore")

ALL = ("custom", "clevercsv", "duckdb", "charset_normalizer", "chardet")
DEFAULT = ALL


def _custom(head: bytes, size: int | None) -> dict[str, Any]:
    from . import features

    return features.describe(head, size=size).as_dict()


def _clevercsv(head: bytes, size: int | None) -> dict[str, Any]:
    import clevercsv

    text = head.decode("utf-8", "replace")[:8192]
    try:
        d = clevercsv.Sniffer().sniff(text)
    except Exception as exc:
        return {"error": type(exc).__name__}
    if d is None:
        return {"error": "no_dialect"}
    # Verbatim: SimpleDialect exposes exactly these, and `strict` is a
    # constructor default rather than a detection result.
    return {"delimiter": d.delimiter, "quotechar": d.quotechar,
            "escapechar": d.escapechar, "strict": d.strict}


_duckdb_local = threading.local()


def _duckdb_conn():
    """One connection per worker thread.

    `duckdb.sql()` runs on a module-level default connection and returns a
    lazily-evaluated relation, so concurrent callers sharing it can have one
    file's result attributed to another. Per-thread connections are DuckDB's
    own answer to this.
    """
    conn = getattr(_duckdb_local, "conn", None)
    if conn is None:
        import duckdb

        # DuckDB's defaults assume it is the only thing on the machine: a
        # memory limit of 80% of RAM and a thread per core, per connection --
        # and there is one connection in each of `--sniff-procs` processes.
        # One thread, because the processes already run in parallel and a
        # `sniff_csv` over a head does not want a thread pool. The memory
        # limit is a guard rail (see config.SNIFF_DUCKDB_MEMORY), and
        # `temp_directory` is named because it defaults to `.tmp` in the
        # working directory.
        conn = _duckdb_local.conn = duckdb.connect(config={
            "memory_limit": config.SNIFF_DUCKDB_MEMORY,
            "threads": 1,
            "temp_directory": tempfile.gettempdir(),
        })
    return conn


def duckdb_path(path: str | Path, has_pipe: bool | None = None) -> dict[str, Any]:
    """`sniff_csv` against a file already on disk.

    The probe pass spills its in-memory head to a temp file for this, since
    DuckDB takes a path and not bytes. A downloaded corpus has the real file,
    which saves the copy and removes the truncated last row. It does not let
    DuckDB see further into the file: dialect detection reads only the first
    buffer, whatever `sample_size` says.
    """
    try:
        cur = _duckdb_conn().execute(
            "SELECT Delimiter, Quote, Escape, NewLineDelimiter, Comment, "
            "SkipRows, HasHeader, len(Columns) AS n_columns "
            "FROM sniff_csv(?, ignore_errors=true)",
            [str(path)],
        )
        names = [d[0] for d in cur.description]
        row = cur.fetchone()
        if row is None:
            return {"error": "no_row"}
        out = dict(zip(names, row))
        # Recorded verbatim, including a `|` on a file with no pipe byte;
        # `head_has_pipe` carries what is needed to tell the sentinel apart.
        if has_pipe is not None:
            out["head_has_pipe"] = has_pipe
        return out
    except Exception as exc:
        return {"error": type(exc).__name__}


def _duckdb(head: bytes, size: int | None) -> dict[str, Any]:
    fh = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        fh.write(head)
        fh.close()
        return duckdb_path(fh.name, has_pipe=b"|" in head)
    finally:
        Path(fh.name).unlink(missing_ok=True)


def _chardet(head: bytes, size: int | None) -> dict[str, Any]:
    """A second opinion on the encoding, from a different detector.

    `custom` already carries charset_normalizer's verdict via
    `validate.probe_encoding`; where two independent detectors disagree is
    where the file is genuinely ambiguous.
    """
    import chardet

    res = chardet.detect(head)
    return {"encoding": res.get("encoding"), "confidence": res.get("confidence"),
            "language": res.get("language")}


def _charset_normalizer(head: bytes, size: int | None) -> dict[str, Any]:
    """The primary detector's own verdict, recorded verbatim.

    `custom.encoding` is this run through `canonical_encoding`; this keeps the
    raw name, the language guess and the multi-byte ratio alongside it.
    """
    from charset_normalizer import from_bytes

    best = from_bytes(head).best()
    if best is None:
        return {"encoding": None}
    return {"encoding": best.encoding, "language": best.language,
            "bom": best.bom, "multi_byte_usage": round(best.multi_byte_usage, 4)}


_RUNNERS = {"custom": _custom, "clevercsv": _clevercsv, "duckdb": _duckdb,
            "charset_normalizer": _charset_normalizer, "chardet": _chardet}


def run(head: bytes, size: int | None = None,
        which: tuple[str, ...] = DEFAULT) -> dict[str, Any]:
    """Every requested sniffer's raw output, keyed by sniffer name."""
    out: dict[str, Any] = {}
    for name in which:
        runner = _RUNNERS.get(name)
        if runner is None:
            continue
        try:
            out[name] = runner(head, size)
        except Exception as exc:  # a sniffer must never kill the pass
            out[name] = {"error": type(exc).__name__}
    return out


def worker_init(which: tuple[str, ...] = DEFAULT) -> None:
    """Prepare a sniffing subprocess. Runs once per worker, at pool startup.

    SIGINT is ignored so that a terminal Ctrl-C reaches only the parent. Ctrl-C
    goes to the whole foreground process group, and a worker that took it would
    raise out of whatever file it was sniffing; ignoring it lets the parent stop
    drawing work, collect what is in flight, and seal the log.

    Stdout is sent to /dev/null because DuckDB draws a progress bar on it for
    any query past `progress_bar_time` when stdout is a terminal, straight from
    C++ (so `redirect_stdout` cannot catch it) and interleaved with the pass's
    own status line. `enable_progress_bar` did not reliably suppress it. A
    worker's results go back over the pool's pipe, so stdout carries nothing
    else; stderr is left alone so tracebacks still reach the terminal.

    Running a tiny sample through `run` warms the imports (about a second for
    DuckDB and CleverCSV) and opens this process's DuckDB connection while the
    pool is starting, rather than on each worker's first real file.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
    finally:
        os.close(devnull)
    run(b"a,b\n1,2\n", size=8, which=which)


def verdict(record: dict[str, Any], name: str) -> dict[str, Any]:
    """One sniffer's output from a record, or an empty dict if it did not run."""
    return record.get(name) or {}


def delimiters(record: dict[str, Any]) -> dict[str, str | None]:
    """Each sniffer's delimiter verdict, for agreement counting."""
    return {
        "custom": verdict(record, "custom").get("delimiter"),
        "clevercsv": verdict(record, "clevercsv").get("delimiter"),
        "duckdb": verdict(record, "duckdb").get("Delimiter"),
    }


def agreement(record: dict[str, Any]) -> int:
    """How many sniffers back the most popular delimiter (0 if none ran)."""
    votes = [v for v in delimiters(record).values() if v is not None]
    if not votes:
        return 0
    return max(votes.count(v) for v in set(votes))
