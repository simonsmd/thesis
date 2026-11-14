"""Run several dialect sniffers and keep each one's verdict verbatim.

No normalisation into a common schema. The tools model different things --
CleverCSV returns delimiter/quote/escape, DuckDB adds terminator, comment,
skip-rows and header, `custom` (this project's own sniffer, see `features`)
adds quote style, the escape flags and statistics about the content -- and
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

Every dialect sniffer reads the same decoding of the head (`Sample`):
CleverCSV the text, DuckDB that text as UTF-8, and `custom` is told the
encoding and converts the head itself. Left to decode on
their own, they disagree over encoding rather than dialect -- CleverCSV takes
text and was handed a UTF-8 decode, which makes U+FFFD (a symbol, so a
delimiter candidate) win on legacy multibyte heads and NULs fail it on
UTF-16; DuckDB reads bytes as UTF-8, so UTF-16 gives it the delimiter but
never the quote, and its own `encoding=` decoders cover too little (strict
latin-1 that rejects 0x80-0x9F, other pages only through an extension).
"""
from __future__ import annotations

import os
import signal
import tempfile
import threading
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

from . import config, validate

warnings.filterwarnings("ignore")

ALL = ("custom", "clevercsv", "duckdb", "charset_normalizer", "chardet")
DEFAULT = ALL

# Classes whose bytes already are UTF-8 (a UTF-8 BOM included, which DuckDB
# skips itself), so DuckDB can be given them untouched.
_UTF8_CLASSES = ("ascii", "utf-8")


class Sample:
    """One head as the sniffers see it, decoded once.

    Lazy, so a run that asks for chardet alone never pays for the encoding
    probe, and only DuckDB pays for the re-encode.
    """

    def __init__(self, head: bytes, size: int | None = None):
        self.head = head
        self.size = size

    @cached_property
    def enc(self) -> validate.Encoding:
        return validate.probe_encoding(self.head)

    @cached_property
    def text(self) -> str:
        return validate.decode(self.head, self.enc)

    @property
    def is_utf8(self) -> bool:
        """Whether the raw bytes are already what a UTF-8 reader should get."""
        return self.enc.encoding_class in _UTF8_CLASSES

    @cached_property
    def utf8(self) -> bytes:
        """`text` as UTF-8, or the head itself where that is the same thing."""
        if self.is_utf8:
            return self.head
        return self.text.encode("utf-8", "replace")


def _custom(s: Sample) -> dict[str, Any]:
    from . import features

    return features.describe(s.head, size=s.size, enc=s.enc).as_dict()


def _clevercsv(s: Sample) -> dict[str, Any]:
    import clevercsv

    text = s.text[:8192]
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
    """`sniff_csv` against a file already on disk, read as UTF-8.

    The probe pass spills its in-memory head to a temp file for this, since
    DuckDB takes a path and not bytes. A downloaded corpus has the real file,
    which saves the copy and removes the truncated last row (`duckdb_file`
    decides when it can be used). It does not let DuckDB see further into the
    file: dialect detection reads only the first buffer, whatever
    `sample_size` says.
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


def _duckdb(s: Sample) -> dict[str, Any]:
    fh = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    try:
        fh.write(s.utf8)
        fh.close()
        return duckdb_path(fh.name, has_pipe="|" in s.text)
    finally:
        Path(fh.name).unlink(missing_ok=True)


def duckdb_file(path: str | Path, s: Sample) -> dict[str, Any]:
    """`sniff_csv` for a downloaded file whose head is `s`.

    The real file when its bytes are already UTF-8, which needs no copy and
    has no truncated final row. Anything else goes through the head
    transcoded to UTF-8, like the probe pass: DuckDB would otherwise read
    UTF-16 as UTF-8 or drop every row with a byte >= 0x80.
    """
    if s.is_utf8:
        return duckdb_path(path, has_pipe="|" in s.text)
    return _duckdb(s)


def _chardet(s: Sample) -> dict[str, Any]:
    """A second opinion on the encoding, from a different detector.

    `custom` already carries charset_normalizer's verdict via
    `validate.probe_encoding`; where two independent detectors disagree is
    where the file is genuinely ambiguous.
    """
    import chardet

    res = chardet.detect(s.head)
    return {"encoding": res.get("encoding"), "confidence": res.get("confidence"),
            "language": res.get("language")}


def _charset_normalizer(s: Sample) -> dict[str, Any]:
    """The primary detector's own verdict, recorded verbatim.

    `custom.encoding` is this run through `canonical_encoding`; this keeps the
    raw name, the language guess and the multi-byte ratio alongside it. It is
    the run `probe_encoding` already made, except where a BOM or NUL parity
    let that skip the detector -- then it runs here, so the field is never
    missing.
    """
    if s.enc.detector is not None:
        return dict(s.enc.detector)
    return validate.charset_normalizer_verdict(s.head)


_RUNNERS = {"custom": _custom, "clevercsv": _clevercsv, "duckdb": _duckdb,
            "charset_normalizer": _charset_normalizer, "chardet": _chardet}


def run(head: bytes, size: int | None = None,
        which: tuple[str, ...] = DEFAULT) -> dict[str, Any]:
    """Every requested sniffer's raw output, keyed by sniffer name."""
    return run_sample(Sample(head, size), which)


def run_sample(s: Sample, which: tuple[str, ...] = DEFAULT) -> dict[str, Any]:
    """`run` on a sample the caller holds, to reuse its decoding afterwards."""
    out: dict[str, Any] = {}
    for name in which:
        runner = _RUNNERS.get(name)
        if runner is None:
            continue
        try:
            out[name] = runner(s)
        except Exception as exc:  # a sniffer must never kill the pass
            out[name] = {"error": type(exc).__name__}
    return out


# Sniffers whose input for a head depends on how the head is decoded.
_DECODED = ("clevercsv", "duckdb")


def run_changed(head: bytes, size: int | None, which: tuple[str, ...],
                have: tuple[str, ...]) -> dict[str, Any]:
    """`run`, minus the sniffers whose earlier verdict in `have` still holds.

    For re-sniffing heads that were sniffed before every sniffer read the same
    decoding. Back then CleverCSV was given the head decoded as UTF-8 and
    DuckDB its raw bytes. For a head that is ASCII or UTF-8 that is exactly
    what they are given now, so running them again would only repeat the old
    answer. Two cases still differ and are run again: a head with a byte order
    mark (CleverCSV used to see it as part of the first field), and any other
    encoding.
    """
    s = Sample(head, size)
    if s.is_utf8 and s.enc.bom is None:
        which = tuple(n for n in which if n not in _DECODED or n not in have)
    return run_sample(s, which)


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
