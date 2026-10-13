"""Named selectors: which slice of the pool a corpus should be drawn from.

`download` fetches; a selector decides what. Keeping them apart means a new
dataset is a new function here rather than an edit to the download path, and
the sniffing metadata is available to every selector because `compact` folds
the probe verdicts into the index as columns.

A selector is two stages, and the split is the whole design:

  * `filter` narrows the index. It sees only what the probe recorded from a
    head sample, so it is a cheap, approximate gate whose job is to make the
    *next* stage pay off -- the more of the drawn pool it can get to match, the
    fewer whole files are downloaded and thrown away.
  * `verify` sees the downloaded file, all of it, and gives the real answer.
    Anything a head sample cannot settle -- "valid UTF-8 to the last byte",
    "the sniffed delimiter still holds at the end" -- belongs here and nowhere
    else.

`needs_probe` says whether stage one is load-bearing. A selector that filters
hard on probe metadata (every dialect selector) can only draw probed files: an
unprobed row has nulls in those columns and would be admitted or rejected by
accident. A selector that does not -- `main`, `all`, `strict-utf8` -- keeps
unprobed files in the draw and lets the verifier settle them against the actual
bytes. That is the difference between drawing from the probed fraction of the
pool and drawing from the pool.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import polars as pl

# A pool filter sees head-sample metadata and narrows the candidate set.
PoolFilter = Callable[[pl.LazyFrame], pl.LazyFrame]
# A verifier sees the downloaded file and returns a reject reason, or None to
# accept. This is where anything requiring the WHOLE file belongs -- the pool
# filter only ever saw the head sample, so "valid UTF-8 throughout" or "the
# sniffed dialect holds to the last row" cannot be decided there.
Verifier = Callable[[Path, dict], "str | None"]


@dataclass(frozen=True)
class Selector:
    name: str
    filter: PoolFilter
    verify: Verifier | None = None
    # False means unprobed files stay in the draw: the filter does not depend
    # on probe metadata, so the verifier decides them from the bytes.
    needs_probe: bool = False
    doc: str = ""


REGISTRY: dict[str, Selector] = {}


def selector(name: str, verify: Verifier | None = None,
             needs_probe: bool = False) -> Callable[[PoolFilter], PoolFilter]:
    def register(fn: PoolFilter) -> PoolFilter:
        REGISTRY[name] = Selector(name=name, filter=fn, verify=verify,
                                  needs_probe=needs_probe,
                                  doc=(fn.__doc__ or name).strip())
        return fn
    return register


# -- whole-file verifiers --------------------------------------------------


def chain(*verifiers: Verifier | None) -> Verifier | None:
    """Run verifiers in order, stopping at the first rejection.

    Cheap tests first is the caller's job, not this function's: a verifier
    that reads the whole file costs the same whether it runs first or last,
    but only if it never runs is it free.
    """
    present = [v for v in verifiers if v is not None]
    if not present:
        return None
    if len(present) == 1:
        return present[0]

    def run(path: Path, row: dict[str, Any]) -> str | None:
        for verifier in present:
            why = verifier(path, row)
            if why:
                return why
        return None

    return run


def not_binary(path: Path, row: dict[str, Any]) -> str | None:
    """The one gate the main corpora apply: is this text at all?

    Deliberately the narrowest useful filter. A corpus meant to say what "a
    CSV file in the wild" looks like is ruined far more by a filter that
    excludes real CSVs than by a stray non-CSV that survives -- one is a
    silent bias across the whole population, the other is a file you notice
    and delete. So this rejects only what is unambiguous from the bytes:

      * a known binary magic prefix (gzip, zip/xlsx, parquet, PNG, ...),
        which is exact rather than statistical
      * a control-byte density no text file reaches, measured over the same
        head depth as the probe (`config.HEAD_BYTES`), with UTF-16 exempted
        because it is ~50% NUL by construction

    Not rejected here, on purpose: HTML and XML (text -- `text-only` is the
    selector for anyone who disagrees), unnameable encodings, zero data rows,
    and anything else that is a judgement about *content* rather than format.
    """
    from . import config, validate

    with path.open("rb") as fh:
        head = fh.read(config.HEAD_BYTES)
    enc = validate.probe_encoding(head)
    if enc.magic is not None:
        return f"magic/{enc.magic}"
    if enc.wide is None and enc.control_frac > validate.CONTROL_BYTE_LIMIT:
        return f"control_bytes({enc.control_frac:.2f})"
    return None


def not_markup(path: Path, row: dict[str, Any]) -> str | None:
    """Reject a file that opens as an HTML or XML document.

    Separate from `not_binary` because it is a different kind of claim: those
    bytes are text, and dropping them is a content judgement. `text-only`
    wants it; the main corpora do not.
    """
    from .validate import HTML_MARKERS

    with path.open("rb") as fh:
        head = fh.read(1024).lstrip().lower()
    for marker in HTML_MARKERS:
        if head.startswith(marker):
            return f"markup/{marker.decode()}"
    return None


def whole_file_utf8(path: Path, row: dict[str, Any]) -> str | None:
    """Decode the entire file, not a sample.

    The probe and the download-time check both look at the head, so a single
    stray byte at 40 MB passes both. Streaming an incremental decoder is the
    only way to answer this, and it reports where it failed.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    offset = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as exc:
                return f"invalid_utf8@{offset + exc.start}"
            offset += len(chunk)
    try:
        decoder.decode(b"", final=True)
    except UnicodeDecodeError:
        return f"truncated_utf8@{offset}"
    return None


def no_nul_bytes(path: Path, row: dict[str, Any]) -> str | None:
    """Reject a NUL anywhere in the file, not just the sampled head."""
    with path.open("rb") as fh:
        pos = 0
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                return None
            idx = chunk.find(b"\x00")
            if idx >= 0:
                return f"nul_byte@{pos + idx}"
            pos += len(chunk)


# How much of the end of a file `dialect_holds` re-sniffs. Capped so the check
# costs one seek and one read however large the file is, and floored at the
# second half so that on a file smaller than the cap the window cannot slide
# back over the head -- which would have the check agree with the very sample
# it is supposed to be contradicting.
TAIL_WINDOW = 256 * 1024


def dialect_holds(path: Path, row: dict[str, Any]) -> str | None:
    """Re-sniff the end of the file and check it still reads the same way.

    A dialect selector picks files on a verdict drawn from the head sample, and
    the failure that matters is a file that changes underneath it: a preamble
    in one dialect followed by a table in another, two concatenated exports,
    a semicolon section appended to a comma one. Those are not rare, and every
    one of them lands in a corpus labelled with a delimiter that describes its
    first page only.

    The tail is described with `features.describe` -- the same sniffer the
    probe pass uses -- so agreement here means agreement in one model rather
    than between two.

    A row with no recorded delimiter (an unprobed file) has nothing to
    contradict, so it passes.
    """
    from .features import describe

    expected = row.get("delimiter")
    if not expected:
        return None
    size = path.stat().st_size
    start = max(size // 2, size - TAIL_WINDOW)
    with path.open("rb") as fh:
        fh.seek(start)
        tail = fh.read()
    if not tail:
        return None
    # The read almost certainly landed mid-line; a partial first record would
    # be counted as a short row by the describer.
    cut = tail.find(b"\n")
    if 0 <= cut < len(tail) - 1:
        tail = tail[cut + 1:]
    observed = describe(tail, size=len(tail)).delimiter
    if observed is not None and observed != expected:
        return f"dialect_drift(head={expected!r} tail={observed!r})"
    return None


# -- pool filters ----------------------------------------------------------


@selector("all")
def all_files(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Everything in the pool, probed or not, filtered by nothing at all."""
    return pool


@selector("main", verify=not_binary)
def main(pool: pl.LazyFrame) -> pl.LazyFrame:
    """The corpus default: everything except what is plainly not text.

    Both headline datasets draw from here -- the benchmarking corpus through a
    log-uniform draw over size, the correctness corpus through a uniform one --
    and the filtering is the bare minimum for both, because a file wrongly
    excluded is invisible while a file wrongly included is a row you delete.

    Only the exact signals are consulted: a binary magic prefix, an unresolved
    LFS pointer, and a control-byte density outside anything text reaches.
    Every one of them is null for an unprobed file, which is deliberate --
    unprobed files stay in the draw and `not_binary` settles them against the
    bytes that actually land. See that verifier for what is *not* excluded.
    """
    from .validate import CONTROL_BYTE_LIMIT

    return pool.filter(
        pl.col("magic_kind").is_null()
        & ~pl.col("is_lfs_pointer").fill_null(False)
        & (
            pl.col("wide_encoding").is_not_null()
            | (pl.col("control_frac").fill_null(0.0) <= CONTROL_BYTE_LIMIT)
        )
    )


@selector("probed", needs_probe=True)
def probed(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Only files with sniffing metadata."""
    return pool.filter(pl.col("agreement").is_not_null())


@selector("text-only", verify=chain(not_binary, not_markup))
def text_only(pool: pl.LazyFrame) -> pl.LazyFrame:
    """`main`, plus dropping HTML and XML documents.

    The rule runs against fields recorded on every row, so it can be
    tightened, loosened or ignored without re-probing.
    """
    return main(pool).filter(~pl.col("looks_like_html").fill_null(False))


@selector("contested", verify=dialect_holds, needs_probe=True)
def contested(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Files the sniffers disagree about -- the discriminating cases.

    Fewer than three votes for the winning delimiter means at least one tool
    read the file differently, which is a decent operational definition of
    "hard" that does not require trusting any single sniffer.
    """
    return pool.filter(pl.col("agreement").is_not_null() & (pl.col("agreement") < 3))


@selector("unanimous", verify=dialect_holds, needs_probe=True)
def unanimous(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Files all three sniffers agree on -- the easy baseline."""
    return pool.filter(pl.col("agreement") == 3)


@selector("non-comma", verify=dialect_holds, needs_probe=True)
def non_comma(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Anything not plainly comma-delimited."""
    return pool.filter(pl.col("delimiter").is_not_null() & (pl.col("delimiter") != ","))


@selector("non-utf8", needs_probe=True)
def non_utf8(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Files that are text but not UTF-8 -- 8-bit pages, CJK multibyte, UTF-16.

    Filters on `encoding_class`, not the codepage name. No detector names the
    exact page reliably; all of them get the class right.
    """
    return pool.filter(
        pl.col("encoding_class").is_in(["single-byte", "multibyte", "wide"])
    )


@selector("multibyte-legacy", needs_probe=True)
def multibyte_legacy(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Legacy CJK encodings, where a trail byte can look like a delimiter.

    The sharpest test a byte-oriented parser faces: in Shift-JIS, GBK, Big5 and
    friends the second byte of a character may land in 0x40-0x7E, so `|` (0x7C)
    and `\\` (0x5C) occur *inside* characters. A parser that scans bytes without
    tracking character boundaries splits fields in the middle of a glyph.
    """
    return pool.filter(pl.col("encoding_class") == "multibyte")


@selector("wide-encoding", needs_probe=True)
def wide_encoding(pool: pl.LazyFrame) -> pl.LazyFrame:
    """UTF-16 and UTF-32, which no byte-wise scan can read without decoding."""
    return pool.filter(pl.col("encoding_class") == "wide")


@selector("quoted", verify=dialect_holds, needs_probe=True)
def quoted(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Files that actually use quoting."""
    return pool.filter(pl.col("quoting").is_in(["minimal", "all"]))


@selector("awkward-quoting", verify=dialect_holds, needs_probe=True)
def awkward_quoting(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Quoting that a naive parser gets wrong: embedded newlines or
    delimiters, toggle-style quotes, quotes inside bare fields, escapes."""
    return pool.filter(
        pl.col("embedded_newline").fill_null(False)
        | pl.col("embedded_delimiter").fill_null(False)
        | (pl.col("quote_style") == "toggle")
        | pl.col("quotes_in_unquoted_fields").fill_null(False)
        | pl.col("escape").is_not_null()
    )


@selector("structural-oddity", needs_probe=True)
def structural_oddity(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Preambles, multiple tables, ragged rows, mixed line endings, NULs."""
    return pool.filter(
        (pl.col("preamble").fill_null(0) > 0)
        | pl.col("multi_table").fill_null(False)
        | (pl.col("ragged_rows").fill_null(0) > 0)
        | pl.col("mixed_terminators").fill_null(False)
        | pl.col("has_nul").fill_null(False)
    )


@selector("crlf", needs_probe=True)
def crlf(pool: pl.LazyFrame) -> pl.LazyFrame:
    """CRLF-terminated files."""
    return pool.filter(pl.col("terminator") == "crlf")


@selector("strict-utf8", verify=chain(not_binary, whole_file_utf8))
def strict_utf8(pool: pl.LazyFrame) -> pl.LazyFrame:
    """`main`, restricted to UTF-8. The same thing as `--select main --utf8`."""
    return utf8_only(main(pool))


# -- composition -----------------------------------------------------------


def utf8_only(pool: pl.LazyFrame) -> pl.LazyFrame:
    """Narrow a pool to what could still be UTF-8, keeping the undecided.

    `encoding_class` comes off a head sample and is null for an unprobed file,
    so both of those are kept rather than assumed either way -- the head can
    only ever rule a file *out*, and `whole_file_utf8` is what rules one in.
    """
    return pool.filter(
        pl.col("encoding_class").is_null()
        | pl.col("encoding_class").is_in(["utf-8", "ascii"])
    )


def resolve(name: str | None, utf8: bool = False) -> Selector:
    """The selector to draw with, with `--utf8` folded in if asked for.

    Composing rather than defining a utf8 twin of every selector: the flag is
    orthogonal to what a selector is *for*, and both stages compose cleanly --
    the pool filters conjoin, the verifiers run in sequence. `whole_file_utf8`
    goes last because it reads every byte and the selector's own verifier may
    reject on the first page.
    """
    base = get(name or "all")
    if not utf8:
        return base
    return dataclasses.replace(
        base,
        name=f"{base.name}+utf8",
        filter=lambda pool, _f=base.filter: utf8_only(_f(pool)),
        verify=chain(base.verify, whole_file_utf8),
        doc=f"{base.doc.splitlines()[0]} (UTF-8 only)",
    )


def get(name: str) -> Selector:
    if name not in REGISTRY:
        raise SystemExit(
            f"unknown selector {name!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]


def describe_all() -> str:
    lines = [
        "  every selector also takes --utf8, which keeps only files that "
        "decode as UTF-8 end to end.",
        "",
        f"  {'name':<20}{'stage 2':<12}{'draws':<12} what it selects",
    ]
    for name, sel in sorted(REGISTRY.items()):
        mark = "verifies" if sel.verify else ""
        draws = "probed" if sel.needs_probe else "all files"
        lines.append(f"  {name:<20}{mark:<12}{draws:<12} {sel.doc.splitlines()[0]}")
    return "\n".join(lines)
