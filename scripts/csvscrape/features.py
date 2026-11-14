"""This project's own sniffer: the dialect of a CSV and a description of it.

The work is done by `csvsniff`, a binary of the Rust harness, which parses the
sample with the parser's reference DFA (see `harness/src/sniff.rs`). This
module runs it and flattens its answer into a `Shape`, the record stored under
`custom` in the heads log.

Everything in a `Shape` describes one sample under one guessed dialect. It is
a label to select files by and to count over, not ground truth: independent
sniffers disagree on a meaningful share of real files.

Build the binary once with

    cargo build --release --bin csvsniff

or point the environment variable CSVSNIFF at one.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .validate import Encoding, decode, probe_encoding

_REPO = Path(__file__).resolve().parents[2]


def binary() -> str:
    """Path of the `csvsniff` binary."""
    explicit = os.environ.get("CSVSNIFF")
    if explicit:
        return explicit
    path = _REPO / "target" / "release" / "csvsniff"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist: run `cargo build --release --bin csvsniff` "
            "or set CSVSNIFF to the binary")
    return str(path)


def sniff(data: bytes | None = None, *, path: str | Path | None = None,
          size: int | None = None, encoding: str | None = None,
          parts: int = 0) -> dict[str, Any]:
    """Run `csvsniff` on bytes or on a file and return what it printed.

    `size` is the size of the whole file when `data` is only its start.
    `encoding` is a Python codec name; `csvsniff` converts the input from it
    to UTF-8 before parsing, the same way the harness does for a benchmark.
    `parts` asks for separate record counts for that many pieces of the input.
    """
    command = [binary()]
    if path is not None:
        command.append(str(path))
    if size is not None:
        command += ["--size", str(size)]
    if encoding:
        command += ["--encoding", encoding]
    if parts:
        command += ["--parts", str(parts)]
    done = subprocess.run(command, input=None if path is not None else data,
                          capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip()
                           or f"csvsniff exited with {done.returncode}")
    return json.loads(done.stdout)


@dataclass
class Shape:
    """One sample as `csvsniff` sees it, plus what is known of its encoding.

    The dialect fields are named like the fields of the parser's `Config`, so
    a file can be selected on exactly what the parser is configured by.
    """
    # --- dialect ------------------------------------------------------------
    delimiter: str | None
    # What ends a record in the parser's terms: "crlf" (any of \r, \n, \r\n)
    # or "\n" for a file where no record ends with a \r.
    record_terminator: str | None
    quote: str | None
    quote_style: str | None            # strict | toggle, None without a quote
    # The bool inside the parser's `Strict(bool)`: a quote in the middle of an
    # unquoted field is ordinary text. None unless quote_style is strict.
    quotes_in_unquoted_fields: bool | None
    escape: str | None
    escape_in_unquoted_fields: bool
    double_quote_escape: bool
    comment: str | None

    # --- how sure the guess is -----------------------------------------------
    # False if no delimiter splits the sample into two or more columns. The
    # delimiter is then a comma and the file is read as one column.
    delimiter_found: bool
    # Share of the sample's bytes in records with the most common field count
    # under the chosen dialect, 0 to 1.
    score: float
    # How far `score` is ahead of the best other delimiter. Near 0 means that
    # another delimiter fits the sample about as well.
    margin: float
    sniffer_version: int

    # --- encoding, from `validate.probe_encoding` ---------------------------
    # The detector's own verdict, or None where it declined to name one.
    encoding: str | None
    # What selectors filter on, see `validate.encoding_class`.
    encoding_class: str
    bom: str | None
    wide_encoding: str | None
    magic_kind: str | None
    control_frac: float

    # --- the sample -----------------------------------------------------------
    # Whether the sample is the whole file. If not, its cut off last record is
    # left out of every count below.
    complete: bool
    # Bytes that were parsed: the sample as UTF-8, without a byte order mark.
    bytes: int
    # None when the sample does not reach the end of the file.
    ends_with_newline: bool | None
    non_ascii_bytes: int
    nul_bytes: int
    has_nul: bool

    # --- records --------------------------------------------------------------
    records: int
    # A record that ran into a parse error. A parser stops there; the sniffer
    # skips to the end of the line and carries on.
    errors: int
    # The sample ends inside quotes.
    unterminated_quote: bool
    comment_lines: int
    empty_lines: int
    # Line endings outside of quoted fields.
    lf: int
    crlf: int
    cr: int
    # The most common of the three, as "lf", "crlf" or "cr".
    terminator: str
    mixed_terminators: bool
    # The field count that most records have, and the share of records that
    # have it.
    common_field_count: int
    common_field_count_share: float
    # The same share measured in bytes instead of records.
    consistent_bytes_share: float
    min_field_count: int
    max_field_count: int
    distinct_field_counts: int
    # Records with fewer / more fields than the common count, and both together.
    short_records: int
    long_records: int
    ragged_rows: int
    # Records before the first one with the common field count (among the
    # first 64): a title or notes above the table.
    preamble_records: int
    # Records whose last field is empty, i.e. lines that end with a delimiter.
    records_ending_with_empty_field: int
    longest_record: int
    # Record sizes in bytes. Slot 0 counts size 0, slot n counts sizes from
    # 2^(n-1) to 2^n - 1 (so: 0, 1, 2-3, 4-7, 8-15, ...).
    record_length_histogram: list[int]

    # --- fields ---------------------------------------------------------------
    fields: int
    # Total length of all fields after removing quotes and escapes.
    field_bytes: int
    empty_fields: int
    longest_field: int
    # Field lengths, in the same slots as `record_length_histogram`.
    field_length_histogram: list[int]
    # How unpredictable the length of a field is, in bits (Shannon entropy):
    # 0 if every field has the same length. `column_` is the same measured
    # within each column and averaged, which is lower when the columns differ
    # from each other but each one is regular.
    field_length_entropy: float
    column_length_entropy: float
    # Share of fields with the same length as the field above them.
    same_length_as_previous_record: float

    # --- quoting --------------------------------------------------------------
    quoting: str                       # none | minimal | all (95% of fields)
    quoted_fields: int
    # Quoted from the first to the last byte, the only form RFC 4180 knows.
    fully_quoted_fields: int
    quoted_field_bytes: int
    # Fields that are quoted when the field before them in the record is not,
    # or the other way round.
    quote_switches: int
    columns_always_quoted: int
    columns_never_quoted: int
    columns_sometimes_quoted: int
    fields_with_quoted_delimiter: int
    fields_with_quoted_terminator: int
    embedded_delimiter: bool           # fields_with_quoted_delimiter > 0
    embedded_newline: bool             # fields_with_quoted_terminator > 0
    doubled_quotes: int
    escapes_in_quoted_fields: int
    escapes_in_unquoted_fields: int
    # A quote that opens a quoted part after the field has started, and a
    # field that goes on after its closing quote: both need toggle quoting.
    quotes_opening_inside_field: int
    text_after_closing_quote: int
    # Quote characters that are ordinary text of an unquoted field.
    quote_chars_in_unquoted_fields: int
    # Fields whose content is not one unbroken piece of the input (doubled
    # quotes, escapes, quotes in the middle), so a parser that returns slices
    # of its input has to copy them.
    copied_fields: int
    copied_field_bytes: int

    # --- for SIMD parsers -----------------------------------------------------
    # Bytes at which a parser that skips ahead has to stop: delimiter,
    # terminator, quote and escape anywhere, or only those that mean something
    # in the state the parser is in (inside quotes only quote and escape).
    structural_bytes: int
    structural_bytes_in_state: int
    # Distances between two such bytes, in the slots described above.
    structural_gap_histogram: list[int]
    structural_gap_in_state_histogram: list[int]

    def as_dict(self) -> dict:
        return asdict(self)


# Keys of csvsniff's `stats` that a Shape takes over under the same name.
_SAME_NAME = (
    "records", "errors", "unterminated_quote", "comment_lines", "empty_lines",
    "common_field_count", "common_field_count_share", "consistent_bytes_share",
    "min_field_count", "max_field_count", "distinct_field_counts",
    "short_records", "long_records", "preamble_records",
    "records_ending_with_empty_field", "longest_record",
    "record_length_histogram", "fields", "field_bytes", "empty_fields",
    "longest_field", "field_length_histogram", "field_length_entropy",
    "column_length_entropy", "same_length_as_previous_record",
    "quoted_fields", "fully_quoted_fields", "quoted_field_bytes",
    "quote_switches", "columns_always_quoted", "columns_never_quoted",
    "columns_sometimes_quoted", "fields_with_quoted_delimiter",
    "fields_with_quoted_terminator", "doubled_quotes",
    "escapes_in_quoted_fields", "escapes_in_unquoted_fields",
    "quotes_opening_inside_field", "text_after_closing_quote",
    "copied_fields", "copied_field_bytes", "structural_bytes",
    "structural_bytes_in_state", "structural_gap_histogram",
    "structural_gap_in_state_histogram", "non_ascii_bytes", "nul_bytes",
)

# Share of quoted fields from which a file counts as quoting everything.
QUOTES_ALL = 0.95


def shape(out: dict[str, Any], enc: Encoding,
          ends_with_newline: bool | None) -> Shape:
    """Flatten the output of `sniff` into a `Shape`."""
    dialect, stats, detection = out["dialect"], out["stats"], out["detection"]
    endings = stats["line_endings"]
    if not stats["quoted_fields"]:
        quoting = "none"
    elif stats["quoted_fields"] >= QUOTES_ALL * stats["fields"]:
        quoting = "all"
    else:
        quoting = "minimal"
    return Shape(
        delimiter=dialect["delimiter"],
        record_terminator=dialect["terminator"],
        quote=dialect["quote"],
        quote_style=dialect["quote_style"],
        quotes_in_unquoted_fields=dialect["quotes_in_unquoted_fields"],
        escape=dialect["escape"],
        escape_in_unquoted_fields=dialect["escape_in_unquoted_fields"],
        double_quote_escape=dialect["double_quote_escape"],
        comment=dialect["comment"],
        delimiter_found=detection["delimiter_found"],
        score=detection["score"],
        margin=detection["margin"],
        sniffer_version=out["version"],
        encoding=enc.encoding,
        encoding_class=enc.encoding_class,
        bom=enc.bom,
        wide_encoding=enc.wide,
        magic_kind=enc.magic,
        control_frac=round(enc.control_frac, 5),
        complete=out["complete"],
        bytes=out["bytes"],
        ends_with_newline=ends_with_newline,
        has_nul=stats["nul_bytes"] > 0,
        lf=endings["lf"], crlf=endings["crlf"], cr=endings["cr"],
        # on a tie the more common convention
        terminator=max(("lf", "crlf", "cr"), key=lambda name: endings[name]),
        mixed_terminators=sum(1 for n in endings.values() if n) > 1,
        ragged_rows=stats["short_records"] + stats["long_records"],
        quoting=quoting,
        embedded_delimiter=stats["fields_with_quoted_delimiter"] > 0,
        embedded_newline=stats["fields_with_quoted_terminator"] > 0,
        quote_chars_in_unquoted_fields=stats["quotes_in_unquoted_fields"],
        **{name: stats[name] for name in _SAME_NAME},
    )


def source_encoding(enc: Encoding) -> str | None:
    """The encoding to convert a sample from, or None to use its bytes as
    they are: the byte order mark if there is one, then UTF-16 recognised by
    its zero bytes, then what the detector said."""
    return enc.bom or enc.wide or enc.encoding


def run(data: bytes | None, enc: Encoding, *, path: str | Path | None = None,
        size: int | None = None, parts: int = 0) -> dict[str, Any]:
    """`sniff` with the sample's encoding, for input in any encoding.

    `csvsniff` has no decoder for a few encodings Python knows (utf-7, hz,
    some DOS code pages are read byte by byte). A sample in one of those is
    decoded here and handed over as UTF-8. A file is not: the harness could
    not convert it for a benchmark either, so the error stands.
    """
    try:
        return sniff(data, path=path, size=size, parts=parts,
                     encoding=source_encoding(enc))
    except RuntimeError as exc:
        if path is not None or "no decoder" not in str(exc):
            raise
    text = decode(data, enc).encode("utf-8", "replace")
    # `size` counts bytes in the original encoding, so it cannot be compared
    # with the converted sample. Any size above the sample's own says "there
    # is more", which is all csvsniff needs to know.
    complete = size is None or size <= len(data)
    return sniff(text, parts=parts, size=None if complete else len(text) + 1)


def describe(head: bytes, size: int | None = None,
             enc: Encoding | None = None) -> Shape:
    """Describe a sample: the start of a file, or all of it.

    `size` is the size of the whole file; left out, the sample is taken to be
    complete. `enc` is the sample's `probe_encoding`, passed in by
    `sniffers.Sample` so that every sniffer works from the same reading.
    """
    if enc is None:
        enc = probe_encoding(head)
    out = run(head, enc, size=size)
    # Only a sample that reaches the end of the file can answer this.
    ends_nl = head.endswith((b"\n", b"\r")) if out["complete"] else None
    return shape(out, enc, ends_nl)
