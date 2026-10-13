"""Sniffer verdicts, normalised into the parser's own dialect vocabulary.

Sniffer specifics:
  * "no quote character" is `""` from CleverCSV and the literal seven-character
    string `"(empty)"` from DuckDB. Stored unexamined, the second becomes a
    seven-byte quote character.
  * DuckDB's `NewLineDelimiter` is the *escape text* `"\\n"` -- backslash, n --
    and not a newline byte.
  * DuckDB answers `,` for a file with no delimiter at all, the same give-up
    default as the `|` its sniffer reaches for elsewhere.
  * CleverCSV reads a leading `# note` block as `#`-delimited data, while
    DuckDB reports `#` as the comment character. Both are describing the same
    two lines.

FIELDS is the parser's `Config`, one JSON key per struct field, and every
normaliser returns exactly it with None wherever that sniffer has no opinion.
`Shape` was named to match the struct, so `custom` is already in this
vocabulary and its normaliser only has to handle the one place the two models
genuinely differ -- see `from_custom`.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any

# One key per field of the parser's `Config`, in its order. `quote_style` and
# `quotes_in_unquoted_fields` are two keys for the one `QuoteStyle` field:
# `Strict(bool)` carries a payload and flat keys are what a human can edit and
# a diff can show.
FIELDS = (
    "delimiter", "terminator", "quote", "quote_style",
    "quotes_in_unquoted_fields", "escape", "escape_in_unquoted_fields",
    "double_quote_escape", "comment",
)

# Sniffers that describe a dialect. The encoding detectors have verdicts too,
# but nothing in FIELDS to say them in.
DIALECT_SNIFFERS = ("custom", "clevercsv", "duckdb")

# What each tool prints for "there is no such character". `""` is CleverCSV's;
# `"(empty)"` is DuckDB's, and it is a *value* rather than a formatting choice,
# so it has to be recognised rather than trimmed.
_NOTHING = ("", "(empty)", "\\0")

_EMPTY: dict[str, Any] = dict.fromkeys(FIELDS)


def byte(value: Any) -> str | None:
    """One character that fits in a byte, or None.

    The parser's fields are `u8`, and the manifest stores them as one-character
    strings so that `";"` and `"\\t"` stay readable in a diff and in an override
    file. Latin-1 is the mapping in both directions: `describe` decodes an
    unnameable 8-bit page as latin-1, so a delimiter byte from a non-UTF-8 file
    arrives here as the character of the same ordinal and goes back to that
    byte on the Rust side.

    Anything that is not exactly one such character -- a sentinel, a multi-byte
    character, a codepoint above 0xFF -- is no opinion rather than a guess.
    """
    if not isinstance(value, str) or value in _NOTHING:
        return None
    if len(value) != 1 or ord(value) > 0xFF:
        return None
    return value


def from_custom(verdict: dict[str, Any]) -> dict[str, Any]:
    """This project's own sniffer, which already speaks the struct's language.

    One genuine difference to reconcile: `Shape.quote_style` has a third value
    `"none"` for "quoting is off", which the parser expresses by `quote` being
    `None` instead. Carrying both would put a contradiction in the manifest --
    a quote character next to a quote style saying there is none -- so the
    redundant degree of freedom is collapsed here, in the direction the parser
    can represent.
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    out = {f: verdict.get(f) for f in FIELDS}
    for field in ("delimiter", "quote", "escape", "comment"):
        out[field] = byte(out[field])
    if verdict.get("quote_style") == "none" or out["quote"] is None:
        out["quote"] = None
        out["quote_style"] = None
        out["quotes_in_unquoted_fields"] = None
    out["terminator"] = _terminator(verdict.get("terminator"))
    return out


def from_clevercsv(verdict: dict[str, Any]) -> dict[str, Any]:
    """CleverCSV: three fields, and `""` throughout for "none".

    It has no notion of a terminator, a comment character or a quoting style,
    so those stay None rather than being defaulted -- a sniffer that was not
    asked must not get a vote.
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    return dict(_EMPTY, delimiter=byte(verdict.get("delimiter")),
                quote=byte(verdict.get("quotechar")),
                escape=byte(verdict.get("escapechar")))


def from_duckdb(verdict: dict[str, Any]) -> dict[str, Any]:
    """DuckDB, whose column names are capitalised and whose nulls are strings.

    `Escape` is dropped when it equals `Quote`: DuckDB reports `"` as the
    escape character for ordinary doubled-quote files, which is RFC 4180
    doubling rather than an escape character, and the parser has a separate
    `double_quote_escape` flag for it. Recording it as `escape` would configure
    the parser to treat a closing quote as an escape.
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    quote = byte(verdict.get("Quote"))
    escape = byte(verdict.get("Escape"))
    doubled = escape is not None and escape == quote
    return dict(
        _EMPTY,
        delimiter=byte(verdict.get("Delimiter")),
        terminator=_terminator(verdict.get("NewLineDelimiter")),
        quote=quote,
        escape=None if doubled else escape,
        double_quote_escape=True if doubled else None,
        comment=byte(verdict.get("Comment")),
    )


NORMALISERS = {"custom": from_custom, "clevercsv": from_clevercsv,
               "duckdb": from_duckdb}


def _terminator(value: Any) -> str | None:
    """A record terminator as the manifest spells it: "crlf", or one byte.

    Three vocabularies arrive here -- this project's `"crlf"`/`"lf"`/`"cr"`,
    DuckDB's escape text `"\\r\\n"`/`"\\n"`, and a literal byte -- and one leaves.
    `"crlf"` is four characters and a byte is one, so the two forms a reader
    has to tell apart are distinguishable by length, and `RecordTerminator`
    stays open to the bytes nothing here emits (`\\x1e` and friends) rather than
    being narrowed to a three-way enum it is not.
    """
    if not isinstance(value, str) or value in _NOTHING:
        return None
    lowered = value.lower()
    if lowered in ("crlf", "\\r\\n", "\r\n"):
        return "crlf"
    if lowered in ("lf", "\\n", "\n"):
        return "\n"
    if lowered in ("cr", "\\r", "\r"):
        return "\r"
    return byte(value)


def normalise(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every dialect sniffer's verdict for one file, in the struct's terms."""
    return {name: NORMALISERS[name](record.get(name) or {})
            for name in DIALECT_SNIFFERS if name in record}


def consensus(views: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge the normalised views field by field, and say how it went.

    A plain majority of the sniffers that hold an opinion on that field, with
    `custom` breaking ties. Abstention is not a vote -- CleverCSV says nothing
    about comment characters, and counting that silence as "no comment
    character" would let a sniffer that was never asked outvote one that was.

    `agreement` counts votes for the winning *delimiter* specifically, keeping
    it the same number the index's `agreement` column carries, so a dataset and
    the pool it was drawn from can be compared on it. `contested` names every
    field the sniffers split on, which is what a review queue is built from.

    Returns the dialect alongside its own provenance rather than as a bare
    dict: a consensus that nothing voted for and one that three tools agreed on
    are different facts, and the manifest should not have to be re-derived to
    tell them apart.
    """
    dialect: dict[str, Any] = {}
    contested: list[str] = []
    agreement = 0
    for field in FIELDS:
        votes = collections.Counter(
            view[field] for view in views.values() if view.get(field) is not None
        )
        if not votes:
            dialect[field] = None
            continue
        top = votes.most_common()
        best, count = top[0]
        tied = [value for value, n in top if n == count]
        if len(tied) > 1:
            contested.append(field)
            own = views.get("custom", {}).get(field)
            best = own if own in tied else sorted(tied)[0]
            count = votes[best]
        elif len(votes) > 1:
            contested.append(field)
        dialect[field] = best
        if field == "delimiter":
            agreement = count
    return {"dialect": dialect, "agreement": agreement, "contested": contested}


def _outside(line: bytes, delim: bytes, quote: bytes, inside: bool) -> tuple[int, bool]:
    """Delimiters in this line that are *not* inside quotes, and the end state.

    Splitting on the quote byte yields alternating outside/inside runs, so both
    the selection and the counting stay in C. Doubled quotes need no special
    case: `a,"x""y",b` splits to [`a,`, `x`, ``, `y`, `,b`] and the escape's
    empty run lands on an even index where it contributes nothing.
    """
    parts = line.split(quote)
    n = 0
    for i in range(0 if not inside else 1, len(parts), 2):
        n += parts[i].count(delim)
    return n, inside ^ (len(parts) % 2 == 0)


def consistency(path: Path, delimiter: str | None, quote: str | None,
                comment: str | None = None, chunk: int = 1 << 22) -> dict[str, Any]:
    """Stream the whole file and report whether one dialect describes all of it.

    The only evidence in the manifest that is not another prefix guess. Every
    sniffer here reads the front of a file -- CleverCSV hard-caps at 8 KiB,
    DuckDB decides the dialect from its first buffer, and `custom` is given a
    bounded window -- so a file whose dialect changes at 40 MB satisfies all
    three, and only reading the rest of it says so.

    Deliberately a scan and not a parse: every inner operation is a `bytes`
    method rather than a Python loop over bytes, so the cost scales with
    records rather than bytes and a whole-file pass stays affordable.

    It is quote-aware and comment-aware all the same: otherwise a clean
    `a,b,c` / `1,"x,y",3` file would score 50% and land in the review queue,
    and so would a leading `# comment` block.

    Reported as a histogram rather than a verdict: `modal_fields` and the share
    of records agreeing with it. A clean file is a spike at one value, drift is
    two clusters, and genuine raggedness is a tail. Calling any of those
    "ragged" here would claim a parse this still does not do.
    """
    if delimiter is None:
        return {"scanned": 0, "records": 0, "modal_fields": None,
                "modal_share": None, "quotes": 0, "reason": "no_delimiter"}
    delim = delimiter.encode("latin-1")
    quote_byte = quote.encode("latin-1") if quote else None
    comment_byte = comment.encode("latin-1") if comment else None
    hist: collections.Counter = collections.Counter()
    quotes = scanned = comments = 0
    inside = False
    pending = 0
    remainder = b""

    def take(line: bytes) -> None:
        nonlocal inside, pending, comments
        if comment_byte and not inside and not pending and line[:1] == comment_byte:
            comments += 1
            return
        if quote_byte:
            found, inside = _outside(line, delim, quote_byte, inside)
        else:
            found = line.count(delim)
        pending += found
        if not inside:
            # The newline ended a record rather than sitting inside a quoted
            # field, so the accumulated count is one row's worth.
            hist[pending] += 1
            pending = 0

    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            scanned += len(buf)
            if quote_byte:
                quotes += buf.count(quote_byte)
            # The chunk boundary almost never falls on a newline, so the
            # trailing fragment is carried rather than counted as a short
            # record.
            lines = (remainder + buf).split(b"\n")
            remainder = lines.pop()
            for line in lines:
                take(line.rstrip(b"\r") if not inside else line)
    if remainder:
        take(remainder.rstrip(b"\r"))
    if inside:
        # An unterminated quote runs to EOF; what it accumulated is one record.
        hist[pending] += 1

    total = sum(hist.values())
    if not total:
        return {"scanned": scanned, "records": 0, "modal_fields": None,
                "modal_share": None, "quotes": quotes, "comments": comments}
    mode, at_mode = hist.most_common(1)[0]
    return {
        "scanned": scanned,
        "records": total,
        "modal_fields": mode + 1,
        "modal_share": round(at_mode / total, 4),
        "quotes": quotes,
        "comments": comments,
    }
