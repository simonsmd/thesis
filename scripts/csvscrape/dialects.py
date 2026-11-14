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
normaliser returns exactly it. A view keeps three answers apart:

  * a byte -- the sniffer found this character;
  * `NONE` -- the sniffer looked and says there is no such character;
  * `None` -- the sniffer has no opinion on the field (it does not model it,
    the field does not apply, or the sniffer failed).

`NONE` is an answer like any byte, so "no quote" from one sniffer against `"`
from another contests the field. `None` is not an answer. Sniffer failures are not in
the views at all -- `errors` collects them so a failed sniffer can send a file
to review instead of quietly becoming an abstention.

`features.Shape` uses the struct's names, so `custom` is already in this
vocabulary.

The dialect of a file is `custom`'s. The other two are used to find files
worth a second look, see `settle`.
"""
from __future__ import annotations

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

# A sniffer's explicit "there is no such character", as stored in a view. Four
# characters, so it cannot be mistaken for a byte (one character); it never
# reaches a merged dialect, where "none" is spelled null as the parser reads it.
NONE = "none"

# The character fields a sniffer can answer `NONE` for. The terminator is not
# one: every file has one, and a sentinel there is a failure to say which.
_CHARS = ("delimiter", "quote", "escape", "comment")

_EMPTY: dict[str, Any] = dict.fromkeys(FIELDS)


def byte(value: Any) -> str | None:
    """One character that fits in a byte, `NONE`, or None.

    The parser's fields are `u8`, and the manifest stores them as one-character
    strings so that `";"` and `"\\t"` stay readable in a diff and in an override
    file. Latin-1 is the mapping in both directions: `describe` decodes an
    unnameable 8-bit page as latin-1, so a delimiter byte from a non-UTF-8 file
    arrives here as the character of the same ordinal and goes back to that
    byte on the Rust side.

    A tool's spelling of "there is none" (`_NOTHING`) is `NONE`, an answer.
    Anything else that is not exactly one such character -- a multi-byte
    character, a codepoint above 0xFF, a non-string -- is no opinion rather
    than a guess.
    """
    if not isinstance(value, str):
        return None
    if value in _NOTHING:
        return NONE
    if len(value) != 1 or ord(value) > 0xFF:
        return None
    return value


def _found(value: Any) -> str | None:
    """`byte` for a sniffer whose missing value means "looked, found none".

    `custom` reports an absent character as None rather than a sentinel string;
    it always looks for every character it reports, so None from it is an
    answer, not an abstention.
    """
    return NONE if value is None else byte(value)


def from_custom(verdict: dict[str, Any]) -> dict[str, Any]:
    """This project's own sniffer, which already speaks the struct's language.

    Its record (`features.Shape`) carries the dialect under the names of
    FIELDS, except for the terminator: `terminator` there is the line ending
    the file mostly uses, and `record_terminator` is the parser setting.

    A missing character is None in the record. The sniffer looks for every
    character it reports, so that is an answer (`NONE`), not "no opinion".
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    out = {f: verdict.get(f) for f in FIELDS}
    for field in _CHARS:
        out[field] = _found(out[field])
    out["terminator"] = _terminator(verdict.get("record_terminator"))
    return out


def from_clevercsv(verdict: dict[str, Any]) -> dict[str, Any]:
    """CleverCSV: three fields, and `""` throughout for "none".

    It has no notion of a terminator, a comment character or a quoting style,
    so those stay None rather than being defaulted -- a sniffer that was not
    asked must not get a vote. Its `""` is an answer (`NONE`): it looked for a
    quote and found none.
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    return dict(_EMPTY, delimiter=byte(verdict.get("delimiter")),
                quote=byte(verdict.get("quotechar")),
                escape=byte(verdict.get("escapechar")))


def from_duckdb(verdict: dict[str, Any]) -> dict[str, Any]:
    """DuckDB, whose column names are capitalised and whose nulls are strings.

    `Escape` equal to `Quote` is not an escape character: DuckDB reports `"`
    as the escape for ordinary doubled-quote files, which is RFC 4180 doubling,
    and the parser has a separate `double_quote_escape` flag for it. Recording
    it as `escape` would configure the parser to treat a closing quote as an
    escape, so it is "no escape character" plus the flag.
    """
    if not verdict or verdict.get("error"):
        return dict(_EMPTY)
    quote = byte(verdict.get("Quote"))
    escape = byte(verdict.get("Escape"))
    doubled = escape not in (None, NONE) and escape == quote
    return dict(
        _EMPTY,
        delimiter=byte(verdict.get("Delimiter")),
        terminator=_terminator(verdict.get("NewLineDelimiter")),
        quote=quote,
        escape=NONE if doubled else escape,
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
        # No file lacks a terminator, so a sentinel here is no opinion.
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


def errors(record: dict[str, Any]) -> dict[str, str]:
    """Every sniffer in a raw record that failed, with what it said.

    Covers every sniffer that ran, the encoding detectors too: a failed sniffer
    leaves an all-None view that looks exactly like one with no opinion, and a
    dialect settled without it was settled on less evidence than it appears.
    A sniffer that returned nothing at all counts as failed.
    """
    out = {}
    for name, verdict in record.items():
        if not isinstance(verdict, dict) or not verdict:
            out[name] = "no_output"
        elif verdict.get("error"):
            out[name] = str(verdict["error"])
    return out


# The fields the other sniffers are compared on: the ones they report.
COMPARED = ("delimiter", "quote", "escape", "comment")


def settle(views: dict[str, dict[str, Any]], sample: bytes = b"",
           delimiter_found: bool = True) -> dict[str, Any]:
    """The dialect of a file, and where the other sniffers disagree with it.

    The dialect is `custom`'s: it is the only sniffer that knows every field
    of the parser's `Config`. CleverCSV and DuckDB do not vote. They are a
    check: `contested` lists the fields of COMPARED where one of them gives a
    different answer, which is what sends a file to manual review.

    Two things are not a disagreement:

      * None, a sniffer without an opinion on the field.
      * A character that does not occur in `sample`. The sniffer cannot have
        found it there, so it is the tool's default or its give-up answer
        (DuckDB reports `"` as the quote of a file without any, and `|` for a
        file it could not make sense of).

    `NONE` ("there is no such character") is an answer like any other, and it
    contests a field where `custom` found a character.

    `delimiter_found` is False when `custom` found a single column. Its
    dialect then has a comma, the parser needs some delimiter, and another
    sniffer saying "no delimiter" agrees with that.

    `agreement` is the number of sniffers, `custom` included, that name the
    chosen delimiter.
    """
    own = views.get("custom") or _EMPTY
    dialect = {f: None if own.get(f) == NONE else own.get(f) for f in FIELDS}
    contested = []
    for field in COMPARED:
        for name, view in views.items():
            theirs = view.get(field)
            if name == "custom" or theirs is None or theirs == own.get(field):
                continue
            if field == "delimiter" and not delimiter_found and theirs == NONE:
                continue
            if theirs != NONE and sample and theirs.encode("latin-1") not in sample:
                continue
            contested.append(field)
            break
    delimiter = own.get("delimiter")
    agreement = 0 if delimiter is None else sum(
        1 for view in views.values() if view.get("delimiter") == delimiter)
    return {"dialect": dialect, "agreement": agreement, "contested": contested}
