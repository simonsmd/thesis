"""Parsing-relevant description of a CSV, computed from a head sample.

Every metric is a *description for stratification*, never ground truth.
Independent sniffers disagree on a meaningful share of real files, so any
single verdict here is a label to sample by, not a fact to score against.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .validate import encoding_class, probe_encoding

DELIMITERS = (",", ";", "\t", "|", ":", "\x01")

# Whitespace is only accepted when nothing else fits AND it splits every
# sampled row identically. Without that bar it wins on single-column files,
# where comma yields one field and is skipped while space happily splits prose.
WEAK_DELIMITERS = (" ",)
_QUOTES = ('"', "'")

# How much better a later delimiter candidate must score to displace an earlier
# one. Candidate order encodes prior likelihood, so this is what stops a
# coincidentally-consistent character from winning on a near-tie.
DISPLACE_MARGIN = 0.10


@dataclass
class Shape:
    """Mirrors the parser's own dialect configuration where it can.

    Field names deliberately match the Rust struct (`delimiter`, `terminator`,
    `quote`, `quote_style`, `escape`, `escape_in_unquoted_fields`,
    `double_quote_escape`, `comment`) so a sampled corpus can be selected
    directly on the axes the parser is configured by, rather than on a
    parallel vocabulary that has to be mentally translated.
    """
    # --- dialect, in the parser's own terms --------------------------------
    delimiter: str | None
    terminator: str                    # crlf | lf | cr
    quote: str | None
    quote_style: str | None            # strict | toggle | none
    quotes_in_unquoted_fields: bool    # the bool inside Strict(bool)
    escape: str | None
    escape_in_unquoted_fields: bool
    double_quote_escape: bool
    comment: str | None
    # --- everything else ----------------------------------------------------
    # The detector's own verdict, or None where it declined to name one. Not
    # substituted, so `charset_normalizer said nothing` stays recoverable.
    encoding: str | None
    # Structural observations, kept beside the verdict rather than folded into
    # it. A binary filter is `magic_kind is not None or control_frac > 0.02`,
    # and it lives in the selector stage so it can change without re-probing.
    magic_kind: str | None
    control_frac: float
    wide_encoding: str | None
    # The class is what selectors filter on: no detector names the exact
    # codepage reliably, but every one of them gets the class right, and the
    # class is what changes how a parser must treat the bytes. See
    # validate.encoding_class.
    encoding_class: str
    bom: str | None
    mixed_terminators: bool
    columns: int
    rows_sampled: int
    ragged_rows: int
    max_field_len: int
    max_line_len: int
    trailing_delimiter: bool
    padded_fields: bool
    empty_lines: int
    quoting: str                       # none | minimal | all
    embedded_delimiter: bool
    embedded_newline: bool
    escaped_quotes: int
    unterminated_quote: bool
    preamble: int
    has_nul: bool
    non_ascii_frac: float
    multi_table: bool
    complete: bool
    # None when the sample does not reach the end of the file, which is the
    # ordinary case for anything larger than the head: unknown is recorded as
    # unknown rather than guessed.
    ends_with_newline: bool | None

    def as_dict(self) -> dict:
        return asdict(self)


def _split_lines(text: str) -> tuple[list[str], str, bool]:
    crlf, lone_lf = text.count("\r\n"), text.count("\n") - text.count("\r\n")
    lone_cr = text.count("\r") - crlf
    if crlf and (lone_lf or lone_cr):
        mixed = True
    else:
        mixed = bool(lone_lf and lone_cr)
    if crlf >= max(lone_lf, lone_cr):
        return text.split("\r\n"), "crlf", mixed
    if lone_cr > lone_lf:
        return text.split("\r"), "cr", mixed
    return text.split("\n"), "lf", mixed


def _split_fields(line: str, delim: str, quote: str | None,
                  style: str = "toggle", escape: str | None = None,
                  raw: bool = False, tally: dict[str, int] | None = None) -> list[str]:
    """Split one line into fields. A probe, not a parser -- it never errors.

    `style` mirrors the parser's own QuoteStyle:
      * "toggle" -- any quote character flips quoting state, mid-field included,
        so `ab"c,d"e` is the single field `abc,de`.
      * "strict" -- a quote only opens a field when it appears at the field
        boundary; elsewhere it is a literal character.
      * "none"   -- quoting disabled entirely.

    Doubled quotes always emit one literal quote. `escape` additionally treats
    the next character as literal.

    `tally`, when given, accumulates `doubled`: the number of genuine
    doubled-quote escapes. That needs the quoting state this walk tracks --
    the `""` in `1,"",3` is an empty field, not an escape.
    """
    if quote is None or style == "none":
        return line.split(delim)
    out: list[str] = []
    buf: list[str] = []
    start = 0
    in_q = False
    at_field_start = True
    i = 0
    while i < len(line):
        ch = line[i]
        if escape and ch == escape and i + 1 < len(line):
            buf.append(line[i + 1])
            i += 2
            at_field_start = False
            continue
        if in_q:
            if ch == quote:
                if i + 1 < len(line) and line[i + 1] == quote:
                    buf.append(quote)
                    if tally is not None:
                        tally["doubled"] += 1
                    i += 2
                    continue
                in_q = False
            else:
                buf.append(ch)
        elif ch == quote and (style == "toggle" or at_field_start):
            in_q = True
            at_field_start = False
        elif ch == delim:
            out.append(line[start:i] if raw else "".join(buf))
            buf = []
            start = i + 1
            at_field_start = True
        else:
            buf.append(ch)
            at_field_start = False
        i += 1
    out.append(line[start:] if raw else "".join(buf))
    return out


def _score(rows: list[str], cand: str, quote: str | None, style: str,
           escape: str | None = None):
    counts = [len(_split_fields(r, cand, quote, style, escape)) for r in rows[:25]]
    if not counts:
        return None
    mode = max(set(counts), key=counts.count)
    # Gate on the modal row, not the first. A title banner ahead of the table
    # has no delimiter, and testing rows[0] rejected the real delimiter for the
    # whole file rather than treating the banner as one dissenting row.
    if mode < 2:
        return None
    return counts.count(mode) / len(counts), mode


def _pick_dialect(rows: list[str], escape: str | None
                  ) -> tuple[str | None, str | None, str | None]:
    """Jointly choose (delimiter, quote, quote_style) by consistency.

    All three interact, so none can be decided first. Detecting the quote
    character by adjacency (`",` / `,"`) misses toggle-style quoting entirely,
    because concatenating quotes never sit next to a delimiter -- the only way
    to find them is to try each interpretation and keep whichever splits the
    file most consistently. The winning combination is itself the answer to
    "which QuoteStyle does this file use".
    """
    best = None
    # Quoted interpretations are tried FIRST, but only for a quote character
    # that actually occurs. Consistency alone cannot separate "quotes present
    # but irrelevant to the field count" (`"a","b"` splits the same either way)
    # from "no quotes at all", so presence breaks that tie -- while requiring
    # two occurrences keeps a lone stray apostrophe from being promoted.
    sample = "\n".join(rows[:25])
    combos = []
    for q in _QUOTES:
        if sample.count(q) >= 2:
            combos += [(q, "strict"), (q, "toggle")]
    combos.append((None, "none"))
    for quote, style in combos:
        for cand in DELIMITERS:
            scored = _score(rows, cand, quote, style, escape)
            if scored is None:
                continue
            agree, mode = scored
            # Candidates are listed most-common-first and the quoted
            # interpretations come before the unquoted one, so a later
            # candidate has to beat the incumbent by a real margin to displace
            # it; on anything closer, a tie included, the earlier one holds.
            # (Breaking ties on column count would let ':' win on "12:34:56,5".)
            if best is None or agree > best[0] + DISPLACE_MARGIN:
                best = (agree, mode, cand, quote, style)
    if best is not None:
        return best[2], best[3], best[4]
    for cand in WEAK_DELIMITERS:
        scored = _score(rows, cand, None, "none", escape)
        if scored and scored[0] == 1.0 and scored[1] >= 3:
            return cand, None, None
    return None, None, None


def describe(head: bytes, size: int | None = None) -> Shape:
    complete = size is None or size <= len(head)
    enc = probe_encoding(head)
    # Every sample is described, including one that looks binary: a dialect
    # sniffed out of a gzip stream is meaningless, but it is recorded next to
    # magic_kind, so a query can drop it -- and a misjudged file keeps its
    # verdict.
    try:
        text = head.decode(enc.codec, errors="replace")
    except (LookupError, TypeError):
        text = head.decode("latin-1", errors="replace")
    text = text.lstrip("﻿")

    lines, terminator, mixed = _split_lines(text)
    # The final line of a truncated sample is a fragment, not a record.
    if not complete and len(lines) > 1:
        lines = lines[:-1]

    preamble, comment_char = 0, None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            preamble += 1
        elif stripped[0] in "#;/" and not stripped.startswith(";;"):
            preamble += 1
            comment_char = comment_char or stripped[0]
        else:
            break
    body = [ln for ln in lines[preamble:] if ln.strip()]

    escape = "\\" if re.search(r'\\["\']', text) else None
    delimiter, quote_char, quote_style = _pick_dialect(body, escape)

    # Real files often open with a title or metadata banner carrying no
    # delimiter before the table. Strip the leading delimiter-free lines and
    # vote again on what remains.
    if delimiter:
        skip = 0
        for line in body:
            if line.count(delimiter) == 0:
                skip += 1
            else:
                break
        if skip and skip < len(body):
            preamble += skip
            body = body[skip:]
            delimiter, quote_char, quote_style = _pick_dialect(body, escape)

    counts, max_field, max_line = [], 0, 0
    tally = {"doubled": 0}
    for line in body[:200]:
        max_line = max(max_line, len(line))
        if delimiter:
            fields = _split_fields(line, delimiter, quote_char, quote_style or "toggle",
                                   escape, tally=tally)
            counts.append(len(fields))
            for f in fields:
                max_field = max(max_field, len(f))
        else:
            max_field = max(max_field, len(line))
    columns = max(set(counts), key=counts.count) if counts else 1
    ragged = sum(1 for c in counts if c != columns)

    # How quoting is *used*, field by field: a quote opening a field is real
    # quoting; a quote inside an otherwise-bare field is the Strict(bool) case.
    # Splitting must be quote-aware here or a strict file's `"x,1"` looks like
    # a bare field `1"` containing a quote.
    quoted_cells = total_cells = bare_with_quote = 0
    escape_unquoted = False
    if delimiter:
        for line in body[:50]:
            for raw_field in _split_fields(line, delimiter, quote_char,
                                           quote_style or "none", escape, raw=True):
                stripped = raw_field.strip()
                total_cells += 1
                if quote_char and stripped.startswith(quote_char):
                    quoted_cells += 1
                elif quote_char and quote_char in stripped:
                    bare_with_quote += 1
                if escape and escape in stripped and not (
                        quote_char and stripped.startswith(quote_char)):
                    escape_unquoted = True
    if not quote_char or total_cells == 0:
        quoting = "none"
    elif quoted_cells >= total_cells * 0.95:
        quoting = "all"
    elif quoted_cells:
        quoting = "minimal"
    else:
        quoting = "none"

    # Doubled quotes come from the field walker above, which can tell an
    # escape from an empty quoted field `""`.
    escaped = tally["doubled"]
    embedded_delim = embedded_nl = False
    if quote_char:
        quoted = re.findall(rf"{re.escape(quote_char)}([^{re.escape(quote_char)}]*)"
                            rf"{re.escape(quote_char)}", text, flags=re.S)
        if delimiter:
            embedded_delim = any(delimiter in q for q in quoted)
        embedded_nl = any("\n" in q or "\r" in q for q in quoted)

    # Parity over the same sample the escapes were counted in, so the two agree.
    sampled = "\n".join(body[:200])
    unterminated = bool(quote_char) and (
        sampled.count(quote_char) - 2 * escaped) % 2 == 1

    trailing = bool(delimiter) and sum(
        1 for ln in body[:50] if ln.endswith(delimiter)) > len(body[:50]) * 0.5
    padded = bool(delimiter) and any(
        f" {delimiter}" in ln or f"{delimiter} " in ln for ln in body[:50])

    multi_table = False
    if delimiter and len(counts) > 2:
        seen_blank = False
        for line in lines[preamble:][:200]:
            if not line.strip():
                seen_blank = True
            elif seen_blank:
                if len(_split_fields(line, delimiter, quote_char,
                                     quote_style or "toggle", escape)) != columns:
                    multi_table = True
                break

    non_ascii = sum(1 for b in head if b > 127) / max(len(head), 1)
    # Only a sample that reaches the end of the file can answer this.
    ends_nl = head.endswith((b"\n", b"\r")) if complete else None

    return Shape(
        delimiter=delimiter, terminator=terminator, quote=quote_char,
        quote_style=(quote_style if quote_char else None),
        quotes_in_unquoted_fields=bare_with_quote > 0,
        escape=escape, escape_in_unquoted_fields=escape_unquoted,
        double_quote_escape=escaped > 0, comment=comment_char,
        encoding=enc.encoding,
        encoding_class=encoding_class(enc.bom or enc.wide or enc.encoding),
        magic_kind=enc.magic, control_frac=round(enc.control_frac, 5),
        wide_encoding=enc.wide, bom=enc.bom, mixed_terminators=mixed,
        columns=columns, rows_sampled=len(body), ragged_rows=ragged,
        max_field_len=max_field, max_line_len=max_line,
        trailing_delimiter=trailing, padded_fields=padded,
        empty_lines=sum(1 for ln in lines[preamble:] if not ln.strip()),
        quoting=quoting, embedded_delimiter=embedded_delim,
        embedded_newline=embedded_nl, escaped_quotes=escaped,
        unterminated_quote=unterminated, preamble=preamble,
        # Checked on the decoded text, not the raw bytes: UTF-16 encodes ASCII
        # with a NUL in every other byte, and the flag is meant to catch an
        # embedded NUL in the text itself.
        has_nul="\x00" in text, non_ascii_frac=round(non_ascii, 5),
        multi_table=multi_table, complete=complete,
        ends_with_newline=ends_nl,
    )
