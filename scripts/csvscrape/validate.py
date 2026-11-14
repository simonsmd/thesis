"""Encoding and binary-content observations, and content identity helpers.

The distinction that matters here is *binary* versus *not UTF-8*. A file that
fails UTF-8 decoding is usually not broken -- Latin-1 headers and UTF-16 Excel
exports are ordinary CSV in the wild, and they are precisely the files that
separate a tolerant parser from a naive one. Those are kept and labelled. What
a selector may throw out is content that is not text at all: gzip, parquet,
xlsx and friends hiding behind a .csv extension.
"""
from __future__ import annotations

import codecs
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Checked against the first bytes of the file. A .csv extension is a claim, not
# evidence, and archives mislabelled this way are common in scraped data.
MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x1f\x8b", "gzip"),
    (b"PK\x03\x04", "zip/xlsx"),
    (b"PAR1", "parquet"),
    (b"\x89PNG\r\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"%PDF", "pdf"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"SQLite format 3", "sqlite"),
    (b"\xd0\xcf\x11\xe0", "ole2/xls"),
    (b"\x50\x4b\x05\x06", "zip-empty"),
    (b"\x7fELF", "elf"),
    (b"ARROW1", "arrow"),
    (b"Obj\x01", "avro"),
)

HTML_MARKERS = (b"<!doctype html", b"<html", b"<?xml")
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# Bytes that are legitimate in text: printable ASCII, plus the whitespace a
# record separator or an indented field can carry. Everything else in the
# 0x00-0x1f range is a control byte.
_TEXT_CONTROL = {0x09, 0x0a, 0x0d, 0x0c, 0x0b, 0x1b}

# Above this share of control bytes the content is not text, whatever a
# statistical detector says. chardet reports plain "ascii" for a run of 0x01
# 0x02 -- true byte-wise, useless as an answer -- so binary formats outside the
# magic table would otherwise reach the dialect sniffers.
CONTROL_BYTE_LIMIT = 0.02


def magic_kind(head: bytes) -> str | None:
    """The format name if these bytes open with a known magic prefix."""
    for marker, kind in MAGIC:
        if head.startswith(marker):
            return kind
    return None


def git_blob_sha1(size: int, chunks) -> str:
    """Git's content hash: sha1("blob <len>\\0" + bytes).

    Streamed, using the *expected* length in the header. If the real length
    differs the digest will not match -- which is the same rejection the length
    check makes, so one pass suffices.
    """
    digest = hashlib.sha1(b"blob %d\x00" % size)
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Encoding:
    """Everything a sample says about its own encoding, nothing folded away.

    Deliberately not a verdict. Each field is one observation, recorded
    separately so a later stage can weigh them however it likes:

      encoding      the detector's name for it, canonicalised. None means the
                    detector declined to answer -- charset_normalizer abstains
                    when chaos crosses its threshold, which low-entropy text
                    does. That is "I cannot name it", not "this is not text",
                    and it is stored as itself rather than substituted.
      bom           byte-for-byte BOM match, exact
      magic         format name from the magic-byte table, exact
      wide          utf-16-le/be inferred from NUL parity, near-exact
      control_frac  share of bytes below 0x20 that are not text whitespace

    `magic` and `control_frac` are what a binary filter would use. The filter
    itself lives in the selector stage, so it can be changed without re-probing
    12M files.
    """

    encoding: str | None
    bom: str | None
    magic: str | None
    wide: str | None
    control_frac: float
    # charset_normalizer's own answer, verbatim (raw name, language, BOM flag,
    # multi-byte ratio), so the `charset_normalizer` sniffer can record it
    # without running the detector a second time over the same bytes. None
    # when the detector was skipped because a BOM or NUL parity settled it.
    detector: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def codec(self) -> str:
        """The codec to decode this sample with.

        latin-1 as the floor because it never raises and maps every byte to a
        distinct codepoint -- so an unnamed 8-bit page still yields text whose
        delimiter structure is intact, which is all the dialect sniffers need.
        """
        return self.bom or self.wide or self.encoding or "latin-1"

    @property
    def encoding_class(self) -> str:
        """`encoding_class` of the strongest observation: BOM, then NUL
        parity, then the detector -- the same precedence as `codec`."""
        return encoding_class(self.bom or self.wide or self.encoding)

    @property
    def looks_binary(self) -> bool:
        """Whether the sample is plainly not text. A helper, not a gate.

        UTF-16 is exempt from the density test: it is ~50% NUL by construction.
        Must agree with `selectors.not_binary` and the `main` selector, which
        apply the same rule.
        """
        if self.magic is not None:
            return True
        return self.wide is None and self.control_frac > CONTROL_BYTE_LIMIT


def probe_encoding(head: bytes) -> Encoding:
    """Observe a sample's encoding. Never rejects, never substitutes.

    The codepage comes from charset_normalizer (chardet's verdict is recorded
    separately by `sniffers`, as a second opinion). Neither names the exact
    codepage reliably, but misses stay within an encoding class, which is why
    selectors filter on `encoding_class`. charset_normalizer abstains on
    low-entropy input rather than guessing; that is recorded as None.

    The structural tests are exact rather than statistical: BOM prefixes (the
    only way to know a BOM was *present*), the magic-byte table, NUL parity
    for BOM-less UTF-16, and the control-byte share. Every observation is
    returned and none suppresses another, so a text/binary policy can be
    applied, and changed, at selection time.

    Names are canonicalised through the codecs registry, so one encoding is one
    label whether it arrived from the BOM table or the detector.
    """
    bom = next((name for marker, name in BOMS if head.startswith(marker)), None)
    magic = magic_kind(head)
    wide = _wide_encoding(head)
    control = _control_frac(head)

    encoding = detector = None
    if bom is None and wide is None:
        detector = charset_normalizer_verdict(_whole_utf8_chars(head))
        raw = detector["encoding"]
        encoding = canonical_encoding(raw) if raw else None
    return Encoding(encoding=encoding, bom=bom, magic=magic, wide=wide,
                    control_frac=control, detector=detector)


def _whole_utf8_chars(head: bytes) -> bytes:
    """`head` without a UTF-8 sequence cut off by the end of the sample.

    Heads are cut at a byte count, so a UTF-8 file's head can end inside a
    character. charset_normalizer decodes strictly, and that one partial
    character rules UTF-8 out: a 512 KB head that is UTF-8 throughout comes
    back as cp1250 or None. GraphQL heads never have this, since
    `client._blob_body` drops the partial character; range-request and Kaggle
    heads end at exactly HEAD_BYTES.

    Only the detector sees the trimmed bytes. On a file that is not UTF-8 the
    trim costs it at most three bytes, which changes nothing.
    """
    for back in range(1, min(4, len(head)) + 1):
        byte = head[-back]
        if byte < 0x80:
            return head
        if byte >= 0xC0:
            need = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            return head[:-back] if back < need else head
    return head


def charset_normalizer_verdict(head: bytes) -> dict[str, Any]:
    """charset_normalizer's best match, as the raw fields it reports."""
    from charset_normalizer import from_bytes

    best = from_bytes(head).best()
    if best is None:
        return {"encoding": None}
    return {"encoding": best.encoding, "language": best.language,
            "bom": best.bom, "multi_byte_usage": round(best.multi_byte_usage, 4)}


def decode(head: bytes, enc: Encoding) -> str:
    """The sample as text, the one decoding every dialect sniffer reads.

    `errors="replace"` because a sniffer needs the structure, not every glyph:
    a mislabelled byte becomes one U+FFFD and the delimiters around it stay
    put. A BOM is dropped -- `utf-8-sig` eats its own, but the UTF-16/32
    codecs named by byte order leave U+FEFF at the front, where it would read
    as part of the first header field.
    """
    try:
        text = head.decode(enc.codec, errors="replace")
    except (LookupError, TypeError):
        text = head.decode("latin-1", errors="replace")
    return text.lstrip("﻿")


def canonical_encoding(name: str) -> str:
    """The codecs registry's own spelling, so one encoding is one label."""
    try:
        return codecs.lookup(name).name
    except LookupError:
        return name.lower()


# Codec name fragments that mark a variable-width legacy CJK encoding. These
# are the ones that matter to a byte-oriented parser: a trail byte may land in
# 0x40-0x7E, so `|` (0x7C) and `\` (0x5C) can occur *inside* a character and be
# mistaken for a delimiter or an escape. Comma and semicolon sit below the 0x40
# trail floor and are safe in all of them.
_CJK_MARKERS = ("shift_jis", "cp932", "euc_jp", "euc_jis", "euc_kr", "cp949",
                "johab", "iso2022", "gb2312", "gbk", "gb18030", "big5",
                "cp950", "hz")


def encoding_class(encoding: str | None) -> str:
    """The part of an encoding verdict that is actually trustworthy.

    No detector gets the exact codepage reliably -- single-byte Western pages
    overlap so heavily that the distinction is often not recoverable from
    bytes at all -- but the misses stay *within* a class (cp1252 read as
    cp1250, euc_jp as euc_jis_2004). The class is what changes how a parser
    must behave, so selectors filter on this, while the raw label stays
    recorded verbatim.

      ascii        7-bit; every delimiter byte means itself
      utf-8        multi-byte, but continuation bytes are >= 0x80 -- safe
      utf-16/32    wide; NUL-interleaved, needs decoding before any byte scan
      single-byte  8-bit legacy page; safe byte-wise, wrong glyphs if mislabelled
      multibyte    legacy CJK; trail bytes can collide with delimiters
      binary       not text
    """
    if not encoding:
        # The detector declined to name it. Recorded as its own answer -- the
        # structural fields (magic, control_frac, wide) stand on their own and a
        # query can combine them however it wants.
        return "unknown"
    name = canonical_encoding(encoding)
    if name == "binary":
        return "binary"
    if name == "ascii":
        return "ascii"
    if name.startswith("utf-8") or name == "utf8":
        return "utf-8"
    if name.startswith(("utf-16", "utf-32")):
        return "wide"
    if any(marker in name for marker in _CJK_MARKERS):
        return "multibyte"
    if name.startswith("utf-7"):
        return "multibyte"
    return "single-byte"


def _wide_encoding(head: bytes) -> str | None:
    """BOM-less UTF-16, decided structurally rather than statistically.

    ASCII encoded as UTF-16 puts a NUL in every other byte, always at the same
    parity. That is a byte-for-byte pattern rather than a guess, and it works
    on short samples where a statistical detector has too little evidence.
    """
    pairs = len(head) // 2 * 2
    if pairs < 8:
        return None
    n = pairs // 2
    even_nul = head[0:pairs:2].count(0)
    odd_nul = head[1:pairs:2].count(0)
    if odd_nul >= n * 0.9 and even_nul <= n * 0.1:
        return "utf-16-le"
    if even_nul >= n * 0.9 and odd_nul <= n * 0.1:
        return "utf-16-be"
    return None


def _control_frac(head: bytes) -> float:
    """Share of bytes below 0x20 that are not text whitespace.

    A number, not a verdict, because the threshold that turns it into one is a
    policy question and belongs where policy lives. Note that UTF-16 scores
    ~0.5 by construction (every other byte is NUL), so a filter on this must
    consult `wide` too -- which is why both are recorded.
    """
    if not head:
        return 0.0
    control = sum(1 for b in head if b < 0x20 and b not in _TEXT_CONTROL)
    return control / len(head)


def check(path: Path) -> str | None:
    """A reject reason when downloaded bytes are not the file's content.

    By the time this runs, `download` has already checked the bytes against
    the index's content key (git blob sha, LFS sha256, or Kaggle CRC and
    length). The one failure that survives that check is an unresolved LFS
    pointer indexed as a plain file: it hashes correctly -- it *is* that blob
    -- but is not the content. Everything about what a usable file *is* is
    left to the selector's verifier, and its dialect to `dataset.build`.
    """
    with path.open("rb") as fh:
        head = fh.read(len(LFS_MAGIC))
    return "lfs_pointer" if head == LFS_MAGIC else None
