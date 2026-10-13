"""Reading a remote zip's table of contents by HTTP Range, without the zip.

Kaggle bundles every dataset into one `archive.zip` on GCS, and GCS serves
ranges. That turns a dataset's file list into a range read or two of the
archive's tail, where the `/datasets/list/{ref}` endpoint returns 20 names per
metered call and stops after a few pages, truncating larger datasets.

The central directory carries more than the names, and two of the extra fields
are what the rest of the Kaggle pipeline is built on:

  * `crc32` and `usize` together are a content identity, available *before*
    downloading anything. Kaggle exposes no other hash, so this is what stands
    in for GitHub's blob sha -- see `index.build_kaggle` for what it does and
    does not buy.
  * `header_offset` plus `csize` locate the member's compressed bytes exactly,
    so a head is one more range read and a whole file is one after that. The
    bytes arrive deflated, which is a bandwidth saving on the download path
    rather than a cost.

Nothing here touches the network: it takes bytes and returns structure, which
is what makes it testable against a zip built in-process.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

# Signatures, in the order this module meets them.
EOCD = b"PK\x05\x06"
EOCD64 = b"PK\x06\x06"
CENTRAL = b"PK\x01\x02"
LOCAL = b"PK\x03\x04"

# The end-of-central-directory record is 22 bytes plus a comment of at most
# 65,535, so the last 65,557 bytes always contain it. Zip64 puts its own
# locator and record in front of that, hence the slack.
EOCD_SEARCH = 65_557 + 128

# A 32-bit field at its maximum means "the real value is in the zip64 extra".
U32_MAX = 0xFFFFFFFF
U16_MAX = 0xFFFF

# Compression methods this module can read a prefix of.
STORED = 0
DEFLATED = 8

# Slack allowed for a local header's extra field when a member's bytes are
# fetched in one read. The local and central extras are allowed to differ, so
# the length cannot be known in advance -- it is read back out of the header
# that arrives, and only a member whose extra exceeds this needs a second
# request. 1 KiB covers zip64 plus the usual timestamp and unix-attribute
# blocks many times over.
LOCAL_EXTRA_SLACK = 1024


class BadZip(Exception):
    """The bytes are not a zip, or not one this module can read."""


@dataclass(frozen=True)
class Member:
    """One entry of the central directory."""

    name: str
    method: int
    crc: int
    csize: int
    usize: int
    header_offset: int

    @property
    def readable(self) -> bool:
        """Whether a prefix of this member's content can be reconstructed.

        Stored and deflated cover everything Kaggle's bundler emits. Anything
        else -- bzip2, lzma, an encrypted entry -- is recorded and skipped
        rather than guessed at.
        """
        return self.method in (STORED, DEFLATED)

    @property
    def is_dir(self) -> bool:
        return self.name.endswith("/")


@dataclass(frozen=True)
class Directory:
    """Where the central directory lives, and how many entries it holds."""

    entries: int
    size: int
    offset: int


def find_directory(tail: bytes, total: int) -> Directory:
    """Locate the central directory, given the last bytes of the archive.

    `tail` must be the final `min(total, EOCD_SEARCH)` bytes of the file and
    `total` its full length, because the offsets recorded in the record are
    absolute and the tail has to be placed within the file to resolve them.

    Zip64 is not optional here. Kaggle bundles run to hundreds of gigabytes
    and hundreds of thousands of entries, so the 32-bit fields saturate
    routinely and the real values live in the zip64 record.
    """
    start = tail.rfind(EOCD)
    if start < 0:
        raise BadZip("no end-of-central-directory record")
    if start + 22 > len(tail):
        raise BadZip("truncated end-of-central-directory record")
    entries, size, offset = struct.unpack("<HII", tail[start + 10:start + 20])

    # The zip64 record supersedes the 32-bit one whenever it is present. Its
    # own fields are read unconditionally rather than only for the saturated
    # ones: a writer may emit it with every value filled in.
    z64 = tail.rfind(EOCD64, 0, start)
    if z64 >= 0 and z64 + 56 <= len(tail):
        entries, size, offset = struct.unpack("<QQQ", tail[z64 + 32:z64 + 56])
    elif entries == U16_MAX or size == U32_MAX or offset == U32_MAX:
        raise BadZip("zip64 fields saturated but no zip64 record in the tail")

    if offset + size > total:
        raise BadZip(f"central directory {offset}+{size} past end {total}")
    return Directory(entries=entries, size=size, offset=offset)


def _zip64_extra(extra: bytes, usize: int, csize: int,
                 offset: int) -> tuple[int, int, int]:
    """Replace saturated 32-bit fields from the zip64 extra block.

    The block holds only the fields that were saturated, in the fixed order
    uncompressed / compressed / offset / disk. Reading them positionally is
    therefore conditional on which ones saturated -- taking the first value as
    `usize` unconditionally is the classic way to get this wrong, and yields a
    member whose bytes are read from the wrong place in a 100 GB archive.
    """
    pos = 0
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[pos:pos + 4])
        body = extra[pos + 4:pos + 4 + size]
        if header_id == 0x0001:
            values = list(struct.unpack(f"<{len(body) // 8}Q",
                                        body[:len(body) // 8 * 8]))
            taken = iter(values)
            if usize == U32_MAX:
                usize = next(taken, usize)
            if csize == U32_MAX:
                csize = next(taken, csize)
            if offset == U32_MAX:
                offset = next(taken, offset)
            break
        pos += 4 + size
    return usize, csize, offset


def parse_directory(buf: bytes) -> list[Member]:
    """Every entry in a central directory, in the order it was written.

    A short or corrupt trailing record ends the walk rather than raising: the
    caller may have fetched a truncated range, and the entries already parsed
    are still true. The count is checked against the record by the caller,
    which is where the two facts can be compared.
    """
    out: list[Member] = []
    pos = 0
    while pos + 46 <= len(buf) and buf[pos:pos + 4] == CENTRAL:
        (method, _time, _date, crc, csize, usize,
         name_len, extra_len, comment_len,
         _disk, _internal, _external, offset) = struct.unpack(
            "<HHHIIIHHHHHII", buf[pos + 10:pos + 46])
        head = pos + 46
        name = buf[head:head + name_len].decode("utf-8", "replace")
        extra = buf[head + name_len:head + name_len + extra_len]
        if U32_MAX in (usize, csize, offset):
            usize, csize, offset = _zip64_extra(extra, usize, csize, offset)
        out.append(Member(name=name, method=method, crc=crc, csize=csize,
                          usize=usize, header_offset=offset))
        pos = head + name_len + extra_len + comment_len
    return out


def data_offset(local_header: bytes, header_offset: int) -> int:
    """Where a member's compressed bytes begin, from its local header.

    The local header's name and extra lengths are read rather than reused from
    the central directory, because the two are permitted to differ and
    routinely do -- the local extra often carries a zip64 block the central one
    does not, or vice versa. Assuming they match puts the read a few bytes off,
    which deflate reports as a corrupt stream.
    """
    if local_header[:4] != LOCAL:
        raise BadZip("not a local file header")
    if len(local_header) < 30:
        raise BadZip("truncated local file header")
    name_len, extra_len = struct.unpack("<HH", local_header[26:30])
    return header_offset + 30 + name_len + extra_len


def inflate_prefix(blob: bytes, method: int, want: int) -> bytes:
    """Up to `want` uncompressed bytes from the start of a member's stream.

    A deflate stream can be decoded from its beginning without the rest, which
    is the whole reason a head is cheap here: the member's first `want` bytes
    cost roughly `want` compressed bytes of transfer, whatever the file's full
    size.

    A truncated stream is expected, not exceptional -- the caller deliberately
    fetches less than the member. `decompressobj` decodes incrementally and
    returns whatever the input it was given yields, so a clean prefix survives
    the tail being absent. Corrupt input yields nothing.
    """
    if method == STORED:
        return blob[:want]
    if method != DEFLATED:
        raise BadZip(f"unsupported compression method {method}")
    engine = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        return engine.decompress(blob, want)
    except zlib.error:
        return b""


def inflate_all(blob: bytes, method: int, usize: int) -> bytes:
    """A member's whole content, with its declared length enforced.

    Used by the download path, where a short read is a failure rather than
    the prefix it is on the probe path.
    """
    if method == STORED:
        data = blob[:usize]
    elif method == DEFLATED:
        data = zlib.decompressobj(-zlib.MAX_WBITS).decompress(blob, usize)
    else:
        raise BadZip(f"unsupported compression method {method}")
    if len(data) != usize:
        raise BadZip(f"inflated {len(data)} bytes, central directory says {usize}")
    return data


def crc_of(data: bytes) -> int:
    """The CRC32 the central directory would record for these bytes."""
    return zlib.crc32(data) & U32_MAX
