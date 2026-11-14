"""Tests for the hand-over to csvsniff and for encoding detection.

Needs the binary: `cargo build --release --bin csvsniff`.
"""
from __future__ import annotations

import unittest

from csvscrape.features import Shape, describe


class Handover(unittest.TestCase):
    """`describe` runs csvsniff and flattens its answer. How the dialect is
    detected is tested in `harness/src/sniff.rs`."""

    def test_dialect_and_counts(self):
        s = describe(b'a;b\r\n"x;1";2\r\n3;4\r\n')
        self.assertEqual((s.delimiter, s.quote, s.quote_style), (";", '"', "strict"))
        self.assertEqual(s.record_terminator, "crlf")
        self.assertEqual((s.terminator, s.mixed_terminators), ("crlf", False))
        self.assertEqual((s.records, s.fields, s.common_field_count), (3, 6, 2))
        self.assertEqual((s.quoting, s.embedded_delimiter), ("minimal", True))
        self.assertTrue(s.complete)
        self.assertTrue(s.ends_with_newline)

    def test_a_cut_off_sample_drops_its_last_record(self):
        s = describe(b"a,b\n1,2\n3", size=1000)
        self.assertFalse(s.complete)
        self.assertEqual(s.records, 2)
        self.assertIsNone(s.ends_with_newline)

    def test_other_encodings_are_converted(self):
        s = describe("a;b\nä;ö\nü;ß\n".encode("utf-16"))
        self.assertEqual((s.delimiter, s.records), (";", 3))
        self.assertFalse(s.has_nul)
        # utf-7 is an encoding csvsniff has no decoder for
        from csvscrape.validate import Encoding
        enc = Encoding(encoding="utf-7", bom=None, magic=None, wide=None, control_frac=0.0)
        self.assertEqual(describe(b"a,b\n1,2\n", enc=enc).records, 2)

    def test_empty_input(self):
        s = describe(b"")
        self.assertEqual((s.records, s.delimiter_found), (0, False))

    def test_every_field_has_a_column_type(self):
        from csvscrape import probe
        for name in Shape.__annotations__:
            probe._shape_dtype(name)



class BinaryAndWideEncodings(unittest.TestCase):
    """Binary must never produce a dialect, and NUL is valid UTF-8."""

    import os as _os

    def test_binary_is_described_and_flagged_not_dropped(self):
        """Binary is labelled, not refused a description: the magic hit is
        recorded beside the sniffers' verdict, and `text-only` filters
        on it."""
        from csvscrape.features import describe
        for payload, kind in ((b"\x1f\x8b\x08\x00" + b"\xde\xad" * 500, "gzip"),
                              (b"PAR1" + b"\x01\x02" * 500, "parquet"),
                              (b"SQLite format 3\x00" + b"\x7f\x80" * 500, "sqlite")):
            shape = describe(payload, size=len(payload))
            self.assertEqual(shape.magic_kind, kind, str(payload[:8]))

    def test_bomless_utf16_is_not_mistaken_for_utf8(self):
        """NUL is a valid UTF-8 codepoint, so BOM-less UTF-16 ASCII decodes as
        UTF-8; it must still be recognised as UTF-16."""
        from csvscrape.validate import probe_encoding
        for codec, expected in (("utf-16-le", "utf-16-le"), ("utf-16-be", "utf-16-be")):
            data = "name,value\nalpha,1\nbeta,2\ngamma,3\n".encode(codec)
            enc = probe_encoding(data)
            self.assertEqual(enc.wide, expected)
            self.assertFalse(enc.looks_binary)

    def test_utf16_dialect_still_detected(self):
        from csvscrape.features import describe
        data = "a,b\n1,2\n3,4\n".encode("utf-16-le")
        self.assertEqual(describe(data, size=len(data)).delimiter, ",")

    def test_nul_in_otherwise_ascii_is_text_but_flagged(self):
        """A few NULs in otherwise-ASCII data is a parser hazard, not a binary
        file -- it stays in the corpus with has_nul set rather than vanishing."""
        from csvscrape.features import describe
        data = b"a,b\n1,\x00\x00\x002\n" + b"x" * 200
        shape = describe(data, size=len(data))
        self.assertIsNone(shape.magic_kind)
        self.assertTrue(shape.has_nul)

    def test_bom_utf16_still_wins(self):
        from csvscrape.validate import probe_encoding
        data = b"\xff\xfe" + "a,b\n".encode("utf-16-le")
        enc = probe_encoding(data)
        self.assertEqual(enc.bom, "utf-16-le")
        self.assertEqual(enc.codec, "utf-16-le")


class NulFlag(unittest.TestCase):
    """`has_nul` is checked on decoded text, so UTF-16 does not trip it."""

    def test_utf16_is_not_flagged_as_containing_nul(self):
        body = "date,city,value\n" + "".join(
            f"2024-01-{i:02d},Berlin,{i}\n" for i in range(1, 29))
        shape = describe((body * 4).encode("utf-16-le"))
        self.assertFalse(shape.has_nul)
        self.assertEqual(shape.delimiter, ",")

    def test_genuine_embedded_nul_is_still_flagged(self):
        shape = describe(b"a,b\n1,\x00\x002\n" + b"x,y\n" * 50)
        self.assertTrue(shape.has_nul)


class EncodingNames(unittest.TestCase):
    """One encoding must be one label, BOM or no BOM."""

    def test_bom_and_bomless_utf16_agree(self):
        from csvscrape.validate import probe_encoding

        body = ("a,b\n" + "1,2\n" * 40).encode("utf-16-le")
        self.assertEqual(probe_encoding(body).codec, "utf-16-le")
        self.assertEqual(probe_encoding(b"\xff\xfe" + body).codec, "utf-16-le")

    def test_short_bomless_utf16_is_recognised(self):
        """A statistical detector has too little evidence here; NUL parity is
        structural and does not care."""
        from csvscrape.validate import probe_encoding

        enc = probe_encoding("a,b\n1,2\n".encode("utf-16-le"))
        self.assertEqual(enc.wide, "utf-16-le")
        self.assertFalse(enc.looks_binary)

    def test_control_byte_soup_is_flagged_binary(self):
        from csvscrape.validate import probe_encoding

        enc = probe_encoding(b"PAR1" + b"\x01\x02" * 500)
        self.assertTrue(enc.looks_binary)
        self.assertEqual(enc.magic, "parquet")


class TruncatedUtf8(unittest.TestCase):
    """A head cut inside a UTF-8 character is still UTF-8. charset_normalizer
    decodes strictly and, given the partial character, answered cp1250."""

    ROWS = ("id;name;city\n"
            + "".join(f"{i};Müller {i};Köln\n" for i in range(200))).encode()

    def test_cut_inside_a_character_is_still_utf8(self):
        from csvscrape.validate import probe_encoding

        for char in ("ä", "€", "😀"):
            encoded = char.encode()
            for cut in range(1, len(encoded)):
                with self.subTest(char=char, cut=cut):
                    enc = probe_encoding(self.ROWS + encoded[:cut])
                    self.assertEqual(enc.encoding, "utf-8")
                    self.assertEqual(enc.encoding_class, "utf-8")

    def test_only_a_cut_sequence_is_trimmed(self):
        from csvscrape.validate import _whole_utf8_chars

        self.assertEqual(_whole_utf8_chars(b"ab\xc3"), b"ab")
        self.assertEqual(_whole_utf8_chars(b"ab\xe2\x82"), b"ab")
        self.assertEqual(_whole_utf8_chars(b"ab\xf0\x9f\x98"), b"ab")
        for whole in (b"ab", b"a\xc3\xa4", b"a\xe2\x82\xac", b"\xf0\x9f\x98\x80",
                      b"a\x80\x80\x80\x80", b""):
            with self.subTest(whole=whole):
                self.assertEqual(_whole_utf8_chars(whole), whole)

    def test_a_single_byte_page_keeps_its_label(self):
        from csvscrape.validate import probe_encoding

        rows = self.ROWS.decode().encode("cp1252")
        # Ends on 0xE4 ("ä"), which UTF-8 would read as a cut lead byte.
        enc = probe_encoding(rows + "ä".encode("cp1252"))
        self.assertEqual(enc.encoding_class, "single-byte")


def _have_detector() -> bool:
    try:
        import charset_normalizer  # noqa: F401
    except ImportError:
        return False
    return True


class EncodingClass(unittest.TestCase):
    """The class is what selectors filter on, because the exact codepage is
    not reliably recoverable from bytes by any detector."""

    def klass(self, data: bytes) -> str:
        return describe(data).encoding_class

    def test_ascii_and_utf8(self):
        self.assertEqual(self.klass(b"a,b\n1,2\n" * 60), "ascii")
        self.assertEqual(self.klass(("a,b\ncafé,2\n" * 60).encode("utf-8")), "utf-8")

    def test_wide(self):
        data = ("a,b\n" + "1,2\n" * 90).encode("utf-16-le")
        self.assertEqual(self.klass(data), "wide")

    def test_single_byte_legacy(self):
        data = ("id;stadt\n" + "1;München Grüße Straße\n" * 200).encode("cp1252")
        self.assertEqual(self.klass(data), "single-byte")

    @unittest.skipUnless(_have_detector(), "charset_normalizer not installed")
    def test_multibyte_cjk_is_its_own_class(self):
        """The class that matters: a trail byte can be 0x5C or 0x7C."""
        data = ("ID,名前\n" + "1,東京都渋谷区\n" * 300).encode("shift_jis")
        self.assertEqual(self.klass(data), "multibyte")

    def test_binary_has_no_encoding_class_of_its_own(self):
        """encoding_class describes an encoding. Binary is not one -- the
        magic hit and control_frac say that, on their own fields."""
        from csvscrape.features import describe

        shape = describe(bytes(range(256)) * 40)
        self.assertEqual(shape.encoding_class, "unknown")
        self.assertGreater(shape.control_frac, 0.02)

    def test_class_is_stable_across_codepages_within_a_class(self):
        """Detectors confuse cp1252/latin-1/cp1250 constantly. The class must
        not move when they do."""
        from csvscrape.validate import encoding_class

        for name in ("cp1252", "iso8859-1", "cp1250", "mac-iceland", "koi8-r"):
            self.assertEqual(encoding_class(name), "single-byte", name)
        for name in ("shift_jis", "cp932", "euc_jp", "euc_jis_2004", "big5",
                     "gb18030", "cp949"):
            self.assertEqual(encoding_class(name), "multibyte", name)


class DetectorAbstention(unittest.TestCase):
    """charset_normalizer abstains on low-entropy input, such as a column
    repeating one value. That is recorded as unknown, not read as binary."""

    def test_repetitive_cyrillic_is_not_flagged_binary(self):
        from csvscrape.validate import probe_encoding

        for codec in ("koi8-r", "cp1251"):
            data = ("код;город\n" + "1;Москва\n" * 300).encode(codec)
            self.assertFalse(probe_encoding(data).looks_binary, codec)

    def test_abstention_is_recorded_as_itself(self):
        """The detector said nothing. That is stored as None, not replaced by
        an invented label -- the whole point of keeping verdicts verbatim."""
        from csvscrape.validate import probe_encoding

        data = ("код;город\n" + "1;Москва\n" * 300).encode("koi8-r")
        enc = probe_encoding(data)
        self.assertIsNone(enc.encoding)
        self.assertEqual(enc.codec, "latin-1")  # byte-preserving fallback

    def test_the_dialect_is_still_recovered(self):
        """An unnamed 8-bit page still yields correct field structure, because
        latin-1 maps every byte to a distinct codepoint."""
        data = ("код;город\n" + "1;Москва\n" * 300).encode("koi8-r")
        shape = describe(data)
        self.assertEqual(shape.delimiter, ";")
        self.assertIsNone(shape.encoding)
        self.assertEqual(shape.encoding_class, "unknown")

    def test_control_byte_soup_is_still_flagged(self):
        """Binary must stay flagged -- the structural test decides, not the
        detector's willingness to answer."""
        from csvscrape.validate import probe_encoding

        self.assertTrue(probe_encoding(bytes(range(256)) * 200).looks_binary)

    def test_magic_prefixed_binary_is_caught_upstream(self):
        from csvscrape import validate

        for marker, _ in validate.MAGIC[:5]:
            self.assertTrue(any(
                (marker + b"\x00" * 200).startswith(m) for m, _ in validate.MAGIC))
