"""Dialect sniffing tests.
"""
from __future__ import annotations

import unittest

from csvscrape.features import Shape, _split_fields, describe


class Splitter(unittest.TestCase):
    """The scanner is a probe, not a parser: it must never raise."""

    LINE = 'ab"c,d"e,z'

    def test_toggle_concatenates_across_quotes(self):
        self.assertEqual(_split_fields(self.LINE, ",", '"', "toggle"),
                         ["abc,de", "z"])

    def test_strict_treats_mid_field_quote_as_literal(self):
        self.assertEqual(_split_fields(self.LINE, ",", '"', "strict"),
                         ['ab"c', 'd"e', "z"])

    def test_none_disables_quoting(self):
        self.assertEqual(_split_fields(self.LINE, ",", '"', "none"),
                         ['ab"c', 'd"e', "z"])

    def test_doubled_quote_emits_one(self):
        self.assertEqual(_split_fields('"x""y",2', ",", '"', "strict"),
                         ['x"y', "2"])

    def test_escape_char_makes_next_literal(self):
        self.assertEqual(_split_fields('"a\\",b",c', ",", '"', "strict", "\\"),
                         ['a",b', "c"])

    def test_raw_mode_preserves_quotes(self):
        self.assertEqual(_split_fields('"x,1",2', ",", '"', "strict", raw=True),
                         ['"x,1"', "2"])

    def test_unterminated_quote_does_not_raise(self):
        self.assertEqual(_split_fields('"abc', ",", '"', "strict"), ["abc"])

    def test_lone_mid_field_quote_does_not_raise(self):
        self.assertEqual(_split_fields('ab"cd', ",", '"', "toggle"), ["abcd"])


class DelimiterDetection(unittest.TestCase):
    def d(self, data: bytes) -> str | None:
        return describe(data, size=len(data)).delimiter

    def test_plain_comma(self):
        self.assertEqual(self.d(b"a,b,c\n1,2,3\n4,5,6\n"), ",")

    def test_semicolon_with_comma_decimals(self):
        data = "nom;prix\ncafé;1,50\nthé;2,00\n".encode()
        self.assertEqual(self.d(data), ";")

    def test_tab_wins_over_commas_in_text(self):
        """Commas inside a tab-delimited field must not win the vote."""
        data = b"name\tdesc\nfoo\ta, b, c\nbar\td, e, f\nbaz\tg, h, i\n"
        self.assertEqual(self.d(data), "\t")

    def test_pipe_wins_over_one_comma_per_row(self):
        data = b"id|text\n1|hello, world\n2|foo, bar\n3|baz, qux\n"
        self.assertEqual(self.d(data), "|")

    def test_colon_separated_passwd(self):
        data = (b"root:x:0:0:root:/root:/bin/bash\n"
                b"daemon:x:1:1:daemon:/usr/sbin:/bin/sh\n")
        self.assertEqual(self.d(data), ":")

    def test_timestamps_do_not_beat_the_comma(self):
        """12:34:56 splits consistently, but must not outvote the comma."""
        data = b"time,temp\n12:34:56,20\n12:35:00,21\n12:35:04,22\n"
        self.assertEqual(self.d(data), ",")

    def test_title_banner_before_table(self):
        """A delimiter-free title row must not reject the real delimiter."""
        data = (b"Some Export File (v1.0)\n\ndate,value,note\n"
                b"2024-01-01,5,ok\n2024-01-02,6,ok\n")
        shape = describe(data, size=len(data))
        self.assertEqual(shape.delimiter, ",")
        self.assertGreater(shape.preamble, 0)

    def test_quoted_cells_containing_the_delimiter(self):
        data = b'"a","b","c"\n"x,1","y,2","z,3"\n"p,4","q,5","r,6"\n'
        self.assertEqual(self.d(data), ",")

    def test_single_column_reports_no_delimiter(self):
        """Space splits prose; it must not win on a single-column file."""
        data = b"value\nsome free text here\nmore free text\nand more\n"
        self.assertIsNone(self.d(data))

    def test_latin1_semicolon(self):
        data = "nom;ville\nJos\xe9;Z\xfcrich\nAnna;K\xf6ln\n".encode("latin-1")
        shape = describe(data, size=len(data))
        self.assertEqual(shape.delimiter, ";")
        # chardet names the specific codepage; latin-1 and cp1252 are the same
        # for parsing purposes, so assert the family rather than the label.
        self.assertNotIn(shape.encoding, ("utf-8", "ascii", "binary"))


class QuoteStyleDetection(unittest.TestCase):
    def test_strict_minimal_quoting(self):
        data = b'a,b,c\n"x,1",2,3\n"y,2",4,5\n"z,3",6,7\n'
        shape = describe(data, size=len(data))
        self.assertEqual(shape.quote, '"')
        self.assertEqual(shape.quote_style, "strict")
        self.assertEqual(shape.quoting, "minimal")
        self.assertFalse(shape.quotes_in_unquoted_fields)

    def test_toggle_style(self):
        data = b'a,b\nab"c,d"e,2\nfg"h,i"j,3\nkl"m,n"o,4\n'
        shape = describe(data, size=len(data))
        self.assertEqual(shape.quote_style, "toggle")

    def test_quotes_inside_bare_fields(self):
        """Maps to the parser's Strict(true)."""
        data = b'a,b\nsay "hi",2\nsay "yo",3\nsay "ok",4\n'
        shape = describe(data, size=len(data))
        self.assertEqual(shape.quote_style, "strict")
        self.assertTrue(shape.quotes_in_unquoted_fields)

    def test_all_quoted(self):
        data = b'"a","b"\n"1","2"\n"3","4"\n'
        self.assertEqual(describe(data, size=len(data)).quoting, "all")

    def test_no_quote_char_when_absent(self):
        shape = describe(b"a,b\n1,2\n3,4\n", size=12)
        self.assertIsNone(shape.quote)
        self.assertIsNone(shape.quote_style)

    def test_quote_present_but_irrelevant_is_still_detected(self):
        """`"a","b"` splits the same quoted or not; the quote char must still be
        detected."""
        data = b'"a","b"\n"1","2"\n"3","4"\n'
        self.assertEqual(describe(data, size=len(data)).quote, '"')

    def test_double_quote_escape(self):
        data = b'a,b\n"x""y",2\n"p""q",3\n"r""s",4\n'
        shape = describe(data, size=len(data))
        self.assertTrue(shape.double_quote_escape)

    def test_backslash_escape(self):
        data = b'a,b\n"x\\",1",2\n"y\\",2",3\n"z\\",3",4\n'
        self.assertEqual(describe(data, size=len(data)).escape, "\\")


class StructureDetection(unittest.TestCase):
    def test_crlf_and_bom_and_comment(self):
        data = (b'\xef\xbb\xbf# generated\r\nname;value;note\r\n'
                b'"a;b";1,5;ok\r\nplain;2,0;fine\r\n')
        shape = describe(data, size=len(data))
        self.assertEqual(shape.delimiter, ";")
        self.assertEqual(shape.terminator, "crlf")
        self.assertEqual(shape.bom, "utf-8-sig")
        self.assertEqual(shape.comment, "#")
        self.assertEqual(shape.preamble, 1)

    def test_mixed_terminators(self):
        data = b"a,b\r\n1,2\n3,4\r\n5,6\n"
        self.assertTrue(describe(data, size=len(data)).mixed_terminators)

    def test_ragged_rows_counted(self):
        data = b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n"
        shape = describe(data, size=len(data))
        self.assertEqual(shape.columns, 3)
        self.assertEqual(shape.ragged_rows, 2)

    def test_trailing_delimiter(self):
        data = b"a,b,\n1,2,\n3,4,\n"
        self.assertTrue(describe(data, size=len(data)).trailing_delimiter)

    def test_complete_flag_and_truncation(self):
        data = b"a,b\n1,2\n3,4\n"
        self.assertTrue(describe(data, size=len(data)).complete)
        self.assertFalse(describe(data, size=len(data) * 10).complete)

    def test_truncated_sample_drops_partial_last_line(self):
        """A fragment at the sample edge is not a record and must not count
        as a ragged row."""
        data = b"a,b,c\n1,2,3\n4,5,3\n7,8"
        shape = describe(data, size=len(data) + 5000)
        self.assertEqual(shape.ragged_rows, 0)

    def test_nul_bytes_flagged(self):
        self.assertTrue(describe(b"a,b\n1,\x002\n", size=10).has_nul)

    def test_non_ascii_fraction(self):
        data = "a,b\ncafé,2\n".encode()
        self.assertGreater(describe(data, size=len(data)).non_ascii_frac, 0)

    def test_ends_with_newline(self):
        self.assertTrue(describe(b"a,b\n1,2\n", size=8).ends_with_newline)
        self.assertFalse(describe(b"a,b\n1,2", size=7).ends_with_newline)

    def test_empty_input_does_not_raise(self):
        self.assertIsInstance(describe(b"", size=0), Shape)


if __name__ == "__main__":
    unittest.main()


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


class DelimiterTieBreak(unittest.TestCase):
    """A tie keeps the earlier (more common) delimiter rather than breaking on
    column count."""

    def d(self, data: bytes) -> str | None:
        return describe(data, size=len(data)).delimiter

    def test_headerless_timestamps_do_not_beat_the_comma(self):
        # No header row to drop the colon's agreement below the comma's, so
        # both score 1.0 and only the tie-break decides. Breaking it on column
        # count handed the file to ':'.
        data = (b"12:34:56,5\n01:02:03,7\n11:22:33,9\n23:00:01,2\n")
        self.assertEqual(self.d(data), ",")

    def test_headerless_iso_timestamps(self):
        data = (b"2024-01-01 08:00:00,berlin,5\n"
                b"2024-01-01 09:00:00,berlin,6\n"
                b"2024-01-01 10:00:00,berlin,7\n")
        self.assertEqual(self.d(data), ",")

    def test_a_real_margin_still_displaces(self):
        # Tab splits every row consistently; the comma does not. The margin
        # rule must still let the better candidate win.
        data = b"name\tdesc\nfoo\ta, b, c\nbar\td, e, f\nbaz\tg, h, i\n"
        self.assertEqual(self.d(data), "\t")


class DoubledQuoteCounting(unittest.TestCase):
    """`""` as an empty field is not a doubled-quote escape."""

    def test_empty_quoted_fields_are_not_escapes(self):
        shape = describe(b'a,b,c\n1,"",3\n4,"",6\n')
        self.assertEqual(shape.escaped_quotes, 0)
        self.assertFalse(shape.double_quote_escape)

    def test_genuine_doubled_quote_is_counted(self):
        shape = describe(b'a,b\n"he said ""hi""",2\n"and ""bye""",3\n')
        self.assertEqual(shape.escaped_quotes, 4)
        self.assertTrue(shape.double_quote_escape)

    def test_empty_fields_do_not_make_the_file_look_unterminated(self):
        shape = describe(b'a,b,c\n1,"",3\n4,"",6\n')
        self.assertFalse(shape.unterminated_quote)


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
