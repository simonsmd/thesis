//! Dialect conformance, run against every [`Dfa`](csv_parser::dfa::Dfa).
//!
//! See `tests/harness/mod.rs` for how a case is dispatched to each DFA.

mod harness;

use csv_parser::{
    config::{Config, QuoteStyle, RecordTerminator},
    parser::Error,
};
use harness::{parse, parse_with_capacity, read_after_error, rows, try_parse};

// ---- basics -----------------------------------------------------------

#[test]
fn simple_records() {
    let config = Config::new();
    assert_eq!(
        parse(&config, b"a,b,c\nd,e,f\n"),
        rows(&[&["a", "b", "c"], &["d", "e", "f"]]),
    );
}

#[test]
fn no_trailing_newline() {
    // A final record without a terminator is still emitted.
    let config = Config::new();
    assert_eq!(parse(&config, b"a,b,c"), rows(&[&["a", "b", "c"]]));
}

#[test]
fn empty_fields() {
    let config = Config::new();
    assert_eq!(parse(&config, b"a,,c\n"), rows(&[&["a", "", "c"]]));
    assert_eq!(parse(&config, b",\n"), rows(&[&["", ""]]));
}

#[test]
fn trailing_empty_field() {
    // A trailing delimiter before a terminator yields an empty last field.
    let config = Config::new();
    assert_eq!(parse(&config, b"a,\n"), rows(&[&["a", ""]]));
}

#[test]
fn trailing_empty_field_at_eof() {
    // A trailing delimiter with no terminator still yields the empty field.
    let config = Config::new();
    assert_eq!(parse(&config, b"a,"), rows(&[&["a", ""]]));
    assert_eq!(parse(&config, b"a,b,"), rows(&[&["a", "b", ""]]));
    assert_eq!(parse(&config, b","), rows(&[&["", ""]]));
}

#[test]
fn whitespace_is_significant() {
    let config = Config::new();
    assert_eq!(parse(&config, b" a , b \n"), rows(&[&[" a ", " b "]]));
}

// ---- buffering --------------------------------------------------------

#[test]
fn refills_in_the_middle_of_a_record() {
    // Every buffer size must give the same records: a record that does not
    // fit grows the buffer instead of being cut in two.
    let input = b"a,b,c\n\"quoted, with a delimiter\",\"esc\\\"aped\",x\n#comment\nlast,record\n";
    let expected = rows(&[
        &["a", "b", "c"],
        &["quoted, with a delimiter", "esc\"aped", "x"],
        &["last", "record"],
    ]);

    for capacity in [1, 2, 3, 5, 8, 16, 64, 4096] {
        let parsed = parse_with_capacity(&Config::new(), input, capacity).unwrap();
        assert_eq!(parsed, expected, "with a {capacity} byte buffer");
    }
}

#[test]
fn a_record_longer_than_the_buffer_grows_it() {
    let field = "x".repeat(10_000);
    let input = format!("{field},{field}\nsmall,record\n");
    let parsed = parse_with_capacity(&Config::new(), input.as_bytes(), 8).unwrap();
    assert_eq!(parsed, rows(&[&[&field, &field], &["small", "record"]]));
}

#[test]
fn fields_around_the_skip_block_size() {
    // read_record_simd skips field content in blocks of 64 bytes, so put field ends, closing
    // quotes, escapes and doubled quotes on both sides of a block boundary and of a refill.
    for len in [1, 62, 63, 64, 65, 66, 127, 128, 129, 200] {
        let plain = "p".repeat(len);
        let quoted: String = "q,\n".chars().cycle().take(len).collect();
        let escaped = format!("{}\\,eee", "e".repeat(len));
        let doubled = format!("\"{}\"\"ddd\"", "d".repeat(len));
        let input = format!("{plain},\"{quoted}\",{escaped},{doubled}\r\n{plain}\r\n");

        let escaped = format!("{},eee", "e".repeat(len));
        let doubled = format!("{}\"ddd", "d".repeat(len));
        let expected = rows(&[&[&plain, &quoted, &escaped, &doubled], &[&plain]]);

        for capacity in [1, 7, 64, 65, 300, 4096] {
            let parsed = parse_with_capacity(&Config::new(), input.as_bytes(), capacity).unwrap();
            assert_eq!(parsed, expected, "{len} byte fields with a {capacity} byte buffer");
        }
    }
}

#[test]
fn records_with_many_fields() {
    // Crosses the parser's field slot growth, which starts at 1024, both
    // mid-record and on the final field at end of input.
    for count in [1, 1023, 1024, 1025, 2047, 2048, 2049, 5000] {
        let fields: Vec<String> = (0..count).map(|i| i.to_string()).collect();
        let record: Vec<&str> = fields.iter().map(String::as_str).collect();
        let line = fields.join(",");
        let expected = rows(&[&record, &["x"]]);

        let parsed = parse(&Config::new(), format!("{line}\nx\n").as_bytes());
        assert_eq!(parsed, expected, "{count} fields");
        let parsed = parse(&Config::new(), format!("x\n{line}").as_bytes());
        assert_eq!(parsed, rows(&[&["x"], &record]), "{count} fields at eof");
    }
}

// ---- terminators ------------------------------------------------------

#[test]
fn crlf_terminator() {
    let config = Config::new();
    assert_eq!(
        parse(&config, b"a,b\r\nc,d\r\n"),
        rows(&[&["a", "b"], &["c", "d"]]),
    );
}

#[test]
fn bare_cr_terminator() {
    let config = Config::new();
    assert_eq!(parse(&config, b"a\rb\r"), rows(&[&["a"], &["b"]]));
}

#[test]
fn custom_byte_terminator_keeps_cr() {
    // With a plain '\n' terminator, '\r' is an ordinary field byte.
    let config = Config::new().terminator(RecordTerminator::Byte(b'\n'));
    assert_eq!(parse(&config, b"a\r\nb\n"), rows(&[&["a\r"], &["b"]]));
}

#[test]
fn blank_lines_are_skipped() {
    let config = Config::new();
    assert_eq!(parse(&config, b"a\n\r\r\n\n\rb\n"), rows(&[&["a"], &["b"]]));
}

// ---- quoting ----------------------------------------------------------

#[test]
fn quoted_fields() {
    let config = Config::new();
    assert_eq!(parse(&config, b"\"a\",\"b\"\n"), rows(&[&["a", "b"]]));
}

#[test]
fn quoted_delimiter_and_newline() {
    // Delimiters and newlines inside quotes are literal.
    let config = Config::new();
    assert_eq!(
        parse(&config, b"\"a,b\",\"c\nd\"\n"),
        rows(&[&["a,b", "c\nd"]]),
    );
}

#[test]
fn quoted_empty_field_vs_blank_line() {
    // "" is one empty field; a blank line is no record at all.
    let config = Config::new();
    assert_eq!(parse(&config, b"\"\"\n"), rows(&[&[""]]));
}

#[test]
fn doubled_quote_escape() {
    // "a""b" -> a"b  (RFC 4180 doubling)
    let config = Config::new();
    assert_eq!(parse(&config, b"\"a\"\"b\",c\n"), rows(&[&["a\"b", "c"]]));
}

#[test]
fn concat_quoted_then_unquoted() {
    // Concat mode joins a quoted region with adjacent unquoted text.
    let config = Config::new();
    assert_eq!(parse(&config, b"\"a\"b,c\n"), rows(&[&["ab", "c"]]));
}

#[test]
fn concat_unquoted_then_quoted() {
    // ...and a quote can open a quoted region mid-field.
    let config = Config::new();
    assert_eq!(parse(&config, b"a\"b\",c\n"), rows(&[&["ab", "c"]]));
}

#[test]
fn toggle_with_double_quote_escape() {
    // handle a double quote as a literal quote in toggle mode
    let config = Config::new();
    assert_eq!(
        parse(&config, b"\"a\"\"b,\",c,\"a\"\"\"b,d\n"),
        rows(&[&["a\"b,", "c", "a\"b", "d"]]),
    );
}

#[test]
fn strict_mode_rejects_inner_quote() {
    // In Strict mode a quote that isn't at field start is a parser error,
    // reported at the offending byte.
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let err = try_parse(&config, b"a\"b\",c\n").unwrap_err();
    match err {
        Error::Parser(pos) => assert_eq!(pos.byte, 1),
        other => panic!("expected parser error, got {other:?}"),
    }
}

#[test]
fn strict_mode_rejects_trailing_garbage_after_quote() {
    // After a closing quote, only a delimiter or terminator is allowed.
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let err = try_parse(&config, b"\"ab\"c\n").unwrap_err();
    assert!(matches!(err, Error::Parser(_)));
}

#[test]
fn strict_mode_rejects_unclosed_quote_at_eof() {
    // An unclosed quote at EOF is a parser error
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let err = try_parse(&config, b"\"ab").unwrap_err();
    assert!(matches!(err, Error::Parser(_)));
}

#[test]
fn unclosed_escape_in_quoted_field_at_eof() {
    // An escape at the end of the input has nothing left to escape, so it is an
    // error whatever the quote style, unlike an unclosed quote on its own.
    for style in [
        QuoteStyle::Toggle,
        QuoteStyle::Strict(false),
        QuoteStyle::Strict(true),
    ] {
        let config = Config::new().quote_style(style.clone());
        let err = try_parse(&config, b"\"ab\\").unwrap_err();
        assert!(matches!(err, Error::Parser(_)), "{style:?}");
    }
}

#[test]
fn unclosed_escape_in_unquoted_field_at_eof() {
    // The same holds for an escape in a field that was never quoted.
    for style in [
        QuoteStyle::Toggle,
        QuoteStyle::Strict(false),
        QuoteStyle::Strict(true),
    ] {
        let config = Config::new().quote_style(style.clone());
        let err = try_parse(&config, b"a,b\\").unwrap_err();
        assert!(matches!(err, Error::Parser(_)), "{style:?}");
    }
}

#[test]
fn trailing_backslash_without_unquoted_escapes_at_eof() {
    // With escapes off in unquoted fields the backslash is an ordinary byte, so
    // ending on it is fine.
    let config = Config::new().escape_in_unquoted_fields(false);
    assert_eq!(parse(&config, b"a,b\\"), rows(&[&["a", "b\\"]]));
}

#[test]
fn unterminated_quote_at_eof() {
    // An unclosed quote at EOF yields whatever was accumulated.
    let config = Config::new();
    assert_eq!(parse(&config, b"\"abc"), rows(&[&["abc"]]));
}

#[test]
fn disabled_quoting() {
    let config = Config::new().quote(None);
    assert_eq!(parse(&config, b"\"a\",b\n"), rows(&[&["\"a\"", "b"]]));
}

#[test]
fn disabled_double_quote_escape() {
    let config = Config::new().double_quote_escape(false);
    assert_eq!(parse(&config, b"\"a\"\"b\",b\n"), rows(&[&["ab", "b"]]));
}

// ---- escapes ----------------------------------------------------------

#[test]
fn escaped_delimiter_unquoted() {
    // Backslash escapes a delimiter in an unquoted field.
    let config = Config::new();
    assert_eq!(parse(&config, b"a\\,b,c\n"), rows(&[&["a,b", "c"]]));
}

#[test]
fn escaped_quote_in_quoted_field() {
    let config = Config::new();
    assert_eq!(parse(&config, b"\"a\\\"b\",c\n"), rows(&[&["a\"b", "c"]]));
}

#[test]
fn escaped_backslash() {
    let config = Config::new();
    assert_eq!(parse(&config, b"a\\\\b\n"), rows(&[&["a\\b"]]));
}

#[test]
fn disabled_escape_keeps_backslash() {
    let config = Config::new().escape(None);
    assert_eq!(parse(&config, b"a\\b\n"), rows(&[&["a\\b"]]));
}

// ---- comments ---------------------------------------------------------

#[test]
fn comment_lines_are_skipped() {
    let config = Config::new();
    assert_eq!(
        parse(&config, b"#comment\na,b\n#another\nc,d\n"),
        rows(&[&["a", "b"], &["c", "d"]]),
    );
}

#[test]
fn hash_is_literal_mid_field() {
    // '#' only starts a comment at the beginning of a record.
    let config = Config::new();
    assert_eq!(parse(&config, b"a#b,c\n"), rows(&[&["a#b", "c"]]));
}

#[test]
fn comment_ends_at_newline_with_a_custom_terminator() {
    // A comment is closed by a literal '\n' whatever the terminator is, so
    // '\n' stays significant even when it does not end records.
    let config = Config::new().terminator(RecordTerminator::Byte(b';'));
    assert_eq!(
        parse(&config, b"#comment\na,b;c,d;"),
        rows(&[&["a", "b"], &["c", "d"]]),
    );
}

// ---- strict-mode literal quotes in unquoted fields --------------------

#[test]
fn strict_true_allows_literal_quote_in_unquoted_field() {
    // Strict(true): a quote that isn't at field start is an ordinary byte.
    let config = Config::new().quote_style(QuoteStyle::Strict(true));
    assert_eq!(parse(&config, b"a\"b,c\n"), rows(&[&["a\"b", "c"]]));
}

#[test]
fn strict_true_still_treats_leading_quote_as_quoting() {
    // The flag only affects mid-field quotes; a field that opens with a
    // quote is still a quoted field.
    let config = Config::new().quote_style(QuoteStyle::Strict(true));
    assert_eq!(parse(&config, b"\"a,b\",c\n"), rows(&[&["a,b", "c"]]));
}

#[test]
fn strict_true_still_rejects_garbage_after_close() {
    // The flag does not relax the post-close rule: after a *closing* quote
    // only a delimiter or terminator is allowed.
    let config = Config::new().quote_style(QuoteStyle::Strict(true));
    let err = try_parse(&config, b"\"ab\"c\n").unwrap_err();
    assert!(matches!(err, Error::Parser(_)));
}

// ---- escape_in_unquoted_fields ---------------------------------------

#[test]
fn escape_disabled_in_unquoted_fields() {
    // With escapes off in unquoted fields, a backslash there is literal...
    let config = Config::new().escape_in_unquoted_fields(false);
    assert_eq!(parse(&config, b"a\\b\n"), rows(&[&["a\\b"]]));
    // ...but escapes still apply inside quoted fields.
    assert_eq!(parse(&config, b"\"a\\\"b\"\n"), rows(&[&["a\"b"]]));
}

// ---- error position reporting ----------------------------------------

#[test]
fn parser_error_reports_record_and_field() {
    // Error in the 2nd record (index 1), 2nd field (index 1).
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let err = try_parse(&config, b"a,b,c\nd,e\"f\n").unwrap_err();
    match err {
        Error::Parser(pos) => {
            assert_eq!(pos.record, 1);
            assert_eq!(pos.field, 1);
        }
        other => panic!("expected parser error, got {other:?}"),
    }
}

#[test]
fn parser_error_byte_offset_across_records() {
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let err = try_parse(&config, b"a,b,c\nd,e\"f\n").unwrap_err();
    match err {
        // The offending quote is at absolute byte 9, not 3.
        Error::Parser(pos) => assert_eq!(pos.byte, 9),
        other => panic!("expected parser error, got {other:?}"),
    }
}

#[test]
fn parser_error_is_sticky() {
    // The input after an error is not a record start, so reading on reports the
    // same error again instead of resuming in the middle of the broken record.
    let config = Config::new().quote_style(QuoteStyle::Strict(false));
    let (first, second) = read_after_error(&config, b"a,b\nc\"d,e\nf,g\n");
    assert!(first.starts_with("Err(Parser("), "{first}");
    assert_eq!(first, second);
}

// ---- custom configuration --------------------------------------------

#[test]
fn custom_delimiter_and_quote() {
    let config = Config::new().delimiter(b'\t').quote(Some(b'\''));
    assert_eq!(parse(&config, b"'a\tb'\tc\n"), rows(&[&["a\tb", "c"]]));
}
