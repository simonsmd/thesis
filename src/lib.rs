use std::fmt::Display;
use std::io::Read;
use std::str::Utf8Error;

mod buffer;
mod config;

pub use buffer::{Buffer, DEFAULT_CAPACITY, Refill};

#[derive(Debug, Clone)]
pub struct Config {
    pub delimiter: u8,
    pub terminator: RecordTerminator,
    pub quote: Option<u8>,
    pub quote_style: QuoteStyle,
    pub escape: Option<u8>,
    pub escape_in_unquoted_fields: bool,
    pub double_quote_escape: bool,
    pub comment: Option<u8>,
    // pub ignore_emtpy_lines: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            delimiter: b',',
            terminator: RecordTerminator::CRLF,
            quote: Some(b'"'),
            quote_style: QuoteStyle::Toggle,
            escape: Some(b'\\'),
            escape_in_unquoted_fields: true,
            double_quote_escape: true,
            comment: Some(b'#'),
            // ignore_emtpy_lines: true,
        }
    }
}

#[derive(Debug)]
pub struct Parser<R: Read> {
    state: State,
    position: Position,
    reader: R,
    buffer: Buffer,
    record_start: usize,
    read_index: usize,
    write_index: usize,
    field_ends: Vec<usize>,
    pub config: Config,
}

impl<R: Read> Parser<R> {
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, Config::default())
    }

    pub fn with_config(reader: R, config: Config) -> Self {
        Self::with_capacity(reader, config, DEFAULT_CAPACITY)
    }

    /// `capacity` is the read size; the buffer only grows past it for records
    /// that do not fit.
    pub fn with_capacity(reader: R, config: Config, capacity: usize) -> Self {
        Self {
            state: State::StartRecord,
            position: Position::default(),
            reader,
            buffer: Buffer::with_capacity(capacity),
            record_start: 0,
            read_index: 0,
            write_index: 0,
            field_ends: Vec::new(),
            config,
        }
    }

    pub fn position(&self) -> Position {
        self.position
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Position {
    pub byte: usize,
    pub record: usize,
    pub field: usize,
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "record {}, field {}, byte {}", self.record, self.field, self.byte)
    }
}

#[derive(Debug, Clone)]
pub enum RecordTerminator {
    CRLF,
    Byte(u8),
}

impl RecordTerminator {
    fn equals(&self, value: u8) -> bool {
        match self {
            Self::CRLF => value == b'\n' || value == b'\r',
            Self::Byte(b) => value == *b,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuoteStyle {
    /// Quotes can only start at the beginning of a field, otherwise quotes are not handled
    /// The boolean flag describes if quotes are allowed in unquoted fields
    Strict(bool),
    /// A quote character toggles the quoting mode anywhere inside a field
    Toggle, // Concat, Always?
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum State {
    StartRecord,
    InField,
    InQuotedField,
    EscapeSequenceInField,
    EscapeSequenceInQuotedField,
    AfterQuote,
    InComment,
    EndField,
    EndRecord,
    Error,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TransitionAction {
    Discard,
    Output,
    Epsilon,
}

#[derive(Debug)]
pub enum Error {
    Parser(Position),
    IO(std::io::Error)
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parser(position) => write!(f, "parser error at {position}"),
            Error::IO(error) => write!(f, "IO error: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parser(_) => None,
            Error::IO(error) => Some(error),
        }
    }
}

impl Config {
    fn transition(&self, state: &State, input: u8) -> (State, TransitionAction) {
        use State::*;
        use TransitionAction::*;
        match state {
            StartRecord => {
                if self.terminator.equals(input) {
                    (StartRecord, Discard)
                } else if self.comment == Some(input) {
                    (InComment, Discard)
                } else if self.quote == Some(input) {
                    (InQuotedField, Discard)
                } else {
                    (InField, Epsilon)
                }
            }
            InField => {
                if input == self.delimiter {
                    (EndField, Discard)
                } else if self.terminator.equals(input) {
                    (EndRecord, Discard)
                } else if self.quote == Some(input) {
                    match self.quote_style {
                        QuoteStyle::Strict(false) => (Error, Discard),
                        QuoteStyle::Strict(true) => (InField, Output),
                        QuoteStyle::Toggle => (InQuotedField, Discard),
                    }
                } else if self.escape_in_unquoted_fields && self.escape == Some(input) {
                    (EscapeSequenceInField, Discard)
                } else {
                    (InField, Output)
                }
            }
            InQuotedField => {
                if self.quote == Some(input) {
                    (AfterQuote, Discard)
                } else if self.escape == Some(input) {
                    (EscapeSequenceInQuotedField, Discard)
                } else {
                    (InQuotedField, Output)
                }
            }
            EscapeSequenceInField => (InField, Output),
            EscapeSequenceInQuotedField => (InQuotedField, Output),
            AfterQuote => {
                if self.double_quote_escape && self.quote == Some(input) {
                    (InQuotedField, Output)
                } else if self.delimiter == input {
                    (EndField, Discard)
                } else if self.terminator.equals(input) {
                    (EndRecord, Discard)
                } else if matches!(self.quote_style, QuoteStyle::Strict(_)) {
                    (Error, Discard)
                } else {
                    (InField, Epsilon)
                }
            }
            InComment => {
                if input == b'\n' {
                    (StartRecord, Discard)
                } else {
                    (InComment, Discard)
                }
            }
            EndField => {
                if self.quote == Some(input) {
                    (InQuotedField, Discard)
                } else {
                    (InField, Epsilon)
                }
            },
            EndRecord => (StartRecord, Epsilon),
            Error => (Error, Discard),
        }
    }
}

impl<R: Read> Parser<R> {
    pub fn read_record(&mut self) -> Result<Option<Record<'_>>, Error> {
        self.record_start = self.write_index;
        self.field_ends.clear();
        self.position.field = 0;

        'record: loop {
            if self.read_index == self.buffer.len() {
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, self.record_start, self.write_index)
                    .map_err(Error::IO)?;
                self.record_start -= shift;
                self.write_index -= shift;
                self.read_index = self.write_index;

                if read == 0 {
                    // TODO: any other states that should result in an error?
                    if self.state == State::InQuotedField
                        && matches!(self.config.quote_style, QuoteStyle::Strict(_))
                    {
                        self.position.byte = self.buffer.source_offset(self.read_index);
                        return Err(Error::Parser(self.position));
                    }

                    if self.write_index == self.record_start && self.field_ends.is_empty() {
                        return Ok(None);
                    }
                    self.field_ends.push(self.write_index - self.record_start);
                    break 'record;
                }
            }

            let data = self.buffer.data_mut();
            while self.read_index < data.len() {
                let (state, action) = self.config.transition(&self.state, data[self.read_index]);
                self.state = state;

                if self.state == State::Error {
                    break 'record;
                }

                // TODO: is this or branch faster?
                data[self.write_index] = data[self.read_index];
                self.write_index += (action == TransitionAction::Output) as usize;
                self.read_index += (action != TransitionAction::Epsilon) as usize;

                if self.state == State::EndField || self.state == State::EndRecord {
                    self.field_ends.push(self.write_index - self.record_start);
                    self.position.field += 1;
                }

                if self.state == State::EndRecord {
                    break 'record;
                }
            }
        }

        self.position.byte = self.buffer.source_offset(self.read_index);
        if self.state == State::Error {
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;
        Ok(Some(self.record()))
    }

    /// The record assembled so far.
    fn record(&self) -> Record<'_> {
        Record {
            buffer: &self.buffer.data()[self.record_start..self.write_index],
            field_ends: &self.field_ends,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    buffer: &'a [u8],
    field_ends: &'a [usize],
}

impl<'a> Record<'a> {
    /// Number of fields in the record.
    pub fn len(&self) -> usize {
        self.field_ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.field_ends.is_empty()
    }

    /// The raw bytes of field `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        let end = *self.field_ends.get(index)?;
        let start = if index == 0 {
            0
        } else {
            self.field_ends[index - 1]
        };
        Some(&self.buffer[start..end])
    }

    /// Iterate over the fields as byte slices.
    pub fn iter(&self) -> Fields<'a> {
        Fields {
            buffer: self.buffer,
            field_ends: self.field_ends.iter(),
            start: 0,
        }
    }

    pub fn get_str(&self, i: usize) -> Option<Result<&'a str, Utf8Error>> {
        self.get(i).map(std::str::from_utf8)
    }

    pub fn iter_str(&self) -> impl Iterator<Item = Result<&'a str, Utf8Error>> {
        self.iter().map(std::str::from_utf8)
    }
}

impl<'a> IntoIterator for Record<'a> {
    type Item = &'a [u8];
    type IntoIter = Fields<'a>;

    fn into_iter(self) -> Fields<'a> {
        self.iter()
    }
}

/// Iterator over the fields of a [`Record`].
#[derive(Debug, Clone)]
pub struct Fields<'a> {
    buffer: &'a [u8],
    field_ends: std::slice::Iter<'a, usize>,
    start: usize,
}

impl<'a> Iterator for Fields<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let end = *self.field_ends.next()?;
        let field = &self.buffer[self.start..end];
        self.start = end;
        Some(field)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.field_ends.size_hint()
    }
}

impl ExactSizeIterator for Fields<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse `input` to completion, returning each record as a `Vec<String>`.
    /// Panics on any parser/IO error; use [`try_parse`] to inspect the error.
    fn parse(config: &Config, input: &[u8]) -> Vec<Vec<String>> {
        try_parse(config, input).unwrap()
    }

    /// Like [`parse`], but surfaces the first parser/IO error instead of
    /// panicking. Records parsed before the error are discarded.
    fn try_parse(config: &Config, input: &[u8]) -> Result<Vec<Vec<String>>, Error> {
        parse_with_capacity(config, input, DEFAULT_CAPACITY)
    }

    /// [`try_parse`] with an explicit buffer size, to exercise refills in the
    /// middle of a record.
    fn parse_with_capacity(
        config: &Config,
        input: &[u8],
        capacity: usize,
    ) -> Result<Vec<Vec<String>>, Error> {
        let mut parser = Parser::with_capacity(input, config.clone(), capacity);
        let mut out = Vec::new();
        while let Some(record) = parser.read_record()? {
            out.push(
                record
                    .iter()
                    .map(|f| String::from_utf8(f.to_vec()).unwrap())
                    .collect(),
            );
        }
        Ok(out)
    }

    /// Build the expected `Vec<Vec<String>>` from string literals.
    fn rows(expected: &[&[&str]]) -> Vec<Vec<String>> {
        expected
            .iter()
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect()
    }

    // ---- basics -----------------------------------------------------------

    #[test]
    fn simple_records() {
        let dfa = Config::default();
        assert_eq!(
            parse(&dfa, b"a,b,c\nd,e,f\n"),
            rows(&[&["a", "b", "c"], &["d", "e", "f"]]),
        );
    }

    #[test]
    fn no_trailing_newline() {
        // A final record without a terminator is still emitted.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a,b,c"), rows(&[&["a", "b", "c"]]));
    }

    #[test]
    fn empty_fields() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a,,c\n"), rows(&[&["a", "", "c"]]));
        assert_eq!(parse(&dfa, b",\n"), rows(&[&["", ""]]));
    }

    #[test]
    fn trailing_empty_field() {
        // A trailing delimiter before a terminator yields an empty last field.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a,\n"), rows(&[&["a", ""]]));
    }

    #[test]
    fn trailing_empty_field_at_eof() {
        // A trailing delimiter with no terminator still yields the empty field.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a,"), rows(&[&["a", ""]]));
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a,b,"), rows(&[&["a", "b", ""]]));
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b","), rows(&[&["", ""]]));
    }

    #[test]
    fn whitespace_is_significant() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b" a , b \n"), rows(&[&[" a ", " b "]]));
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
            let parsed = parse_with_capacity(&Config::default(), input, capacity).unwrap();
            assert_eq!(parsed, expected, "with a {capacity} byte buffer");
        }
    }

    #[test]
    fn a_record_longer_than_the_buffer_grows_it() {
        let field = "x".repeat(10_000);
        let input = format!("{field},{field}\nsmall,record\n");
        let parsed = parse_with_capacity(&Config::default(), input.as_bytes(), 8).unwrap();
        assert_eq!(
            parsed,
            rows(&[&[&field, &field], &["small", "record"]]),
        );
    }

    // ---- terminators ------------------------------------------------------

    #[test]
    fn crlf_terminator() {
        let dfa = Config::default();
        assert_eq!(
            parse(&dfa, b"a,b\r\nc,d\r\n"),
            rows(&[&["a", "b"], &["c", "d"]]),
        );
    }

    #[test]
    fn bare_cr_terminator() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a\rb\r"), rows(&[&["a"], &["b"]]));
    }

    #[test]
    fn custom_byte_terminator_keeps_cr() {
        // With a plain '\n' terminator, '\r' is an ordinary field byte.
        let mut dfa = Config::default();
        dfa.terminator = RecordTerminator::Byte(b'\n');
        assert_eq!(parse(&dfa, b"a\r\nb\n"), rows(&[&["a\r"], &["b"]]));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a\n\r\r\n\n\rb\n"), rows(&[&["a"], &["b"]]));
    }

    // ---- quoting ----------------------------------------------------------

    #[test]
    fn quoted_fields() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"a\",\"b\"\n"), rows(&[&["a", "b"]]));
    }

    #[test]
    fn quoted_delimiter_and_newline() {
        // Delimiters and newlines inside quotes are literal.
        let dfa = Config::default();
        assert_eq!(
            parse(&dfa, b"\"a,b\",\"c\nd\"\n"),
            rows(&[&["a,b", "c\nd"]]),
        );
    }

    #[test]
    fn quoted_empty_field_vs_blank_line() {
        // "" is one empty field; a blank line is no record at all.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"\"\n"), rows(&[&[""]]));
    }

    #[test]
    fn doubled_quote_escape() {
        // "a""b" -> a"b  (RFC 4180 doubling)
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"a\"\"b\",c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn concat_quoted_then_unquoted() {
        // Concat mode joins a quoted region with adjacent unquoted text.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"a\"b,c\n"), rows(&[&["ab", "c"]]));
    }

    #[test]
    fn concat_unquoted_then_quoted() {
        // ...and a quote can open a quoted region mid-field.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a\"b\",c\n"), rows(&[&["ab", "c"]]));
    }
    
    #[test]
    fn toggle_with_double_quote_escape() {
        // handle a double quote as a literal quote in toggle mode
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"a\"\"b,\",c,\"a\"\"\"b,d\n"), rows(&[&["a\"b,", "c", "a\"b", "d"]]));
    }

    #[test]
    fn strict_mode_rejects_inner_quote() {
        // In Strict mode a quote that isn't at field start is a parser error,
        // reported at the offending byte.
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&dfa, b"a\"b\",c\n").unwrap_err();
        match err {
            Error::Parser(pos) => assert_eq!(pos.byte, 1),
            other => panic!("expected parser error, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_rejects_trailing_garbage_after_quote() {
        // After a closing quote, only a delimiter or terminator is allowed.
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&dfa, b"\"ab\"c\n").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    #[test]
    fn strict_mode_rejects_unclosed_quote_at_eof() {
        // An unclosed quote at EOF is a parser error
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&dfa, b"\"ab").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    #[test]
    fn unterminated_quote_at_eof() {
        // An unclosed quote at EOF yields whatever was accumulated.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"abc"), rows(&[&["abc"]]));
    }

    #[test]
    fn disabled_quoting() {
        let mut dfa = Config::default();
        dfa.quote = None;
        assert_eq!(parse(&dfa, b"\"a\",b\n"), rows(&[&["\"a\"", "b"]]));
    }

    #[test]
    fn disabled_double_quote_escape() {
        let mut dfa = Config::default();
        dfa.double_quote_escape = false;
        assert_eq!(parse(&dfa, b"\"a\"\"b\",b\n"), rows(&[&["ab", "b"]]));
    }

    // ---- escapes ----------------------------------------------------------

    #[test]
    fn escaped_delimiter_unquoted() {
        // Backslash escapes a delimiter in an unquoted field.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a\\,b,c\n"), rows(&[&["a,b", "c"]]));
    }

    #[test]
    fn escaped_quote_in_quoted_field() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"\"a\\\"b\",c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn escaped_backslash() {
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a\\\\b\n"), rows(&[&["a\\b"]]));
    }

    #[test]
    fn disabled_escape_keeps_backslash() {
        let mut dfa = Config::default();
        dfa.escape = None;
        assert_eq!(parse(&dfa, b"a\\b\n"), rows(&[&["a\\b"]]));
    }

    // ---- comments ---------------------------------------------------------

    #[test]
    fn comment_lines_are_skipped() {
        let dfa = Config::default();
        assert_eq!(
            parse(&dfa, b"#comment\na,b\n#another\nc,d\n"),
            rows(&[&["a", "b"], &["c", "d"]]),
        );
    }

    #[test]
    fn hash_is_literal_mid_field() {
        // '#' only starts a comment at the beginning of a record.
        let dfa = Config::default();
        assert_eq!(parse(&dfa, b"a#b,c\n"), rows(&[&["a#b", "c"]]));
    }

    // ---- strict-mode literal quotes in unquoted fields --------------------

    #[test]
    fn strict_true_allows_literal_quote_in_unquoted_field() {
        // Strict(true): a quote that isn't at field start is an ordinary byte.
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        assert_eq!(parse(&dfa, b"a\"b,c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn strict_true_still_treats_leading_quote_as_quoting() {
        // The flag only affects mid-field quotes; a field that opens with a
        // quote is still a quoted field.
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        assert_eq!(parse(&dfa, b"\"a,b\",c\n"), rows(&[&["a,b", "c"]]));
    }

    #[test]
    fn strict_true_still_rejects_garbage_after_close() {
        // The flag does not relax the post-close rule: after a *closing* quote
        // only a delimiter or terminator is allowed.
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        let err = try_parse(&dfa, b"\"ab\"c\n").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    // ---- escape_in_unquoted_fields ---------------------------------------

    #[test]
    fn escape_disabled_in_unquoted_fields() {
        // With escapes off in unquoted fields, a backslash there is literal...
        let mut dfa = Config::default();
        dfa.escape_in_unquoted_fields = false;
        assert_eq!(parse(&dfa, b"a\\b\n"), rows(&[&["a\\b"]]));
        // ...but escapes still apply inside quoted fields.
        let mut dfa = Config::default();
        dfa.escape_in_unquoted_fields = false;
        assert_eq!(parse(&dfa, b"\"a\\\"b\"\n"), rows(&[&["a\"b"]]));
    }

    // ---- error position reporting ----------------------------------------

    #[test]
    fn parser_error_reports_record_and_field() {
        // Error in the 2nd record (index 1), 2nd field (index 1).
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&dfa, b"a,b,c\nd,e\"f\n").unwrap_err();
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
        let mut dfa = Config::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&dfa, b"a,b,c\nd,e\"f\n").unwrap_err();
        match err {
            // The offending quote is at absolute byte 9, not 3.
            Error::Parser(pos) => assert_eq!(pos.byte, 9),
            other => panic!("expected parser error, got {other:?}"),
        }
    }

    // ---- custom configuration --------------------------------------------

    #[test]
    fn custom_delimiter_and_quote() {
        let mut dfa = Config::default();
        dfa.delimiter = b'\t';
        dfa.quote = Some(b'\'');
        assert_eq!(
            parse(&dfa, b"'a\tb'\tc\n"),
            rows(&[&["a\tb", "c"]]),
        );
    }
}
