use std::fmt::Display;
use std::io::BufRead;
use std::str::Utf8Error;

mod config;

#[derive(Debug)]
pub struct Dfa {
    state: State,
    position: Position,
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

impl Default for Dfa {
    fn default() -> Self {
        Self {
            state: State::StartRecord,
            position: Position::default(),
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

#[derive(Debug)]
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

#[derive(Debug, PartialEq, Eq)]
pub enum QuoteStyle {
    /// Quotes can only start at the beginning of a field, otherwise quotes are not handled
    /// The boolean flag describes if quotes are allowed in unquoted fields
    Strict(bool),
    /// A quote character toggles the quoting mode anywhere inside a field
    Toggle, // Concat, Always?
}

#[derive(Debug, PartialEq, Eq)]
pub enum State {
    StartRecord,
    StartField,
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

impl Dfa {
    fn transition(&self, state: &State, input: u8) -> (State, TransitionAction) {
        use State::*;
        use TransitionAction::*;
        match state {
            StartRecord => {
                if self.terminator.equals(input) {
                    (StartRecord, Discard)
                } else if self.comment == Some(input) {
                    (InComment, Discard)
                } else {
                    (StartField, Epsilon)
                }
            }
            StartField => {
                if self.quote == Some(input) {
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
            EndField => (StartField, Epsilon),
            EndRecord => (StartRecord, Epsilon),
            Error => (Error, Discard),
        }
    }

    pub fn read_record(
        &mut self,
        reader: &mut impl BufRead,
        record: &mut Record,
    ) -> Result<bool, Error> {
        record.clear();
        self.position.field = 0;
        let mut input_index = 0;
        let mut input = reader.fill_buf().map_err(Error::IO)?;
        loop {
            if input_index >= input.len() {
                let consumed = input.len();
                reader.consume(consumed);
                self.position.byte += consumed;
                input = reader.fill_buf().map_err(Error::IO)?;
                input_index = 0;
                if input.is_empty() {
                    if self.state == State::InQuotedField
                        && matches!(self.quote_style, QuoteStyle::Strict(_))
                    {
                        return Err(Error::Parser(self.position));
                    }

                    if record.buffer.is_empty() && record.field_ends.is_empty() {
                        return Ok(false);
                    }
                    record.field_ends.push(record.buffer.len());
                    break;
                }
            }

            let (new_state, action) = self.transition(&self.state, input[input_index]);
            self.state = new_state;

            if self.state == State::Error {
                reader.consume(input_index);
                self.position.byte += input_index;
                return Err(Error::Parser(self.position));
            }

            if action == TransitionAction::Output {
                record.buffer.push(input[input_index]);
            }
            if action != TransitionAction::Epsilon {
                input_index += 1;
            }

            if self.state == State::EndField || self.state == State::EndRecord {
                record.field_ends.push(record.buffer.len());
                self.position.field += 1;
            }

            if self.state == State::EndRecord {
                break;
            }
        }

        self.position.record += 1;
        self.position.byte += input_index;
        reader.consume(input_index);
        Ok(true)
    }

    pub fn read_record_zerocopy(&mut self, reader: &mut impl BufRead, record: &mut Record) -> Result<bool, Error> {
        Ok(true)
    }

}


#[derive(Debug, Default, Clone)]
pub struct Record {
    buffer: Vec<u8>,
    field_ends: Vec<usize>,
}

impl Record {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of fields in the record.
    pub fn len(&self) -> usize {
        self.field_ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.field_ends.is_empty()
    }

    /// The raw bytes of field `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        let end = *self.field_ends.get(index)?;
        let start = if index == 0 {
            0
        } else {
            self.field_ends[index - 1]
        };
        Some(&self.buffer[start..end])
    }

    /// Iterate over the fields as byte slices.
    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        // TODO: use manual loop to avoid bound checks in get
        (0..self.len()).map(move |i| self.get(i).unwrap())
    }

    pub fn get_str(&self, i: usize) -> Option<Result<&str, Utf8Error>> {
        self.get(i).map(std::str::from_utf8)
    }
    pub fn iter_str(&self) -> impl Iterator<Item = Result<&str, Utf8Error>> {
        self.iter().map(std::str::from_utf8)
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.field_ends.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    /// Parse `input` to completion, returning each record as a `Vec<String>`.
    /// Panics on any parser/IO error; use [`try_parse`] to inspect the error.
    fn parse(dfa: &mut Dfa, input: &[u8]) -> Vec<Vec<String>> {
        try_parse(dfa, input).unwrap()
    }

    /// Like [`parse`], but surfaces the first parser/IO error instead of
    /// panicking. Records parsed before the error are discarded.
    fn try_parse(dfa: &mut Dfa, input: &[u8]) -> Result<Vec<Vec<String>>, Error> {
        let mut reader = BufReader::new(input);
        let mut record = Record::new();
        let mut out = Vec::new();
        while dfa.read_record(&mut reader, &mut record)? {
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
        let mut dfa = Dfa::default();
        assert_eq!(
            parse(&mut dfa, b"a,b,c\nd,e,f\n"),
            rows(&[&["a", "b", "c"], &["d", "e", "f"]]),
        );
    }

    #[test]
    fn no_trailing_newline() {
        // A final record without a terminator is still emitted.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a,b,c"), rows(&[&["a", "b", "c"]]));
    }

    #[test]
    fn empty_fields() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a,,c\n"), rows(&[&["a", "", "c"]]));
        assert_eq!(parse(&mut dfa, b",\n"), rows(&[&["", ""]]));
    }

    #[test]
    fn trailing_empty_field() {
        // A trailing delimiter before a terminator yields an empty last field.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a,\n"), rows(&[&["a", ""]]));
    }

    #[test]
    fn trailing_empty_field_at_eof() {
        // A trailing delimiter with no terminator still yields the empty field.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a,"), rows(&[&["a", ""]]));
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a,b,"), rows(&[&["a", "b", ""]]));
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b","), rows(&[&["", ""]]));
    }

    #[test]
    fn whitespace_is_significant() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b" a , b \n"), rows(&[&[" a ", " b "]]));
    }

    // ---- terminators ------------------------------------------------------

    #[test]
    fn crlf_terminator() {
        let mut dfa = Dfa::default();
        assert_eq!(
            parse(&mut dfa, b"a,b\r\nc,d\r\n"),
            rows(&[&["a", "b"], &["c", "d"]]),
        );
    }

    #[test]
    fn bare_cr_terminator() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a\rb\r"), rows(&[&["a"], &["b"]]));
    }

    #[test]
    fn custom_byte_terminator_keeps_cr() {
        // With a plain '\n' terminator, '\r' is an ordinary field byte.
        let mut dfa = Dfa::default();
        dfa.terminator = RecordTerminator::Byte(b'\n');
        assert_eq!(parse(&mut dfa, b"a\r\nb\n"), rows(&[&["a\r"], &["b"]]));
    }

    #[test]
    fn blank_lines_are_skipped() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a\n\r\r\n\n\rb\n"), rows(&[&["a"], &["b"]]));
    }

    // ---- quoting ----------------------------------------------------------

    #[test]
    fn quoted_fields() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"a\",\"b\"\n"), rows(&[&["a", "b"]]));
    }

    #[test]
    fn quoted_delimiter_and_newline() {
        // Delimiters and newlines inside quotes are literal.
        let mut dfa = Dfa::default();
        assert_eq!(
            parse(&mut dfa, b"\"a,b\",\"c\nd\"\n"),
            rows(&[&["a,b", "c\nd"]]),
        );
    }

    #[test]
    fn quoted_empty_field_vs_blank_line() {
        // "" is one empty field; a blank line is no record at all.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"\"\n"), rows(&[&[""]]));
    }

    #[test]
    fn doubled_quote_escape() {
        // "a""b" -> a"b  (RFC 4180 doubling)
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"a\"\"b\",c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn concat_quoted_then_unquoted() {
        // Concat mode joins a quoted region with adjacent unquoted text.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"a\"b,c\n"), rows(&[&["ab", "c"]]));
    }

    #[test]
    fn concat_unquoted_then_quoted() {
        // ...and a quote can open a quoted region mid-field.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a\"b\",c\n"), rows(&[&["ab", "c"]]));
    }
    
    #[test]
    fn toggle_with_double_quote_escape() {
        // handle a double quote as a literal quote in toggle mode
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"a\"\"b,\",c,\"a\"\"\"b,d\n"), rows(&[&["a\"b,", "c", "a\"b", "d"]]));
    }

    #[test]
    fn strict_mode_rejects_inner_quote() {
        // In Strict mode a quote that isn't at field start is a parser error,
        // reported at the offending byte.
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&mut dfa, b"a\"b\",c\n").unwrap_err();
        match err {
            Error::Parser(pos) => assert_eq!(pos.byte, 1),
            other => panic!("expected parser error, got {other:?}"),
        }
    }

    #[test]
    fn strict_mode_rejects_trailing_garbage_after_quote() {
        // After a closing quote, only a delimiter or terminator is allowed.
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&mut dfa, b"\"ab\"c\n").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    #[test]
    fn strict_mode_rejects_unclosed_quote_at_eof() {
        // An unclosed quote at EOF is a parser error
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&mut dfa, b"\"ab").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    #[test]
    fn unterminated_quote_at_eof() {
        // An unclosed quote at EOF yields whatever was accumulated.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"abc"), rows(&[&["abc"]]));
    }

    #[test]
    fn disabled_quoting() {
        let mut dfa = Dfa::default();
        dfa.quote = None;
        assert_eq!(parse(&mut dfa, b"\"a\",b\n"), rows(&[&["\"a\"", "b"]]));
    }

    #[test]
    fn disabled_double_quote_escape() {
        let mut dfa = Dfa::default();
        dfa.double_quote_escape = false;
        assert_eq!(parse(&mut dfa, b"\"a\"\"b\",b\n"), rows(&[&["ab", "b"]]));
    }

    // ---- escapes ----------------------------------------------------------

    #[test]
    fn escaped_delimiter_unquoted() {
        // Backslash escapes a delimiter in an unquoted field.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a\\,b,c\n"), rows(&[&["a,b", "c"]]));
    }

    #[test]
    fn escaped_quote_in_quoted_field() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"\"a\\\"b\",c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn escaped_backslash() {
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a\\\\b\n"), rows(&[&["a\\b"]]));
    }

    #[test]
    fn disabled_escape_keeps_backslash() {
        let mut dfa = Dfa::default();
        dfa.escape = None;
        assert_eq!(parse(&mut dfa, b"a\\b\n"), rows(&[&["a\\b"]]));
    }

    // ---- comments ---------------------------------------------------------

    #[test]
    fn comment_lines_are_skipped() {
        let mut dfa = Dfa::default();
        assert_eq!(
            parse(&mut dfa, b"#comment\na,b\n#another\nc,d\n"),
            rows(&[&["a", "b"], &["c", "d"]]),
        );
    }

    #[test]
    fn hash_is_literal_mid_field() {
        // '#' only starts a comment at the beginning of a record.
        let mut dfa = Dfa::default();
        assert_eq!(parse(&mut dfa, b"a#b,c\n"), rows(&[&["a#b", "c"]]));
    }

    // ---- strict-mode literal quotes in unquoted fields --------------------

    #[test]
    fn strict_true_allows_literal_quote_in_unquoted_field() {
        // Strict(true): a quote that isn't at field start is an ordinary byte.
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        assert_eq!(parse(&mut dfa, b"a\"b,c\n"), rows(&[&["a\"b", "c"]]));
    }

    #[test]
    fn strict_true_still_treats_leading_quote_as_quoting() {
        // The flag only affects mid-field quotes; a field that opens with a
        // quote is still a quoted field.
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        assert_eq!(parse(&mut dfa, b"\"a,b\",c\n"), rows(&[&["a,b", "c"]]));
    }

    #[test]
    fn strict_true_still_rejects_garbage_after_close() {
        // The flag does not relax the post-close rule: after a *closing* quote
        // only a delimiter or terminator is allowed.
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(true);
        let err = try_parse(&mut dfa, b"\"ab\"c\n").unwrap_err();
        assert!(matches!(err, Error::Parser(_)));
    }

    // ---- escape_in_unquoted_fields ---------------------------------------

    #[test]
    fn escape_disabled_in_unquoted_fields() {
        // With escapes off in unquoted fields, a backslash there is literal...
        let mut dfa = Dfa::default();
        dfa.escape_in_unquoted_fields = false;
        assert_eq!(parse(&mut dfa, b"a\\b\n"), rows(&[&["a\\b"]]));
        // ...but escapes still apply inside quoted fields.
        let mut dfa = Dfa::default();
        dfa.escape_in_unquoted_fields = false;
        assert_eq!(parse(&mut dfa, b"\"a\\\"b\"\n"), rows(&[&["a\"b"]]));
    }

    // ---- error position reporting ----------------------------------------

    #[test]
    fn parser_error_reports_record_and_field() {
        // Error in the 2nd record (index 1), 2nd field (index 1).
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&mut dfa, b"a,b,c\nd,e\"f\n").unwrap_err();
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
        let mut dfa = Dfa::default();
        dfa.quote_style = QuoteStyle::Strict(false);
        let err = try_parse(&mut dfa, b"a,b,c\nd,e\"f\n").unwrap_err();
        match err {
            // The offending quote is at absolute byte 9, not 3.
            Error::Parser(pos) => assert_eq!(pos.byte, 9),
            other => panic!("expected parser error, got {other:?}"),
        }
    }

    // ---- custom configuration --------------------------------------------

    #[test]
    fn custom_delimiter_and_quote() {
        let mut dfa = Dfa::default();
        dfa.delimiter = b'\t';
        dfa.quote = Some(b'\'');
        assert_eq!(
            parse(&mut dfa, b"'a\tb'\tc\n"),
            rows(&[&["a\tb", "c"]]),
        );
    }
}
