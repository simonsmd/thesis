/// Dialect settings shared by every [`Dfa`](crate::parser::Dfa).
///
/// Build one by chaining onto [`Config::new`]:
///
/// ```
/// use csv_parser::config::{Config, QuoteStyle};
///
/// let config = Config::new()
///     .delimiter(b';')
///     .quote(Some(b'\''))
///     .quote_style(QuoteStyle::Strict(false))
///     .comment(None);
/// ```
#[derive(Debug, Clone)]
pub struct Config {
    pub(crate) delimiter: u8,
    pub(crate) terminator: RecordTerminator,
    pub(crate) quote: Option<u8>,
    pub(crate) quote_style: QuoteStyle,
    pub(crate) escape: Option<u8>,
    pub(crate) escape_in_unquoted_fields: bool,
    pub(crate) double_quote_escape: bool,
    pub(crate) comment: Option<u8>,
    // pub ignore_emtpy_lines: bool,
}

#[derive(Debug, Clone)]
pub enum RecordTerminator {
    CRLF,
    Byte(u8),
}

impl RecordTerminator {
    pub fn equals(&self, value: u8) -> bool {
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

impl Config {
    /// The default dialect, as a starting point for the builder methods below.
    pub fn new() -> Self {
        Self::default()
    }

    /// The byte separating fields within a record.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.delimiter = delimiter;
        self
    }

    /// What ends a record: either `\r`/`\n` in any combination, or one byte.
    pub fn terminator(mut self, terminator: RecordTerminator) -> Self {
        self.terminator = terminator;
        self
    }

    /// The quote byte, or `None` to treat quotes as ordinary field bytes.
    pub fn quote(mut self, quote: Option<u8>) -> Self {
        self.quote = quote;
        self
    }

    /// Where a quote is allowed to open a quoted region.
    pub fn quote_style(mut self, quote_style: QuoteStyle) -> Self {
        self.quote_style = quote_style;
        self
    }

    /// The escape byte, or `None` to treat it as an ordinary field byte.
    pub fn escape(mut self, escape: Option<u8>) -> Self {
        self.escape = escape;
        self
    }

    /// Whether [`Config::escape`] also applies outside quoted fields.
    pub fn escape_in_unquoted_fields(mut self, enabled: bool) -> Self {
        self.escape_in_unquoted_fields = enabled;
        self
    }

    /// Whether a doubled quote inside a quoted field is a literal quote.
    pub fn double_quote_escape(mut self, enabled: bool) -> Self {
        self.double_quote_escape = enabled;
        self
    }

    /// The byte that starts a comment line, or `None` to disable comments.
    pub fn comment(mut self, comment: Option<u8>) -> Self {
        self.comment = comment;
        self
    }
}
