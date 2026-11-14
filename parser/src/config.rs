/// Dialect settings shared by every [`Dfa`](crate::dfa::Dfa).
///
/// [`Config::new`] is the default dialect. Any other is built with [`Config::builder`], which
/// checks the settings, so a `Config` is always valid:
///
/// ```
/// use csv_parser::config::{Config, QuoteStyle};
///
/// let config = Config::builder()
///     .delimiter(b';')
///     .quote(Some(b'\''))
///     .quote_style(QuoteStyle::Strict(false))
///     .comment(None)
///     .build()
///     .unwrap();
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// A structural byte is not ASCII.
    NotAscii { setting: &'static str, byte: u8 },
    /// Two settings use the same byte.
    Conflict { first: &'static str, second: &'static str, byte: u8 },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAscii { setting, byte } => {
                write!(f, "{setting} must be an ASCII byte, got 0x{byte:02x}")
            }
            Self::Conflict { first, second, byte } => {
                write!(f, "{first} and {second} are both '{}'", byte.escape_ascii())
            }
        }
    }
}

impl std::error::Error for ConfigError {}

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
        }
    }
}

impl Config {
    /// The default dialect.
    pub fn new() -> Self {
        Self::default()
    }

    /// A builder starting from the default dialect.
    pub fn builder() -> ConfigBuilder {
        ConfigBuilder { config: Self::default() }
    }
}

/// Collects the settings of a [`Config`] and checks them in [`ConfigBuilder::build`].
#[derive(Debug, Clone)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// The byte separating fields within a record.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.config.delimiter = delimiter;
        self
    }

    /// What ends a record: either `\r`/`\n` in any combination, or one byte.
    pub fn terminator(mut self, terminator: RecordTerminator) -> Self {
        self.config.terminator = terminator;
        self
    }

    /// The quote byte, or `None` to treat quotes as ordinary field bytes.
    pub fn quote(mut self, quote: Option<u8>) -> Self {
        self.config.quote = quote;
        self
    }

    /// Where a quote is allowed to open a quoted region.
    pub fn quote_style(mut self, quote_style: QuoteStyle) -> Self {
        self.config.quote_style = quote_style;
        self
    }

    /// The escape byte, or `None` to treat it as an ordinary field byte.
    pub fn escape(mut self, escape: Option<u8>) -> Self {
        self.config.escape = escape;
        self
    }

    /// Whether [`ConfigBuilder::escape`] also applies outside quoted fields.
    pub fn escape_in_unquoted_fields(mut self, enabled: bool) -> Self {
        self.config.escape_in_unquoted_fields = enabled;
        self
    }

    /// Whether a doubled quote inside a quoted field is a literal quote.
    pub fn double_quote_escape(mut self, enabled: bool) -> Self {
        self.config.double_quote_escape = enabled;
        self
    }

    /// The byte that starts a comment line, or `None` to disable comments.
    pub fn comment(mut self, comment: Option<u8>) -> Self {
        self.config.comment = comment;
        self
    }

    /// Build and validate the parser configuration
    pub fn build(self) -> Result<Config, ConfigError> {
        let config = self.config;
        let (terminator, second_terminator) = match config.terminator {
            RecordTerminator::CRLF => (b'\r', Some(b'\n')),
            RecordTerminator::Byte(b) => (b, None),
        };
        let bytes = [
            ("delimiter", Some(config.delimiter)),
            ("terminator", Some(terminator)),
            ("terminator", second_terminator),
            ("quote", config.quote),
            ("escape", config.escape),
            ("comment", config.comment),
        ];
        let bytes = bytes.iter().filter_map(|(setting, byte)| byte.map(|b| (*setting, b)));

        for (i, (setting, byte)) in bytes.clone().enumerate() {
            if !byte.is_ascii() {
                return Err(ConfigError::NotAscii { setting, byte });
            }
            if let Some((first, _)) = bytes.clone().take(i).find(|(_, earlier)| *earlier == byte) {
                return Err(ConfigError::Conflict { first, second: setting, byte });
            }
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_default() {
        assert_eq!(Config::builder().build().map(|_| ()), Ok(()));
        assert_eq!(Config::builder().delimiter(0).quote(None).escape(None).comment(None).build().map(|_| ()), Ok(()));
    }

    #[test]
    fn rejects_non_ascii_bytes() {
        assert_eq!(
            Config::builder().delimiter(0xa7).build().map(|_| ()),
            Err(ConfigError::NotAscii { setting: "delimiter", byte: 0xa7 }),
        );
        assert_eq!(
            Config::builder().terminator(RecordTerminator::Byte(0x80)).build().map(|_| ()),
            Err(ConfigError::NotAscii { setting: "terminator", byte: 0x80 }),
        );
        assert!(Config::builder().quote(Some(0xff)).build().is_err());
        assert!(Config::builder().escape(Some(0xff)).build().is_err());
        assert!(Config::builder().comment(Some(0xff)).build().is_err());
    }

    #[test]
    fn rejects_conflicting_bytes() {
        assert_eq!(
            Config::builder().delimiter(b'"').build().map(|_| ()),
            Err(ConfigError::Conflict { first: "delimiter", second: "quote", byte: b'"' }),
        );
        assert_eq!(
            Config::builder().comment(Some(b'\n')).build().map(|_| ()),
            Err(ConfigError::Conflict { first: "terminator", second: "comment", byte: b'\n' }),
        );
        assert!(Config::builder().escape(Some(b'"')).build().is_err());
        assert!(Config::builder().terminator(RecordTerminator::Byte(b',')).build().is_err());
        // a disabled setting cannot conflict
        assert_eq!(Config::builder().delimiter(b'#').comment(None).build().map(|_| ()), Ok(()));
    }
}
