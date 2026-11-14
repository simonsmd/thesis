//! Dialects as the dataset files describe them, and their translation into a parser [`Config`].
//!
//! The field names and value formats follow `csvscrape`'s manifest: one key per `Config` field,
//! with `quote_style` and `quotes_in_unquoted_fields` together spelling out `QuoteStyle`.

use std::fmt;

use csv_parser::config::{self, Config, RecordTerminator};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

/// A structural byte, written as a one-character string in the dataset files.
///
/// Only ASCII is accepted: the DFAs match structural characters byte by byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Byte(pub u8);

impl Byte {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.as_bytes() {
            [b] if b.is_ascii() => Ok(Byte(*b)),
            _ => Err(format!("expected one ASCII character, got {s:?}")),
        }
    }
}

impl fmt::Display for Byte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.escape_ascii())
    }
}

impl<'de> Deserialize<'de> for Byte {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Byte::parse(&String::deserialize(d)?).map_err(D::Error::custom)
    }
}

impl Serialize for Byte {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&char::from(self.0).to_string())
    }
}

/// What ends a record: `"crlf"` (any of `\r`, `\n`) or a single byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Terminator {
    Crlf,
    Byte(Byte),
}

impl Terminator {
    pub fn parse(s: &str) -> Result<Self, String> {
        if s == "crlf" {
            Ok(Terminator::Crlf)
        } else {
            Byte::parse(s).map(Terminator::Byte)
        }
    }
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Crlf => f.write_str("crlf"),
            Terminator::Byte(b) => b.fmt(f),
        }
    }
}

impl<'de> Deserialize<'de> for Terminator {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Terminator::parse(&String::deserialize(d)?).map_err(D::Error::custom)
    }
}

impl Serialize for Terminator {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Terminator::Crlf => s.serialize_str("crlf"),
            Terminator::Byte(b) => b.serialize(s),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum QuoteStyle {
    Strict,
    Toggle,
}

/// A complete dialect.
///
/// In TOML a missing optional key means "none" (TOML has no null); in JSON it may also be `null`.
/// The booleans are required so a truncated entry fails to load instead of guessing.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dialect {
    pub delimiter: Byte,
    pub terminator: Terminator,
    pub quote: Option<Byte>,
    pub quote_style: Option<QuoteStyle>,
    pub quotes_in_unquoted_fields: Option<bool>,
    pub escape: Option<Byte>,
    pub escape_in_unquoted_fields: bool,
    pub double_quote_escape: bool,
    pub comment: Option<Byte>,
}

impl Default for Dialect {
    /// RFC 4180: comma, CRLF, doubled double quotes, nothing else.
    fn default() -> Self {
        Self {
            delimiter: Byte(b','),
            terminator: Terminator::Crlf,
            quote: Some(Byte(b'"')),
            quote_style: Some(QuoteStyle::Strict),
            quotes_in_unquoted_fields: Some(false),
            escape: None,
            escape_in_unquoted_fields: false,
            double_quote_escape: true,
            comment: None,
        }
    }
}

impl Dialect {
    /// The parser's quote style, or why the dialect does not determine one.
    ///
    /// Without a quote character the style never comes into play, so any value is accepted.
    pub fn parser_quote_style(&self) -> Result<config::QuoteStyle, String> {
        if self.quote.is_none() {
            return Ok(config::QuoteStyle::Strict(false));
        }
        match (self.quote_style, self.quotes_in_unquoted_fields) {
            (Some(QuoteStyle::Toggle), _) => Ok(config::QuoteStyle::Toggle),
            (Some(QuoteStyle::Strict), Some(allowed)) => Ok(config::QuoteStyle::Strict(allowed)),
            (Some(QuoteStyle::Strict), None) => {
                Err("strict quote_style without quotes_in_unquoted_fields".into())
            }
            (None, _) => Err("quote without quote_style".into()),
        }
    }

    /// Check that the dialect fully determines a [`Config`] and that the parser accepts it.
    pub fn validate(&self) -> Result<(), String> {
        self.try_config().map(|_| ())
    }

    /// The parser config for this dialect.
    ///
    /// Panics if the dialect is incomplete or invalid; [`Dialect::validate`] it first.
    pub fn config(&self) -> Config {
        self.try_config().expect("dialect was validated")
    }

    /// Every field is set explicitly: falling back to [`Config::default`] would silently enable
    /// its `#` comments and `\` escapes.
    fn try_config(&self) -> Result<Config, String> {
        Config::builder()
            .delimiter(self.delimiter.0)
            .terminator(match self.terminator {
                Terminator::Crlf => RecordTerminator::CRLF,
                Terminator::Byte(b) => RecordTerminator::Byte(b.0),
            })
            .quote(self.quote.map(|b| b.0))
            .quote_style(self.parser_quote_style()?)
            .escape(self.escape.map(|b| b.0))
            .escape_in_unquoted_fields(self.escape_in_unquoted_fields)
            .double_quote_escape(self.double_quote_escape)
            .comment(self.comment.map(|b| b.0))
            .build()
            .map_err(|e| e.to_string())
    }
}

impl fmt::Display for Dialect {
    /// A compact one-line form for progress output.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn opt(b: Option<Byte>) -> String {
            b.map_or("-".into(), |b| b.to_string())
        }
        write!(
            f,
            "delim={} term={} quote={}",
            self.delimiter,
            self.terminator,
            opt(self.quote)
        )?;
        if self.quote.is_some() {
            match (self.quote_style, self.quotes_in_unquoted_fields) {
                (Some(QuoteStyle::Toggle), _) => write!(f, " toggle")?,
                (Some(QuoteStyle::Strict), Some(true)) => write!(f, " strict(bare quotes)")?,
                (Some(QuoteStyle::Strict), _) => write!(f, " strict")?,
                (None, _) => write!(f, " ?style")?,
            }
            if self.double_quote_escape {
                write!(f, " \"\"")?;
            }
        }
        if let Some(escape) = self.escape {
            write!(f, " esc={escape}")?;
            if self.escape_in_unquoted_fields {
                write!(f, "(everywhere)")?;
            }
        }
        if let Some(comment) = self.comment {
            write!(f, " comment={comment}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dialect_without_optional_bytes_disables_them() {
        let dialect: Dialect = toml::from_str(
            r#"
            delimiter = ";"
            terminator = "\n"
            escape_in_unquoted_fields = false
            double_quote_escape = false
            "#,
        )
        .unwrap();
        assert_eq!(dialect.quote, None);
        assert_eq!(dialect.escape, None);
        assert_eq!(dialect.comment, None);

        // `Config::default` has `#` comments and `\` escapes; none of it may leak through.
        let config = format!("{:?}", dialect.config());
        assert!(config.contains("quote: None"), "{config}");
        assert!(config.contains("escape: None"), "{config}");
        assert!(config.contains("comment: None"), "{config}");
        assert!(config.contains("delimiter: 59"), "{config}");
    }

    #[test]
    fn json_nulls_are_none() {
        let dialect: Dialect = serde_json::from_str(
            r#"{"delimiter": "\t", "terminator": "crlf", "quote": null, "quote_style": null,
                "quotes_in_unquoted_fields": null, "escape": null,
                "escape_in_unquoted_fields": false, "double_quote_escape": false, "comment": null}"#,
        )
        .unwrap();
        assert_eq!(dialect.delimiter, Byte(b'\t'));
        assert_eq!(dialect.quote, None);
        dialect.validate().unwrap();
    }

    #[test]
    fn rejects_bad_entries() {
        let missing_bool = toml::from_str::<Dialect>("delimiter = \",\"\nterminator = \"crlf\"");
        assert!(missing_bool.is_err());

        let typo = toml::from_str::<Dialect>(
            "delimiter = \",\"\nterminator = \"crlf\"\nescape_in_unquoted_fields = false\n\
             double_quote_escape = false\nqoute = \"'\"",
        );
        assert!(typo.unwrap_err().to_string().contains("qoute"));

        assert!(Byte::parse(";;").is_err());
        assert!(Byte::parse("§").is_err());
        assert!(Byte::parse("").is_err());
    }

    #[test]
    fn quote_needs_a_style() {
        let dialect = Dialect {
            quote_style: None,
            ..Dialect::default()
        };
        assert!(dialect.validate().is_err());
        let dialect = Dialect {
            quote: None,
            quote_style: None,
            ..Dialect::default()
        };
        assert!(dialect.validate().is_ok());
    }

    #[test]
    fn structural_bytes_must_differ() {
        let dialect = Dialect {
            comment: Some(Byte(b',')),
            ..Dialect::default()
        };
        assert_eq!(
            dialect.validate().unwrap_err(),
            "delimiter and comment are both ','"
        );
    }
}
