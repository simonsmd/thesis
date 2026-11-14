//! Arbitrary bytes to test cases, shared by the fuzz targets and the property tests.
//!
//! Most bytes are mapped onto a small alphabet of structural characters so that inputs are dense
//! in quotes, escapes and terminators, and every dialect draws its structural bytes from disjoint
//! sets so they never collide.

use crate::{
    check,
    dialect::{Byte, Dialect, QuoteStyle, Terminator},
    parsers::{Options, Parser},
};

/// Bytes the data is mostly made of: every structural candidate plus a few ordinary ones.
const ALPHABET: &[u8] = b",;\t|\"'\\#\r\nab ";

pub struct Case {
    pub dialect: Dialect,
    pub options: Options,
    pub data: Vec<u8>,
}

impl Case {
    /// Two bytes of dialect, two of buffer capacity, one of field capacity, then the data.
    pub fn from_bytes(bytes: &[u8]) -> Option<Case> {
        let (&[d0, d1, c0, c1, f], raw) = bytes.split_first_chunk::<5>()?;
        let dialect = Dialect {
            delimiter: Byte(b",;\t|"[d0 as usize & 3]),
            terminator: match (d0 >> 2) & 3 {
                0 | 1 => Terminator::Crlf,
                2 => Terminator::Byte(Byte(b'\n')),
                _ => Terminator::Byte(Byte(b'\r')),
            },
            quote: [None, Some(Byte(b'"')), Some(Byte(b'"')), Some(Byte(b'\''))][(d0 as usize >> 4) & 3],
            quote_style: Some(if d0 & 0x40 == 0 { QuoteStyle::Strict } else { QuoteStyle::Toggle }),
            quotes_in_unquoted_fields: Some(d0 & 0x80 != 0),
            escape: (d1 & 1 != 0).then_some(Byte(b'\\')),
            escape_in_unquoted_fields: d1 & 2 != 0,
            double_quote_escape: d1 & 4 != 0,
            comment: (d1 & 8 != 0).then_some(Byte(b'#')),
        };
        let options = Options {
            // small read sizes so that refills land everywhere, including inside SIMD blocks
            capacity: 1 + (u16::from_le_bytes([c0, c1]) as usize % 512),
            fields: (f != 0).then_some(f as usize % 16 + 1),
        };
        let data = raw
            .iter()
            .map(|&b| if b < 0x80 { ALPHABET[b as usize % ALPHABET.len()] } else { b })
            .collect();
        Some(Case { dialect, options, data })
    }

    /// Check `parsers` against the reference: ours must agree exactly. rust-csv is compared only
    /// where it is expected to agree:
    ///
    /// - the reference parses without an error, since rust-csv never reports one
    /// - the dialect has no comments: a comment line ends at the record terminator in the
    ///   reference but always at `\n` in rust-csv, and an unterminated comment line at the end of the input
    ///   is an extra empty record in rust-csv
    pub fn check(&self, reference: &Parser, parsers: &[Parser]) -> Result<(), String> {
        let report = check::check(reference, parsers, &self.dialect, &self.data, &[self.options]);
        let failures: Vec<String> = report
            .unexpected_mismatches()
            .filter(|o| o.ours || self.dialect.comment.is_none())
            .map(|o| match &o.verdict {
                check::Verdict::Mismatch { difference, .. } => format!("{} ({:?}): {difference}", o.parser, o.options),
                _ => unreachable!(),
            })
            .collect();
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "dialect: {}\ninput: \"{}\"\n{}",
                self.dialect,
                self.data.escape_ascii(),
                failures.join("\n")
            ))
        }
    }
}
