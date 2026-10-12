//! Correctness: every parser against [`REFERENCE`](crate::parsers::REFERENCE).
//!
//! A parse is reduced to a [`Fingerprint`]: a hash over the records plus where it stopped with an
//! error, if it did. This streams, so multi-gigabyte files need no memory for their records. Only
//! when two fingerprints differ are both parsers run again record by record to show the first
//! difference.
//!
//! Every parser parses the input twice, the second time after a reset, which is how benchmarks
//! reuse a parser; both parses have to agree with the reference.

use std::fmt::Write as _;

use serde::Serialize;
use xxhash_rust::xxh3::Xxh3;

use crate::{
    dialect::Dialect,
    parsers::{self, Consumer, ErrorAt, Options, OwnedRecord, Parser, Statistics},
};

/// Hashes records with their structure: field count, then each field's length and bytes.
#[derive(Default, Clone)]
pub struct Digest {
    hasher: Xxh3,
    pub statistics: Statistics,
}

impl Consumer for Digest {
    #[inline]
    fn record<'a>(&mut self, fields: impl ExactSizeIterator<Item = &'a [u8]>) {
        self.statistics.records += 1;
        self.statistics.fields += fields.len() as u64;
        self.hasher.update(&(fields.len() as u64).to_le_bytes());
        for field in fields {
            self.statistics.field_bytes += field.len() as u64;
            self.hasher.update(&(field.len() as u64).to_le_bytes());
            self.hasher.update(field);
        }
    }
}

/// Everything a parse produced, in a form that is cheap to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Fingerprint {
    #[serde(serialize_with = "hex")]
    pub hash: u128,
    pub statistics: Statistics,
    pub error: Option<ErrorAt>,
}

fn hex<S: serde::Serializer>(hash: &u128, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&format!("{hash:032x}"))
}

impl Fingerprint {
    fn new((digest, error): (Digest, Option<ErrorAt>)) -> Self {
        Fingerprint { hash: digest.hasher.digest128(), statistics: digest.statistics, error }
    }
}

/// Parse `input` twice with one parser: fresh from construction, and again after a reset.
pub fn fingerprints(parser: &Parser, dialect: &Dialect, input: &[u8], options: &Options) -> [Fingerprint; 2] {
    let mut prepared = (parser.prepare)(dialect, input, options);
    [Fingerprint::new(prepared.digest(input)), Fingerprint::new(prepared.digest(input))]
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "verdict", rename_all = "lowercase")]
pub enum Verdict {
    Agree,
    Unsupported { reason: &'static str },
    Mismatch { fingerprint: Fingerprint, difference: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub parser: &'static str,
    /// One of this crate's parsers rather than rust-csv.
    pub ours: bool,
    pub options: Options,
    #[serde(flatten)]
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    /// The reference parse, at default options.
    pub reference: Fingerprint,
    pub outcomes: Vec<Outcome>,
}

impl Report {
    pub fn mismatches(&self) -> impl Iterator<Item = &Outcome> {
        self.outcomes.iter().filter(|o| matches!(o.verdict, Verdict::Mismatch { .. }))
    }

    /// Mismatches, except rust-csv's where the reference stops with an error: rust-csv never
    /// reports one, so it is expected to differ there.
    pub fn unexpected_mismatches(&self) -> impl Iterator<Item = &Outcome> {
        self.mismatches().filter(|o| o.ours || self.reference.error.is_none())
    }
}

/// Run every parser with every set of options on `input` and compare each to the reference parser
/// at default options. The reference itself is checked at the other options.
pub fn check(
    reference: &Parser,
    parsers: &[Parser],
    dialect: &Dialect,
    input: &[u8],
    options: &[Options],
) -> Report {
    let input = parsers::without_bom(input);
    let default = Options::default();
    let [expected, _] = fingerprints(reference, dialect, input, &default);

    let mut outcomes = Vec::new();
    for parser in parsers {
        for options in options {
            if parser.name == reference.name && *options == default {
                continue;
            }
            let verdict = match (parser.supports)(dialect) {
                Err(reason) => Verdict::Unsupported { reason },
                Ok(()) => match fingerprints(parser, dialect, input, options) {
                    [fresh, reset] if fresh == expected && reset == expected => Verdict::Agree,
                    [fresh, _] if fresh != expected => {
                        let difference = first_difference(reference, parser, dialect, input, options);
                        Verdict::Mismatch { fingerprint: fresh, difference }
                    }
                    [_, reset] => Verdict::Mismatch {
                        fingerprint: reset,
                        difference: "differs after a reset".into(),
                    },
                },
            };
            outcomes.push(Outcome { parser: parser.name, ours: parser.is_ours(), options: *options, verdict });
        }
    }

    Report { reference: expected, outcomes }
}

/// Run both parsers in lockstep and describe the first record where they differ.
pub fn first_difference(reference: &Parser, parser: &Parser, dialect: &Dialect, input: &[u8], options: &Options) -> String {
    let mut expected = (reference.records)(dialect, input, &Options::default());
    let mut actual = (parser.records)(dialect, input, options);
    // where the current record starts in the input, according to the reference
    let mut start = 0;

    for index in 0.. {
        let (e, a) = (expected.next(), actual.next());
        if same(&e, &a) {
            match e {
                Some(Ok(record)) => {
                    start = record.end;
                    continue;
                }
                // both ended the same way, yet the fingerprints differ
                _ => return "fingerprints differ but the records are equal".into(),
            }
        }

        let mut out = format!("record {index} differs\n");
        // show the records from their first differing field on
        let skip = match (&e, &a) {
            (Some(Ok(e)), Some(Ok(a))) => e.fields.iter().zip(&a.fields).take_while(|(e, a)| e == a).count(),
            _ => 0,
        };
        let _ = writeln!(out, "  reference: {}", describe(&e, skip));
        let _ = writeln!(out, "  {}: {}", parser.name, describe(&a, skip));
        let end = match (&e, &a) {
            (Some(Ok(r)), _) => r.end,
            (_, Some(Ok(r))) => r.end,
            (Some(Err(err)), _) | (_, Some(Err(err))) => err.byte + 1,
            _ => start,
        };
        let _ = write!(out, "  input at byte {start}: {}", excerpt(input, start, end));
        return out;
    }
    unreachable!()
}

/// Equal records and errors; where a record ends is not compared, since parsers count it
/// differently.
fn same(e: &Option<Result<OwnedRecord, ErrorAt>>, a: &Option<Result<OwnedRecord, ErrorAt>>) -> bool {
    match (e, a) {
        (Some(Ok(e)), Some(Ok(a))) => e.fields == a.fields,
        _ => e == a,
    }
}

/// A record from field `skip` on, with at most a few fields shown.
fn describe(item: &Option<Result<OwnedRecord, ErrorAt>>, skip: usize) -> String {
    const MAX_FIELDS: usize = 8;
    match item {
        None => "end of input".into(),
        Some(Err(e)) => e.to_string(),
        Some(Ok(record)) => {
            let mut fields: Vec<String> = record
                .fields
                .iter()
                .skip(skip)
                .take(MAX_FIELDS)
                .map(|f| format!("\"{}\"", truncate(f, 60)))
                .collect();
            if skip > 0 {
                fields.insert(0, format!("{skip} equal fields ..."));
            }
            let shown = skip + MAX_FIELDS;
            if record.fields.len() > shown {
                fields.push(format!("... {} more", record.fields.len() - shown));
            }
            format!("{} fields [{}]", record.fields.len(), fields.join(", "))
        }
    }
}

fn truncate(bytes: &[u8], max: usize) -> String {
    if bytes.len() <= max {
        bytes.escape_ascii().to_string()
    } else {
        format!("{}...", bytes[..max].escape_ascii())
    }
}

fn excerpt(input: &[u8], start: usize, end: usize) -> String {
    let start = start.min(input.len());
    let end = end.clamp(start, input.len());
    format!("\"{}\"", truncate(&input[start..end], 300))
}
