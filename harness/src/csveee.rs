//! csveee's two chunk parsers as comparison parsers, `csveee/simd` and `csveee/dfa`.
//!
//! csveee parses chunks of a file speculatively on all cores and merges the results. Here it runs
//! single threaded through `parse_stream`: one pass over the input from the start, so there is no
//! chunking, no speculation on the quote state at chunk boundaries and no merge. That leaves the
//! chunk parser itself, which is what is being compared. Each entry pins one backend, so csveee
//! never falls back from the SIMD parser to the DFA.
//!
//! The SIMD parser needs a fixed field count of at most 63, known when the parse starts (the
//! callback's arity). It is found by parsing the input with one of this crate's parsers first, so
//! inputs that this crate's parsers stop on with an error are unsupported as well. It only knows
//! toggle quoting, where every quote toggles. Strict quoting without quotes in unquoted fields is
//! run as toggle quoting: on every input that strict quoting parses without an error, which the
//! field count requires, both produce the same records.
//!
//! The DFA parses every dialect csveee can express, so it runs in flexible mode, where records may
//! have any number of fields, and with csveee's quote handling for each quote style: strict quoting
//! is `Strict`, strict quoting with quotes in unquoted fields `Literal`, toggle quoting `Toggle`.
//! `Literal` behaves like rust-csv: data after a closing quote is appended to the field, where this
//! crate's parsers stop with an error, so the correctness check reports a difference there.
//!
//! What cannot be made fair through csveee's API:
//!
//! - Every parse builds csveee's chunk parser and its 64 KiB ring buffer, which is mapped twice for
//!   wrap-around, so a parse costs a few system calls more than for the other parsers, which are
//!   built once and reset. This only shows on small files.
//! - The input is copied into the ring buffer in reads of csveee's own size; [`Options`] are
//!   ignored.
//! - Errors carry a byte offset only, and records do not say where they end.

use csv_parser::{config::QuoteStyle, parser::CompactionParser, table::TableDfa};
use csveee::{
    Arity, Bytes, ParserBackend, ParserBuilder, QuoteHandling, RecordTerminator, Variadic,
};

use crate::{
    check::Digest,
    dialect::{Dialect, Terminator},
    parsers::{
        Consumer, ErrorAt, Errors, Options, OwnedRecord, Parser, Prepared, Records, Statistics,
    },
};

type Simd = csveee::Parser<Arity, Bytes>;
type Dfa = csveee::Parser<Variadic, Bytes>;

pub fn parsers() -> [Parser; 2] {
    [
        Parser {
            name: "csveee/simd",
            supports: supports_simd,
            errors: byte_errors,
            prepare: prepare_simd,
            records: records_simd,
        },
        Parser {
            name: "csveee/dfa",
            supports: supports_dfa,
            errors: dfa_errors,
            prepare: prepare_dfa,
            records: records_dfa,
        },
    ]
}

/// A single threaded csveee for the dialect, if csveee can express it. `strict_as_toggle` runs
/// strict quoting without quotes in unquoted fields as toggle quoting.
fn builder(
    d: &Dialect,
    strict_as_toggle: bool,
) -> Result<ParserBuilder<Arity, Bytes>, &'static str> {
    let quote_handling = match d.quote.map(|_| d.parser_quote_style()) {
        None => QuoteHandling::default(),
        Some(Ok(QuoteStyle::Toggle)) => QuoteHandling::Toggle,
        Some(Ok(QuoteStyle::Strict(false))) if strict_as_toggle => QuoteHandling::Toggle,
        Some(Ok(QuoteStyle::Strict(false))) => QuoteHandling::Strict,
        Some(Ok(QuoteStyle::Strict(true))) => QuoteHandling::Literal,
        Some(Err(_)) => return Err("no quote style"),
    };
    if d.escape_in_unquoted_fields {
        return Err("escapes in unquoted fields");
    }
    let quote = d.quote.map(|q| q.0);
    // csveee has one escape byte, and one equal to the quote is how it spells doubled quotes
    let escape = match (d.escape, d.double_quote_escape) {
        (Some(_), true) => return Err("both an escape character and doubled quotes"),
        (Some(e), false) => Some(e.0),
        (None, true) => quote,
        (None, false) => None,
    };

    Ok(ParserBuilder::new()
        .bytes()
        .delimiter(d.delimiter.0)
        .terminator(match d.terminator {
            Terminator::Crlf => RecordTerminator::CRLF,
            Terminator::Byte(b) => RecordTerminator::Byte(b.0),
        })
        .quote(quote)
        .quote_handling(quote_handling)
        .escape(escape)
        .comment(d.comment.map(|c| c.0))
        .has_headers(false)
        .concurrency(1))
}

fn byte_errors(_: &Dialect) -> Errors {
    Errors::Byte
}

/// `Literal` quoting appends data after a closing quote to the field, where the reference stops.
fn dfa_errors(d: &Dialect) -> Errors {
    match d.quote.map(|_| d.parser_quote_style()) {
        Some(Ok(QuoteStyle::Strict(true))) => Errors::Never,
        _ => Errors::Byte,
    }
}

fn invalid(error: csveee::Error) -> &'static str {
    match error {
        csveee::Error::InvalidConfig(reason) => reason,
        _ => "invalid configuration",
    }
}

fn build_simd(d: &Dialect) -> Result<Simd, &'static str> {
    let builder = builder(d, true)?.parser_backend(ParserBackend::Simd);
    builder.supports_backend(ParserBackend::Simd)?;
    builder.try_build().map_err(invalid)
}

fn build_dfa(d: &Dialect) -> Result<Dfa, &'static str> {
    builder(d, false)?
        .flexible()
        .parser_backend(ParserBackend::Dfa)
        .try_build()
        .map_err(invalid)
}

fn supports_simd(d: &Dialect, input: &[u8]) -> Result<(), &'static str> {
    build_simd(d)?;
    arity(d, input)?;
    Ok(())
}

fn supports_dfa(d: &Dialect, _: &[u8]) -> Result<(), &'static str> {
    build_dfa(d).map(drop)
}

/// The field count of every record, as this crate's parsers see them.
fn arity(d: &Dialect, input: &[u8]) -> Result<usize, &'static str> {
    let mut parser = CompactionParser::<_, TableDfa>::with_config(input, d.config());
    let mut arity = None;
    while let Some(record) = parser.read_record().map_err(|_| "parse error")? {
        match arity {
            None => arity = Some(record.len()),
            Some(a) if a != record.len() => return Err("records of different lengths"),
            Some(_) => {}
        }
    }
    match arity {
        Some(64..) => Err("more than 63 fields"),
        // an empty input parses the same at any arity
        arity => Ok(arity.unwrap_or(1)),
    }
}

/// Receives the records of a correctness parse. One type for all of them, so the SIMD parser is
/// instantiated twice per arity (with this and with [`Statistics`]) rather than once per consumer.
type Sink<'s> = dyn FnMut(&mut dyn ExactSizeIterator<Item = &[u8]>) + 's;

type Parse<P, C> = fn(&mut P, &[u8], &mut C) -> csveee::Result<()>;
type ParseSink<P> = fn(&mut P, &[u8], &mut Sink<'_>) -> csveee::Result<()>;

#[inline(always)]
fn parse_simd<C: Consumer, const N: usize>(
    parser: &mut Simd,
    input: &[u8],
    consumer: &mut C,
) -> csveee::Result<()> {
    parser.parse_stream(
        input,
        || consumer,
        |consumer, fields: [&mut [u8]; N]| {
            consumer.record(fields.iter().map(|f| &**f));
            Ok(())
        },
        |_| (),
    )
}

fn parse_simd_sink<const N: usize>(
    parser: &mut Simd,
    input: &[u8],
    sink: &mut Sink<'_>,
) -> csveee::Result<()> {
    parser.parse_stream(
        input,
        || sink,
        |sink, fields: [&mut [u8]; N]| {
            sink(&mut fields.iter().map(|f| &**f));
            Ok(())
        },
        |_| (),
    )
}

/// The parse functions for a field count: the arity is a const generic in csveee's API.
macro_rules! arities {
    ($($n:literal)*) => {
        fn simd_for(arity: usize) -> (Parse<Simd, Statistics>, ParseSink<Simd>) {
            match arity {
                $($n => (parse_simd::<Statistics, $n>, parse_simd_sink::<$n>),)*
                _ => unreachable!("arity {arity} is not supported"),
            }
        }
    };
}

arities!(
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
);

#[inline(always)]
fn parse_dfa<C: Consumer>(parser: &mut Dfa, input: &[u8], consumer: &mut C) -> csveee::Result<()> {
    parser.parse_stream(
        input,
        || consumer,
        |consumer, fields: &mut [&mut [u8]]| {
            consumer.record(fields.iter().map(|f| &**f));
            Ok(())
        },
        |_| (),
    )
}

fn parse_dfa_sink(parser: &mut Dfa, input: &[u8], sink: &mut Sink<'_>) -> csveee::Result<()> {
    parser.parse_stream(
        input,
        || sink,
        |sink, fields: &mut [&mut [u8]]| {
            sink(&mut fields.iter().map(|f| &**f));
            Ok(())
        },
        |_| (),
    )
}

fn error_at(error: csveee::Error) -> ErrorAt {
    match error.position() {
        Some(p) => ErrorAt {
            record: 0,
            field: 0,
            byte: p.byte_offset,
        },
        // the configuration was validated, the input is a slice and the callback accepts every record
        None => unreachable!("csveee error without a position: {error}"),
    }
}

fn unsupported<T>(reason: &str) -> T {
    panic!("csveee prepared for an unsupported input: {reason}")
}

/// A parser and its parse functions.
struct PreparedCsveee<P> {
    parser: P,
    statistics: Parse<P, Statistics>,
    sink: ParseSink<P>,
}

impl PreparedCsveee<Simd> {
    fn simd(d: &Dialect, input: &[u8]) -> Self {
        let parser = build_simd(d).unwrap_or_else(unsupported);
        let (statistics, sink) = simd_for(arity(d, input).unwrap_or_else(unsupported));
        Self {
            parser,
            statistics,
            sink,
        }
    }
}

impl PreparedCsveee<Dfa> {
    fn dfa(d: &Dialect) -> Self {
        let parser = build_dfa(d).unwrap_or_else(unsupported);
        Self {
            parser,
            statistics: parse_dfa::<Statistics>,
            sink: parse_dfa_sink,
        }
    }
}

impl<P> PreparedCsveee<P> {
    /// csveee hands records to a callback, so they are all collected first.
    fn records<'a>(mut self, input: &'a [u8]) -> Records<'a> {
        let mut records = Vec::new();
        let result = (self.sink)(&mut self.parser, input, &mut |fields| {
            // where a record ends is not known; the end of the input shows the rest of it in a difference
            records.push(Ok(OwnedRecord {
                fields: fields.map(<[u8]>::to_vec).collect(),
                end: input.len(),
            }));
        });
        records.extend(result.err().map(|e| Err(error_at(e))));
        Box::new(records.into_iter())
    }
}

impl<'a, P> Prepared<'a> for PreparedCsveee<P> {
    fn statistics(&mut self, input: &'a [u8]) -> Result<Statistics, ErrorAt> {
        let mut statistics = Statistics::default();
        (self.statistics)(&mut self.parser, input, &mut statistics).map_err(error_at)?;
        Ok(statistics)
    }

    fn digest(&mut self, input: &'a [u8]) -> (Digest, Option<ErrorAt>) {
        let mut digest = Digest::default();
        let result = (self.sink)(&mut self.parser, input, &mut |fields| digest.record(fields));
        (digest, result.err().map(error_at))
    }
}

fn prepare_simd<'a>(d: &Dialect, input: &'a [u8], _: &Options) -> Box<dyn Prepared<'a> + 'a> {
    Box::new(PreparedCsveee::simd(d, input))
}

fn records_simd<'a>(d: &Dialect, input: &'a [u8], _: &Options) -> Records<'a> {
    PreparedCsveee::simd(d, input).records(input)
}

fn prepare_dfa<'a>(d: &Dialect, _: &'a [u8], _: &Options) -> Box<dyn Prepared<'a> + 'a> {
    Box::new(PreparedCsveee::dfa(d))
}

fn records_dfa<'a>(d: &Dialect, input: &'a [u8], _: &Options) -> Records<'a> {
    PreparedCsveee::dfa(d).records(input)
}
