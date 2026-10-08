//! Shared plumbing for the conformance suite.
//!
//! Every case runs through every engine: each parser and `read_record` variant
//! with each [`Dfa`] implementation. `CompactionParser::read_record` with
//! `ReferenceDfa` defines the expected behaviour and the others must agree with
//! it exactly, on the records and on the error, if any.
#![allow(dead_code)]

use csv_parser::{
    DEFAULT_CAPACITY,
    bitshift::BitshiftDfa,
    class::ClassDfa,
    config::Config,
    parser::{CompactionParser, Error, ZeroCopyParser},
    reference::ReferenceDfa,
    table::TableDfa,
};

pub type Parsed = Result<Vec<Vec<String>>, Error>;

/// The outcome of one `read_record` call, with the record copied out.
type Next = Result<Option<Vec<String>>, Error>;

/// A parser that hands out its records one call at a time.
type Engine<'a> = Box<dyn FnMut() -> Next + 'a>;

/// Parse `input` to completion, returning each record as a `Vec<String>`.
/// Panics on any parser/IO error; use [`try_parse`] to inspect the error.
pub fn parse(config: &Config, input: &[u8]) -> Vec<Vec<String>> {
    try_parse(config, input).unwrap()
}

/// Like [`parse`], but surfaces the first parser/IO error instead of
/// panicking. Records parsed before the error are discarded.
pub fn try_parse(config: &Config, input: &[u8]) -> Parsed {
    parse_with_capacity(config, input, DEFAULT_CAPACITY)
}

/// [`try_parse`] with an explicit buffer size, to exercise refills in the
/// middle of a record.
pub fn parse_with_capacity(config: &Config, input: &[u8], capacity: usize) -> Parsed {
    let mut engines = engines(config, input, capacity).into_iter();
    let (reference_name, reference) = engines.next().unwrap();
    let expected = drive(reference);

    let expected_text = format!("{expected:?}");
    let actual = engines.map(|(name, e)| (name, format!("{:?}", drive(e))));
    disagreements(&reference_name, &expected_text, actual);

    expected
}

/// Parse `input` up to the first error, then call `read_record` once more, and
/// return both outcomes as text. Every engine must agree with the reference.
pub fn read_after_error(config: &Config, input: &[u8]) -> (String, String) {
    let mut engines = engines(config, input, DEFAULT_CAPACITY).into_iter();
    let (reference_name, reference) = engines.next().unwrap();
    let expected = after_error(reference);

    let expected_text = format!("{expected:?}");
    let actual = engines.map(|(name, e)| (name, format!("{:?}", after_error(e))));
    disagreements(&reference_name, &expected_text, actual);

    expected
}

/// Panic listing every engine whose outcome differs from the reference, not
/// just the first one.
fn disagreements(reference: &str, expected: &str, actual: impl Iterator<Item = (String, String)>) {
    let failed: Vec<String> = actual
        .filter(|(_, actual)| actual != expected)
        .map(|(name, actual)| format!("  {name}: {actual}"))
        .collect();
    assert!(
        failed.is_empty(),
        "disagreeing with {reference} ({expected}):\n{}",
        failed.join("\n"),
    );
}

/// The engines for one DFA: every parser and `read_record` variant, compaction
/// first so that it is the reference for the reference DFA.
macro_rules! engines_for {
    ($dfa:ty, $name:literal, $config:expr, $input:expr, $capacity:expr) => {{
        let mut compaction =
            CompactionParser::<_, $dfa>::with_capacity($input, $config.clone(), $capacity);
        let mut zerocopy =
            ZeroCopyParser::<_, $dfa>::with_capacity($input, $config.clone(), $capacity);
        let mut branchless =
            CompactionParser::<_, $dfa>::with_capacity($input, $config.clone(), $capacity);
        [
            (
                concat!("compaction/", $name).to_string(),
                Box::new(move || compaction.read_record().map(|r| r.map(|r| strings(r.iter()))))
                    as Engine,
            ),
            (
                concat!("zerocopy/", $name).to_string(),
                Box::new(move || zerocopy.read_record().map(|r| r.map(|r| strings(r.iter())))),
            ),
            (
                concat!("compaction-branchless/", $name).to_string(),
                Box::new(move || {
                    branchless
                        .read_record_branchless()
                        .map(|r| r.map(|r| strings(r.iter())))
                }),
            ),
        ]
    }};
}

/// Every engine under test, the reference (`compaction/reference`) first.
///
/// To bring a new [`Dfa`] into the suite, add one line to the array below; a
/// new parser or `read_record` variant goes into [`engines_for`].
fn engines<'a>(config: &Config, input: &'a [u8], capacity: usize) -> Vec<(String, Engine<'a>)> {
    [
        engines_for!(ReferenceDfa, "reference", config, input, capacity),
        engines_for!(BitshiftDfa, "bitshift", config, input, capacity),
        engines_for!(TableDfa, "table", config, input, capacity),
        engines_for!(ClassDfa, "class", config, input, capacity),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn strings<'f>(fields: impl Iterator<Item = &'f [u8]>) -> Vec<String> {
    fields
        .map(|f| String::from_utf8(f.to_vec()).unwrap())
        .collect()
}

fn after_error(mut engine: Engine<'_>) -> (String, String) {
    let first = loop {
        match engine() {
            Ok(Some(_)) => continue,
            other => break format!("{:?}", other.map(|r| r.map(|_| ()))),
        }
    };
    let second = format!("{:?}", engine().map(|r| r.map(|_| ())));
    (first, second)
}

/// Read every record out of `engine`.
///
/// Stops at the first error; [`read_after_error`] covers what happens when
/// `read_record` is called again afterwards.
fn drive(mut engine: Engine<'_>) -> Parsed {
    let mut out = Vec::new();
    while let Some(record) = engine()? {
        out.push(record);
    }
    Ok(out)
}

/// Build the expected `Vec<Vec<String>>` from string literals.
pub fn rows(expected: &[&[&str]]) -> Vec<Vec<String>> {
    expected
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}
