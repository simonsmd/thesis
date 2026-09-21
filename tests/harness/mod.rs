//! Shared plumbing for the conformance suite.
//!
//! Every case runs through every [`Dfa`] implementation. `ReferenceDfa` defines
//! the expected behaviour and the others must agree with it exactly, on the
//! records and on the error, if any.
#![allow(dead_code)]

use std::io::Read;

use csv_parser::{
    DEFAULT_CAPACITY,
    bitshift::BitshiftDfa,
    config::Config,
    parser::{Dfa, Error, Parser},
    reference::ReferenceDfa,
};

pub type Parsed = Result<Vec<Vec<String>>, Error>;

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
///
/// To bring a new [`Dfa`] into the suite, add one line to the array below.
pub fn parse_with_capacity(config: &Config, input: &[u8], capacity: usize) -> Parsed {
    let expected = drive(Parser::<_, ReferenceDfa>::with_capacity(
        input,
        config.clone(),
        capacity,
    ));

    for (name, actual) in [(
        "bitshift",
        drive(Parser::<_, BitshiftDfa>::with_capacity(
            input,
            config.clone(),
            capacity,
        )),
    )] {
        assert_eq!(
            format!("{expected:?}"),
            format!("{actual:?}"),
            "the {name} dfa disagrees with the reference dfa",
        );
    }

    expected
}

/// Read every record out of `parser`.
///
/// Stops at the first error, so this does not exercise what a [`Dfa`] does when
/// `read_record` is called again afterwards — the DFAs currently diverge there.
fn drive<R: Read, D: Dfa>(mut parser: Parser<R, D>) -> Parsed {
    let mut out = Vec::new();
    loop {
        let record = parser.read_record()?;
        let Some(record) = record else { break };
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
pub fn rows(expected: &[&[&str]]) -> Vec<Vec<String>> {
    expected
        .iter()
        .map(|r| r.iter().map(|s| s.to_string()).collect())
        .collect()
}
