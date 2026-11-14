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
    classifier::{Classifier, CmpOr, Native, NibbleMap, u8x16, u8x32, u8x64},
    config::Config,
    dfa::Dfa,
    parser::{
        CompactionParser, Error, PerState, Sets, Shared, ZeroCopyParser, compaction, zerocopy,
    },
    reference::ReferenceDfa,
    table::TableDfa,
};

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
use csv_parser::classifier::{Pcmpestr, Pcmpistr};

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
///
/// Every engine also runs with a field buffer of a single entry, which grows on
/// the first record with more than one field.
pub fn parse_with_capacity(config: &Config, input: &[u8], capacity: usize) -> Parsed {
    let mut defaults = engines(config, input, capacity, None).into_iter();
    let (reference_name, reference) = defaults.next().unwrap();
    let expected = drive(reference);

    let expected_text = format!("{expected:?}");
    let actual = defaults.map(|(name, e)| (name, format!("{:?}", drive(e))));
    disagreements(&reference_name, &expected_text, actual);

    let actual = engines(config, input, capacity, Some(1))
        .into_iter()
        .map(|(name, e)| {
            (
                format!("{name} with 1 field end"),
                format!("{:?}", drive(e)),
            )
        });
    disagreements(&reference_name, &expected_text, actual);

    expected
}

/// Parse `input` up to the first error, then call `read_record` once more, and
/// return both outcomes as text. Every engine must agree with the reference.
pub fn read_after_error(config: &Config, input: &[u8]) -> (String, String) {
    let mut engines = engines(config, input, DEFAULT_CAPACITY, None).into_iter();
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

/// The skipping `read_record` variants with one classifier and set, named
/// `compaction-<variant>-<name>`.
fn skipping<'a, D: Dfa + 'a, C: Classifier + 'a, M: Sets + 'a>(
    name: &str,
    config: &Config,
    input: &'a [u8],
    capacity: usize,
    fields: usize,
) -> [(String, Engine<'a>); 3] {
    let new = || CompactionParser::<_, D>::with_capacity(input, config.clone(), capacity, fields);
    let classifier = C::new(config);
    let (mut simd, mut check) = (new(), new());
    let mut zerocopy =
        ZeroCopyParser::<_, D>::with_capacity(input, config.clone(), capacity, fields);
    [
        (
            format!("compaction-simd-{name}"),
            Box::new(move || {
                simd.read_record_simd(&classifier, M::default())
                    .map(|r| r.map(|r| strings(r.iter())))
            }) as Engine,
        ),
        (
            format!("zerocopy-simd-{name}"),
            Box::new(move || {
                zerocopy
                    .read_record_simd(&classifier, M::default())
                    .map(|r| r.map(|r| strings(r.iter())))
            }),
        ),
        (
            format!("compaction-simd-check-{name}"),
            Box::new(move || {
                check
                    .read_record_simd_check(&classifier, M::default())
                    .map(|r| r.map(|r| strings(r.iter())))
            }),
        ),
    ]
}

/// [`skipping`] for every classifier and width, with either set.
macro_rules! skipping_for {
    ($dfa:ty, $name:literal, $config:expr, $input:expr, $capacity:expr, $fields:expr) => {{
        let mut engines = Vec::new();
        macro_rules! sets {
            ($skip:literal, $classifier:ty) => {
                engines.extend(skipping::<$dfa, $classifier, Shared>(
                    concat!($skip, "-shared/", $name),
                    $config,
                    $input,
                    $capacity,
                    $fields,
                ));
                engines.extend(skipping::<$dfa, $classifier, PerState>(
                    concat!($skip, "-state/", $name),
                    $config,
                    $input,
                    $capacity,
                    $fields,
                ));
            };
        }
        sets!("cmp-or-16", CmpOr<u8x16<Native>>);
        sets!("cmp-or-32", CmpOr<u8x32<Native>>);
        sets!("cmp-or-64", CmpOr<u8x64<Native>>);
        sets!("nibble-map-16", NibbleMap<u8x16<Native>>);
        sets!("nibble-map-32", NibbleMap<u8x32<Native>>);
        sets!("nibble-map-64", NibbleMap<u8x64<Native>>);
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ))]
        {
            sets!("pcmpistr-16", Pcmpistr<u8x16<Native>>);
            sets!("pcmpistr-32", Pcmpistr<u8x32<Native>>);
            sets!("pcmpistr-64", Pcmpistr<u8x64<Native>>);
            sets!("pcmpestr-16", Pcmpestr<u8x16<Native>>);
            sets!("pcmpestr-32", Pcmpestr<u8x32<Native>>);
            sets!("pcmpestr-64", Pcmpestr<u8x64<Native>>);
        }
        engines
    }};
}

/// The engines for one DFA: every parser and `read_record` variant, compaction
/// first so that it is the reference for the reference DFA. `$fields` is the
/// initial size of the field buffers, `None` for the parsers' defaults.
macro_rules! engines_for {
    ($dfa:ty, $name:literal, $config:expr, $input:expr, $capacity:expr, $fields:expr) => {{
        let fields = $fields.unwrap_or(compaction::DEFAULT_FIELDS);
        let mut compaction =
            CompactionParser::<_, $dfa>::with_capacity($input, $config.clone(), $capacity, fields);
        let mut zerocopy = ZeroCopyParser::<_, $dfa>::with_capacity(
            $input,
            $config.clone(),
            $capacity,
            $fields.unwrap_or(zerocopy::DEFAULT_FIELDS),
        );
        let mut branchless =
            CompactionParser::<_, $dfa>::with_capacity($input, $config.clone(), $capacity, fields);
        let mut engines = vec![
            (
                concat!("compaction/", $name).to_string(),
                Box::new(move || {
                    compaction
                        .read_record()
                        .map(|r| r.map(|r| strings(r.iter())))
                }) as Engine,
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
        ];
        engines.extend(skipping_for!(
            $dfa, $name, $config, $input, $capacity, fields
        ));
        engines
    }};
}

/// Every engine under test, the reference (`compaction/reference`) first.
///
/// To bring a new [`Dfa`] into the suite, add one line to the array below; a
/// new parser or `read_record` variant goes into [`engines_for`].
fn engines<'a>(
    config: &Config,
    input: &'a [u8],
    capacity: usize,
    fields: Option<usize>,
) -> Vec<(String, Engine<'a>)> {
    [
        engines_for!(ReferenceDfa, "reference", config, input, capacity, fields),
        engines_for!(BitshiftDfa, "bitshift", config, input, capacity, fields),
        engines_for!(TableDfa, "table", config, input, capacity, fields),
        engines_for!(ClassDfa, "class", config, input, capacity, fields),
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
