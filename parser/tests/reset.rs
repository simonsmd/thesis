//! `reset` must make a used parser behave like a new one, whatever state it was left in.

use csv_parser::{
    config::{Config, QuoteStyle},
    dfa::Dfa,
    parser::{CompactionParser, Error, ZeroCopyParser, compaction, zerocopy},
    reference::ReferenceDfa,
    skip::{NibbleMap, Native, Skip},
    table::TableDfa,
};

type Records = Result<Vec<Vec<String>>, String>;

fn strings<'f>(fields: impl Iterator<Item = &'f [u8]>) -> Vec<String> {
    fields.map(|f| String::from_utf8(f.to_vec()).unwrap()).collect()
}

/// Read every record, or up to `limit`. Errors are compared as text, positions included.
macro_rules! drain {
    ($parser:expr, $read:ident $(, $skip:expr)?; $limit:expr) => {{
        let parser = &mut $parser;
        let mut out = Vec::new();
        loop {
            if out.len() == $limit {
                break Ok(out);
            }
            match parser.$read($($skip)?) {
                Ok(Some(record)) => out.push(strings(record.iter())),
                Ok(None) => break Ok(out),
                Err(Error::Parser(p)) => break Err(format!("error at {p}")),
                Err(Error::IO(e)) => panic!("{e}"),
            }
        }
    }};
}

/// The inputs a parser is left in a state by before the reset, and what it parses afterwards.
const BEFORE: &[&[u8]] = &[
    // stops with an error (a quote in an unquoted field)
    b"a,b\nc\"d,e\n",
    // a record longer than the buffer, which grows it
    b"0123456789012345678901234567890123456789012345678901234567890123456789,x\n",
    // read only partly, see `PARTIAL`
    b"p,q\nr,s\nt,u\n",
    // ends inside a quoted field
    b"\"open",
];
const PARTIAL: usize = 1;
const AFTER: &[u8] = b"x,\"y\"\"z\"\n1,2";

fn config() -> Config {
    Config::new().quote_style(QuoteStyle::Strict(false)).escape(None).comment(None)
}

fn check<D: Dfa>() {
    let expected: Records = Ok(vec![vec!["x".into(), "y\"z".into()], vec!["1".into(), "2".into()]]);
    let skip = NibbleMap::<Native>::new(&config());

    for before in BEFORE {
        // small reads so that refills happen mid-record before and after the reset
        let mut compaction = CompactionParser::<_, D>::with_capacity(*before, config(), 8, compaction::DEFAULT_FIELDS);
        let _ = drain!(compaction, read_record; PARTIAL);
        compaction.reset(AFTER);
        assert_eq!(drain!(compaction, read_record; usize::MAX), expected, "compaction after {before:?}");

        let mut branchless = CompactionParser::<_, D>::with_capacity(*before, config(), 8, compaction::DEFAULT_FIELDS);
        let _ = drain!(branchless, read_record_branchless; usize::MAX);
        branchless.reset(AFTER);
        assert_eq!(drain!(branchless, read_record_branchless; usize::MAX), expected, "branchless after {before:?}");

        let mut simd = CompactionParser::<_, D>::with_capacity(*before, config(), 8, compaction::DEFAULT_FIELDS);
        let _ = drain!(simd, read_record_simd, skip; usize::MAX);
        simd.reset(AFTER);
        assert_eq!(drain!(simd, read_record_simd, skip; usize::MAX), expected, "simd after {before:?}");

        let mut zerocopy = ZeroCopyParser::<_, D>::with_capacity(*before, config(), 8, zerocopy::DEFAULT_FIELDS);
        let _ = drain!(zerocopy, read_record; usize::MAX);
        zerocopy.reset(AFTER);
        assert_eq!(drain!(zerocopy, read_record; usize::MAX), expected, "zerocopy after {before:?}");
    }
}

#[test]
fn reset_reference() {
    check::<ReferenceDfa>();
}

#[test]
fn reset_table() {
    check::<TableDfa>();
}

#[test]
fn reset_restarts_the_position() {
    let mut parser = CompactionParser::<_, ReferenceDfa>::with_config(&b"a\nb\"\n"[..], config());
    let _ = drain!(parser, read_record; usize::MAX);
    parser.reset(&b"c\nd\"\n"[..]);
    // the error is reported as in a fresh parse, not offset by the earlier input
    let fresh = drain!(CompactionParser::<_, ReferenceDfa>::with_config(&b"c\nd\"\n"[..], config()), read_record; usize::MAX);
    assert_eq!(drain!(parser, read_record; usize::MAX), fresh);
    assert!(fresh.is_err());
}
