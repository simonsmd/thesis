//! Every parser the harness can run, under one name each.
//!
//! Names are `<record construction>/<read_record variant>/<skip>/<dfa>` for this crate's parsers,
//! with `-` where an axis does not apply, e.g. `compaction/simd/nibble-map-32-state/bitshift`. The
//! skip is the classifier, its vector width and whether it looks for the structural bytes of each
//! state or the shared ones. The rust-csv parsers are `csv-core/dfa` and `csv/byte-record`.
//! Select them with globs.
//!
//! Each entry holds plain function pointers to fully monomorphized drivers, so dispatch costs one
//! indirect call per parse and nothing per record.

use std::fmt;

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
use csv_parser::classifier::{Pcmpestr, Pcmpistr};
use csv_parser::{
    bitshift::BitshiftDfa,
    class::ClassDfa,
    classifier::{Classifier, CmpOr, Native, NibbleMap, u8x16, u8x32, u8x64},
    dfa::Dfa,
    parser::{
        CompactionParser, Error, PerState, Sets, Shared, ZeroCopyParser, compaction, zerocopy,
    },
    reference::ReferenceDfa,
    table::TableDfa,
};
use serde::Serialize;

use crate::{check::Digest, dialect::Dialect, rust_csv};

/// The parser whose output defines correct behaviour.
pub const REFERENCE: &str = "compaction/plain/-/reference";

pub use crate::encoding::without_bom;

/// Receives every record a parser produces.
pub trait Consumer {
    fn record<'a>(&mut self, fields: impl ExactSizeIterator<Item = &'a [u8]>);
}

/// What a parse produced, as counts. Every parser does this same work per record in benchmarks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Statistics {
    pub records: u64,
    pub fields: u64,
    /// Sum of the field lengths after unescaping.
    pub field_bytes: u64,
}

impl Consumer for Statistics {
    #[inline(always)]
    fn record<'a>(&mut self, fields: impl ExactSizeIterator<Item = &'a [u8]>) {
        self.records += 1;
        self.fields += fields.len() as u64;
        // A local per record: the statistics are live across the whole parse loop, and a spilled
        // accumulator would put a store-forward on the chain of every field
        let mut bytes = 0;
        for field in fields {
            bytes += field.len() as u64;
        }
        self.field_bytes += bytes;
    }
}

/// Where a parser stopped with an error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ErrorAt {
    pub record: usize,
    pub field: usize,
    pub byte: usize,
}

impl From<Error> for ErrorAt {
    fn from(error: Error) -> Self {
        match error {
            Error::Parser(p) => ErrorAt {
                record: p.record,
                field: p.field,
                byte: p.byte,
            },
            // all input comes from memory
            Error::IO(e) => unreachable!("IO error reading from a slice: {e}"),
        }
    }
}

impl fmt::Display for ErrorAt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "error at record {}, field {}, byte {}",
            self.record, self.field, self.byte
        )
    }
}

/// Buffer sizes for one parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Options {
    /// Read size of the input buffer.
    pub capacity: usize,
    /// Initial size of the field buffers, `None` for the parser's default.
    pub fields: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            capacity: csv_parser::DEFAULT_CAPACITY,
            fields: None,
        }
    }
}

/// One record, copied out of the parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedRecord {
    pub fields: Vec<Vec<u8>>,
    /// Input offset just past the record.
    pub end: usize,
}

/// Records one at a time; ends after the first error.
pub type Records<'a> = Box<dyn Iterator<Item = Result<OwnedRecord, ErrorAt>> + 'a>;

/// A parser built once and then reset for every parse, so that only parsing is timed: building a
/// DFA and zeroing buffers can cost more than parsing a small file.
///
/// Each call parses its input from the start. The first call parses a parser fresh from
/// construction, later ones a parser reused through `reset`.
pub trait Prepared<'a> {
    /// Parse into [`Statistics`], for benchmarks.
    fn statistics(&mut self, input: &'a [u8]) -> Result<Statistics, ErrorAt>;
    /// Parse into a [`Digest`], for correctness checks.
    fn digest(&mut self, input: &'a [u8]) -> (Digest, Option<ErrorAt>);
}

/// Build a parser for a dialect. The input only fixes the lifetime of what it may parse.
pub type Prepare = for<'a> fn(&Dialect, &'a [u8], &Options) -> Box<dyn Prepared<'a> + 'a>;

pub struct Parser {
    pub name: &'static str,
    /// `Err(reason)` if the parser cannot express the dialect.
    pub supports: fn(&Dialect) -> Result<(), &'static str>,
    pub prepare: Prepare,
    /// Pull records one at a time from a fresh parser, to locate a difference.
    pub records: for<'a> fn(&Dialect, &'a [u8], &Options) -> Records<'a>,
}

impl Parser {
    /// Ours, as opposed to rust-csv: expected to agree exactly with [`REFERENCE`].
    pub fn is_ours(&self) -> bool {
        self.name.starts_with("compaction/") || self.name.starts_with("zerocopy/")
    }
}

impl fmt::Debug for Parser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

fn supports_all(_: &Dialect) -> Result<(), &'static str> {
    Ok(())
}

fn compaction<'a, D: Dfa>(
    d: &Dialect,
    input: &'a [u8],
    o: &Options,
) -> CompactionParser<&'a [u8], D> {
    let fields = o.fields.unwrap_or(compaction::DEFAULT_FIELDS);
    CompactionParser::with_capacity(input, d.config(), o.capacity, fields)
}

fn zerocopy<'a, D: Dfa>(d: &Dialect, input: &'a [u8], o: &Options) -> ZeroCopyParser<&'a [u8], D> {
    let fields = o.fields.unwrap_or(zerocopy::DEFAULT_FIELDS);
    ZeroCopyParser::with_capacity(input, d.config(), o.capacity, fields)
}

/// Defines, for a parser type and a `read_record` call on it, a [`Prepared`] implementation with
/// its constructor and a function pulling [`Records`]. `$s` is built once with the parser, for
/// state the variant takes on every call (the skipper).
macro_rules! variant {
    ($prepared:ident, $prepare:ident, $pull:ident, $ty:ident, $new:ident, <$($generic:ident: $bound:ident),*>,
     $p:ident, let $s:ident: $sty:ty = $setup:expr; $read:expr) => {
        struct $prepared<'a, $($generic: $bound),*> {
            parser: $ty<&'a [u8], D>,
            setup: $sty,
        }

        impl<'a, $($generic: $bound),*> $prepared<'a, $($generic),*> {
            #[inline(always)]
            fn consume<C: Consumer>(&mut self, input: &'a [u8], consumer: &mut C) -> Result<(), ErrorAt> {
                let $p = &mut self.parser;
                $p.reset(input);
                let $s = self.setup;
                while let Some(record) = $read? {
                    consumer.record(record.iter());
                }
                Ok(())
            }
        }

        impl<'a, $($generic: $bound),*> Prepared<'a> for $prepared<'a, $($generic),*> {
            fn statistics(&mut self, input: &'a [u8]) -> Result<Statistics, ErrorAt> {
                let mut statistics = Statistics::default();
                self.consume(input, &mut statistics)?;
                Ok(statistics)
            }

            fn digest(&mut self, input: &'a [u8]) -> (Digest, Option<ErrorAt>) {
                let mut digest = Digest::default();
                let error = self.consume(input, &mut digest).err();
                (digest, error)
            }
        }

        fn $prepare<'a, $($generic: $bound + 'static),*>(
            d: &Dialect,
            input: &'a [u8],
            o: &Options,
        ) -> Box<dyn Prepared<'a> + 'a> {
            let $p = $new::<D>(d, input, o);
            let setup: $sty = $setup;
            Box::new($prepared { parser: $p, setup })
        }

        fn $pull<'a, $($generic: $bound + 'static,)*>(
            d: &Dialect,
            input: &'a [u8],
            o: &Options,
        ) -> Records<'a> {
            let mut $p = $new::<D>(d, input, o);
            let $s: $sty = $setup;
            let mut done = false;
            Box::new(std::iter::from_fn(move || {
                if done {
                    return None;
                }
                match $read {
                    Ok(Some(record)) => {
                        let fields = record.iter().map(<[u8]>::to_vec).collect();
                        Some(Ok(OwnedRecord { fields, end: $p.position().byte }))
                    }
                    Ok(None) => {
                        done = true;
                        None
                    }
                    Err(e) => {
                        done = true;
                        Some(Err(e.into()))
                    }
                }
            }))
        }
    };
}

variant!(CompactionPlain, compaction_plain, compaction_plain_records, CompactionParser, compaction,
    <D: Dfa>, p, let _s: () = (); p.read_record());
variant!(CompactionBranchless, compaction_branchless, compaction_branchless_records, CompactionParser,
    compaction, <D: Dfa>, p, let _s: () = (); p.read_record_branchless());
variant!(CompactionSimdCheck, compaction_simd_check, compaction_simd_check_records, CompactionParser,
    compaction, <D: Dfa, S: Classifier, M: Sets>, p, let s: (S, M) = (S::new(p.config()), M::default());
    p.read_record_simd_check(&s.0, s.1));
variant!(CompactionSimd, compaction_simd, compaction_simd_records, CompactionParser, compaction,
    <D: Dfa, S: Classifier, M: Sets>, p, let s: (S, M) = (S::new(p.config()), M::default());
    p.read_record_simd(&s.0, s.1));
variant!(ZeroCopyPlain, zerocopy_plain, zerocopy_plain_records, ZeroCopyParser, zerocopy,
    <D: Dfa>, p, let _s: () = (); p.read_record());
variant!(ZeroCopySimd, zerocopy_simd, zerocopy_simd_records, ZeroCopyParser, zerocopy,
    <D: Dfa, S: Classifier, M: Sets>, p, let s: (S, M) = (S::new(p.config()), M::default());
    p.read_record_simd(&s.0, s.1));

/// One registry entry from a variant instantiated for the given type arguments.
macro_rules! entry {
    ($name:expr, $prepare:ident, $pull:ident, <$($t:ty),*>) => {
        Parser {
            name: $name,
            supports: supports_all,
            prepare: $prepare::<$($t),*>,
            records: $pull::<$($t),*>,
        }
    };
}

/// The entries of a skipping `read_record` variant for one classifier, with either set:
/// `<skip>-shared` and `<skip>-state`.
macro_rules! sets {
    ($variant:literal, $prepare:ident, $pull:ident, $dfa:ty, $name:literal, $skip:literal, $classifier:ty) => {
        [
            entry!(concat!($variant, "/", $skip, "-shared/", $name), $prepare, $pull,
                <$dfa, $classifier, Shared>),
            entry!(concat!($variant, "/", $skip, "-state/", $name), $prepare, $pull,
                <$dfa, $classifier, PerState>),
        ]
    };
}

/// The entries of a skipping `read_record` variant for every classifier and vector width.
macro_rules! classifiers {
    ($variant:literal, $prepare:ident, $pull:ident, $dfa:ty, $name:literal) => {{
        let mut entries = Vec::new();
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "cmp-or-16",
            CmpOr<u8x16<Native>>
        ));
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "cmp-or-32",
            CmpOr<u8x32<Native>>
        ));
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "cmp-or-64",
            CmpOr<u8x64<Native>>
        ));
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "nibble-map-16",
            NibbleMap<u8x16<Native>>
        ));
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "nibble-map-32",
            NibbleMap<u8x32<Native>>
        ));
        entries.extend(sets!(
            $variant,
            $prepare,
            $pull,
            $dfa,
            $name,
            "nibble-map-64",
            NibbleMap<u8x64<Native>>
        ));
        // SSE 4.2, which every AVX2 cpu has
        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            target_feature = "avx2"
        ))]
        {
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpistr-16",
                Pcmpistr<u8x16<Native>>
            ));
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpistr-32",
                Pcmpistr<u8x32<Native>>
            ));
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpistr-64",
                Pcmpistr<u8x64<Native>>
            ));
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpestr-16",
                Pcmpestr<u8x16<Native>>
            ));
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpestr-32",
                Pcmpestr<u8x32<Native>>
            ));
            entries.extend(sets!(
                $variant,
                $prepare,
                $pull,
                $dfa,
                $name,
                "pcmpestr-64",
                Pcmpestr<u8x64<Native>>
            ));
        }
        entries
    }};
}

/// The `read_record` variants without simd for one DFA.
macro_rules! scalar {
    ($dfa:ty, $name:literal) => {
        [
            entry!(concat!("compaction/plain/-/", $name), compaction_plain, compaction_plain_records, <$dfa>),
            entry!(concat!("compaction/branchless/-/", $name), compaction_branchless,
                compaction_branchless_records, <$dfa>),
            entry!(concat!("zerocopy/plain/-/", $name), zerocopy_plain, zerocopy_plain_records, <$dfa>),
        ]
    };
}

/// The skipping `read_record` variants for one DFA.
macro_rules! simd {
    ($dfa:ty, $name:literal) => {
        classifiers!(
            "compaction/simd",
            compaction_simd,
            compaction_simd_records,
            $dfa,
            $name
        )
        .into_iter()
        .chain(classifiers!(
            "compaction/simd-check",
            compaction_simd_check,
            compaction_simd_check_records,
            $dfa,
            $name
        ))
        .chain(classifiers!(
            "zerocopy/simd",
            zerocopy_simd,
            zerocopy_simd_records,
            $dfa,
            $name
        ))
    };
}

/// All parsers, [`REFERENCE`] first.
///
/// The variants without simd exist for every DFA. The simd variants only exist for the fastest
/// DFA, bitshift: every further DFA would add all classifiers again and take long to compile.
///
/// A new DFA is one more `scalar!` line; a new `read_record` variant is a `variant!` plus its
/// entries in `scalar!` or `simd!`.
pub fn all() -> Vec<Parser> {
    let mut parsers: Vec<Parser> = [
        scalar!(ReferenceDfa, "reference"),
        scalar!(BitshiftDfa, "bitshift"),
        scalar!(TableDfa, "table"),
        scalar!(ClassDfa, "class"),
    ]
    .into_iter()
    .flatten()
    .collect();
    parsers.extend(simd!(BitshiftDfa, "bitshift"));
    parsers.extend(rust_csv::parsers());
    debug_assert_eq!(parsers[0].name, REFERENCE);
    parsers
}

pub fn reference() -> Parser {
    all().into_iter().next().unwrap()
}

/// The parsers matching any of `include` (all if empty) and none of `exclude`, in registry order.
pub fn select(include: &[String], exclude: &[String]) -> Result<Vec<Parser>, String> {
    for pattern in include {
        if !all().iter().any(|p| glob(pattern, p.name)) {
            let names: Vec<_> = all().iter().map(|p| p.name).collect();
            return Err(format!(
                "no parser matches {pattern:?}; available:\n  {}",
                names.join("\n  ")
            ));
        }
    }
    Ok(all()
        .into_iter()
        .filter(|p| include.is_empty() || include.iter().any(|g| glob(g, p.name)))
        .filter(|p| !exclude.iter().any(|g| glob(g, p.name)))
        .collect())
}

/// Shell-style matching where `*` matches any run of characters (including `/`) and `?` one.
pub fn glob(pattern: &str, name: &str) -> bool {
    let (p, n) = (pattern.as_bytes(), name.as_bytes());
    let (mut pi, mut ni) = (0, 0);
    // position after the last `*` and the name position it is currently matched up to
    let mut star = None;
    while ni < n.len() {
        match p.get(pi) {
            Some(b'*') => {
                star = Some((pi + 1, ni));
                pi += 1;
            }
            Some(&c) if c == b'?' || c == n[ni] => {
                pi += 1;
                ni += 1;
            }
            _ => match star {
                Some((sp, sn)) => {
                    pi = sp;
                    ni = sn + 1;
                    star = Some((sp, sn + 1));
                }
                None => return false,
            },
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob("*", "compaction/plain/-/table"));
        assert!(glob(
            "compaction/*/table",
            "compaction/simd/nibble-map/table"
        ));
        assert!(glob(
            "*/nibble-map/*",
            "compaction/simd-check/nibble-map/class"
        ));
        assert!(!glob("compaction/*/table", "compaction/plain/-/class"));
        assert!(glob("csv-core/?fa", "csv-core/dfa"));
        assert!(!glob("csv-core", "csv-core/dfa"));
    }

    #[test]
    fn names_are_unique() {
        let mut names: Vec<_> = all().iter().map(|p| p.name).collect();
        names.sort();
        let len = names.len();
        names.dedup();
        assert_eq!(names.len(), len);
    }
}
