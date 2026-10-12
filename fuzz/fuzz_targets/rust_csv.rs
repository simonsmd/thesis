//! rust-csv against the reference, where the two are expected to agree (see `Case::check`).
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo +nightly fuzz run rust_csv`.
#![no_main]

use std::sync::LazyLock;

use harness::{
    generate::Case,
    parsers::{self, Parser},
};
use libfuzzer_sys::fuzz_target;

static REFERENCE: LazyLock<Parser> = LazyLock::new(parsers::reference);
static RUST_CSV: LazyLock<Vec<Parser>> =
    LazyLock::new(|| parsers::all().into_iter().filter(|p| !p.is_ours()).collect());

fuzz_target!(|bytes: &[u8]| {
    if let Some(case) = Case::from_bytes(bytes) {
        if let Err(failure) = case.check(&REFERENCE, &RUST_CSV) {
            panic!("{failure}");
        }
    }
});
