//! Every one of our parsers against the reference, on random dialects, inputs and buffer sizes.
//!
//! Run with `RUSTFLAGS="-C target-cpu=native" cargo +nightly fuzz run differential`: cargo-fuzz
//! sets RUSTFLAGS itself, which replaces the flags from `.cargo/config.toml`, and without
//! `target-cpu=native` the SIMD skippers would be fuzzed on SSE2 instead of AVX2.
#![no_main]

#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
compile_error!("build with RUSTFLAGS=\"-C target-cpu=native\" so that the AVX2 skippers are fuzzed");

use std::sync::LazyLock;

use harness::{
    generate::Case,
    parsers::{self, Parser},
};
use libfuzzer_sys::fuzz_target;

static REFERENCE: LazyLock<Parser> = LazyLock::new(parsers::reference);
static OURS: LazyLock<Vec<Parser>> = LazyLock::new(|| parsers::all().into_iter().filter(Parser::is_ours).collect());

fuzz_target!(|bytes: &[u8]| {
    if let Some(case) = Case::from_bytes(bytes) {
        if let Err(failure) = case.check(&REFERENCE, &OURS) {
            panic!("{failure}");
        }
    }
});
