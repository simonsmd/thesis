//! Random inputs through every parser, against the reference.
//!
//! The same cases the fuzz targets generate (see `harness::generate`), with a small budget so it
//! runs in `cargo test`. Raise it with `PROPTEST_CASES=100000`.

use harness::{generate::Case, parsers};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn all_parsers_agree_with_the_reference(bytes in proptest::collection::vec(any::<u8>(), 5..600)) {
        let case = Case::from_bytes(&bytes).unwrap();
        if let Err(failure) = case.check(&parsers::reference(), &parsers::all()) {
            panic!("{failure}");
        }
    }
}
