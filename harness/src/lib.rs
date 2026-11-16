//! Benchmark and correctness harness for `csv-parser`.
//!
//! - [`dataset`] and [`inputs`] read the datasets built by `scripts/` and pick files to run on.
//! - [`encoding`] converts every input to UTF-8 before it is parsed.
//! - [`parsers`] is the registry of everything that can be run, including [`rust_csv`] and, with the
//!   `csveee` feature, `csveee`.
//! - [`check`] compares parsers against the reference; [`generate`] makes inputs for fuzzing.
//! - [`sniff`] guesses the dialect of a file and describes its content (binary `csvsniff`).
//! - The other binaries are `csvbench` (throughput), `csvcheck` (correctness on datasets) and `csvcat`
//!   (print the records of one file).

pub mod check;
pub mod counters;
#[cfg(feature = "csveee")]
pub mod csveee;
pub mod dataset;
pub mod dialect;
pub mod encoding;
pub mod generate;
pub mod inputs;
pub mod parsers;
pub mod rust_csv;
pub mod sniff;
pub mod system;
