//! Benchmark and correctness harness for `csv-parser`.
//!
//! - [`dataset`] and [`inputs`] read the datasets built by `scripts/` and pick files to run on.
//! - [`parsers`] is the registry of everything that can be run, including [`rust_csv`].
//! - [`check`] compares parsers against the reference; [`generate`] makes inputs for fuzzing.
//! - The binaries are `csvbench` (throughput), `csvcheck` (correctness on datasets) and `csvcat`
//!   (print the records of one file).

pub mod check;
pub mod counters;
pub mod dataset;
pub mod dialect;
pub mod generate;
pub mod inputs;
pub mod parsers;
pub mod rust_csv;
pub mod system;
