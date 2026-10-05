//! Parsers driving a [`Dfa`](crate::dfa::Dfa) over the input.
//!
//! They share the DFA and the buffering and differ in how records are assembled:
//!
//! - [`CompactionParser`] writes the output bytes of every field back into the buffer, so
//!   escapes and quotes are removed by compacting the record in place.
//! - [`ZeroCopyParser`] hands out slices of the input as is and only copies fields that
//!   contain escaped characters or quotes.

pub mod compaction;
pub mod zerocopy;

use std::fmt::Display;

pub use compaction::CompactionParser;
pub use zerocopy::ZeroCopyParser;

/// Initial size for the field_ends buffer
const INITIAL_FIELDS: usize = 1024;

/// Why the core loop of `read_record` stopped.
enum Exit {
    EndRecord,
    Refill,
    FieldEndsFull,
    Error,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Position {
    pub byte: usize,
    pub record: usize,
    pub field: usize,
}

impl Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "record {}, field {}, byte {}",
            self.record, self.field, self.byte
        )
    }
}

#[derive(Debug)]
pub enum Error {
    Parser(Position),
    IO(std::io::Error),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Parser(position) => write!(f, "parser error at {position}"),
            Error::IO(error) => write!(f, "IO error: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Parser(_) => None,
            Error::IO(error) => Some(error),
        }
    }
}
