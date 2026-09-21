use std::{fmt::Display, io::Read, str::Utf8Error};

use crate::{
    buffer::{Buffer, DEFAULT_CAPACITY, Refill},
    config::Config,
};

#[derive(Debug)]
pub struct Parser<R: Read, D: Dfa> {
    config: Config,
    dfa: D,
    reader: R,
    buffer: Buffer,
    record_start: usize,
    read_index: usize,
    write_index: usize,
    field_ends: Vec<usize>,
    position: Position,
}

impl<R: Read, D: Dfa> Parser<R, D> {
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, Config::default())
    }

    pub fn with_config(reader: R, config: Config) -> Self {
        Self::with_capacity(reader, config, DEFAULT_CAPACITY)
    }

    /// `capacity` is the read size; the buffer only grows past it for records
    /// that do not fit.
    pub fn with_capacity(reader: R, config: Config, capacity: usize) -> Self {
        let dfa = D::new(&config);
        Self {
            config,
            dfa,
            reader,
            buffer: Buffer::with_capacity(capacity),
            record_start: 0,
            read_index: 0,
            write_index: 0,
            field_ends: Vec::new(),
            position: Position::default(),
        }
    }

    pub fn position(&self) -> Position {
        self.position
    }

    /// The dialect this parser was built with.
    ///
    /// It cannot be changed after construction: a [`Dfa`] may bake the config
    /// into a transition table when it is created.
    pub fn config(&self) -> &Config {
        &self.config
    }
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

pub trait Dfa {
    /// Create an instance of this Dfa with the provided configuration
    ///
    /// Used to initialize state like transition tables.
    fn new(config: &Config) -> Self;

    /// Process the buffer of csv data and return when a record ends or the buffer is exhausted
    ///
    /// The fields of a record are compacted continously starting at record_start and the field
    /// boundries are recorded in the field_ends array (relative to record_start).
    fn run(
        &mut self,
        buffer: &mut [u8],
        read_index: &mut usize,
        write_index: &mut usize,
        record_start: &mut usize,
        field_ends: &mut Vec<usize>,
    ) -> DfaResult;

    /// Decide whether the input may end where it currently does.
    ///
    /// Called once the reader is exhausted. Returns [`DfaResult::Error`] if the
    /// input stopped in a state that is not a valid place to end, such as an
    /// unterminated quoted field in a strict dialect, and
    /// [`DfaResult::EndRecord`] otherwise.
    fn finish(&self, config: &Config) -> DfaResult;
}

#[derive(Debug, PartialEq, Eq)]
pub enum DfaResult {
    EndRecord,
    Refill,
    Error,
}

impl<R: Read, D: Dfa> Parser<R, D> {
    pub fn read_record(&mut self) -> Result<Option<Record<'_>>, Error> {
        self.record_start = self.write_index;
        self.field_ends.clear();

        let result = loop {
            if self.read_index == self.buffer.len() {
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, self.record_start, self.write_index)
                    .map_err(Error::IO)?;
                self.record_start -= shift;
                self.write_index -= shift;
                self.read_index = self.write_index;

                if read == 0 {
                    if self.dfa.finish(&self.config) == DfaResult::Error {
                        break DfaResult::Error;
                    }

                    if self.write_index == self.record_start && self.field_ends.is_empty() {
                        return Ok(None);
                    }

                    self.field_ends.push(self.write_index - self.record_start);
                    break DfaResult::EndRecord;
                }
            }

            match self.dfa.run(self.buffer.data_mut(), &mut self.read_index, &mut self.write_index, &mut self.record_start, &mut self.field_ends) {
                DfaResult::Refill => continue,
                result => break result,
            }
        };

        self.position.field = self.field_ends.len();
        self.position.byte = self.buffer.source_offset(self.read_index);

        if result == DfaResult::Error {
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        Ok(Some(Record {
            buffer: &self.buffer.data()[self.record_start..self.write_index],
            field_ends: &self.field_ends,
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    buffer: &'a [u8],
    field_ends: &'a [usize],
}

impl<'a> Record<'a> {
    /// Number of fields in the record.
    pub fn len(&self) -> usize {
        self.field_ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.field_ends.is_empty()
    }

    /// The raw bytes of field `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        let end = *self.field_ends.get(index)?;
        let start = if index == 0 {
            0
        } else {
            self.field_ends[index - 1]
        };
        Some(&self.buffer[start..end])
    }

    /// Iterate over the fields as byte slices.
    pub fn iter(&self) -> Fields<'a> {
        Fields {
            buffer: self.buffer,
            field_ends: self.field_ends.iter(),
            start: 0,
        }
    }

    pub fn get_str(&self, i: usize) -> Option<Result<&'a str, Utf8Error>> {
        self.get(i).map(std::str::from_utf8)
    }

    pub fn iter_str(&self) -> impl Iterator<Item = Result<&'a str, Utf8Error>> {
        self.iter().map(std::str::from_utf8)
    }
}

impl<'a> IntoIterator for Record<'a> {
    type Item = &'a [u8];
    type IntoIter = Fields<'a>;

    fn into_iter(self) -> Fields<'a> {
        self.iter()
    }
}

/// Iterator over the fields of a [`Record`].
#[derive(Debug, Clone)]
pub struct Fields<'a> {
    buffer: &'a [u8],
    field_ends: std::slice::Iter<'a, usize>,
    start: usize,
}

impl<'a> Iterator for Fields<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let end = *self.field_ends.next()?;
        let field = &self.buffer[self.start..end];
        self.start = end;
        Some(field)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.field_ends.size_hint()
    }
}

impl ExactSizeIterator for Fields<'_> {}
