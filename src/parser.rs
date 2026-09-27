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
    cursor: Cursor,
    field_ends: Vec<usize>,
    position: Position,
}

/// The parser's positions in the buffer, bundled so a [`Dfa`] takes them
/// through one pointer instead of one argument each.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cursor {
    /// Next input byte to consume.
    pub read: usize,
    /// Where the next output byte is compacted to. Always `<= read`.
    pub write: usize,
    /// Start of the current record's output bytes.
    pub record_start: usize,
    /// Number of field ends recorded for the current record.
    pub field_count: usize,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Position {
    pub byte: usize,
    pub record: usize,
    pub field: usize,
}

/// Initial size for the field_ends buffer
const INITIAL_FIELDS: usize = 1024;

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
            cursor: Cursor::default(),
            field_ends: vec![0; INITIAL_FIELDS],
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

    /// Process the buffer of csv data and return when a record ends, the buffer is exhausted or
    /// `field_ends` is full.
    ///
    /// The fields of a record are compacted continously starting at `cursor.record_start` and the
    /// field boundries are written to field_ends (relative to
    /// `record_start`) with the number of fields recorded in `cursor.field_count`. The caller must
    /// guarantee `cursor.field_count < field_ends.len()` on entry. When the field_ends buffer
    /// overflows ([`DfaResult::FieldEndsFull`]) the caller must grow the buffer.
    fn run(
        &mut self,
        buffer: &mut [u8],
        cursor: &mut Cursor,
        field_ends: &mut [usize],
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
    FieldEndsFull,
    Error,
}

impl<R: Read, D: Dfa> Parser<R, D> {
    pub fn read_record(&mut self) -> Result<Option<Record<'_>>, Error> {
        let cursor = &mut self.cursor;
        cursor.record_start = cursor.write;
        cursor.field_count = 0;

        let result = loop {
            if cursor.read == self.buffer.len() {
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, cursor.record_start, cursor.write)
                    .map_err(Error::IO)?;
                cursor.record_start -= shift;
                cursor.write -= shift;
                cursor.read = cursor.write;

                if read == 0 {
                    if self.dfa.finish(&self.config) == DfaResult::Error {
                        break DfaResult::Error;
                    }

                    if cursor.write == cursor.record_start && cursor.field_count == 0 {
                        return Ok(None);
                    }

                    self.field_ends[cursor.field_count] = cursor.write - cursor.record_start;
                    cursor.field_count += 1;
                    break DfaResult::EndRecord;
                }
            }

            match self
                .dfa
                .run(self.buffer.data_mut(), cursor, &mut self.field_ends)
            {
                DfaResult::Refill => continue,
                DfaResult::FieldEndsFull => {
                    let len = self.field_ends.len() * 2;
                    self.field_ends.resize(len, 0);
                }
                result => break result,
            }
        };

        self.position.field = cursor.field_count;
        self.position.byte = self.buffer.source_offset(cursor.read);

        if result == DfaResult::Error {
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        Ok(Some(Record {
            buffer: &self.buffer.data()[cursor.record_start..cursor.write],
            field_ends: &self.field_ends[..cursor.field_count],
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
