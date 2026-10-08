use std::{io::Read, str::Utf8Error};

use super::{Error, Exit, Position};
use crate::{
    buffer::{Buffer, DEFAULT_CAPACITY, Refill},
    config::{Config, QuoteStyle},
    dfa::Dfa,
};

#[derive(Debug)]
pub struct CompactionParser<R: Read, D: Dfa> {
    config: Config,
    dfa: D,
    reader: R,
    buffer: Buffer,
    read_index: usize,
    field_ends: Vec<usize>,
    position: Position,
    /// store if this parser encountered an error so read_record does not start parsing from an
    /// invalid state
    error: bool,
}

impl<R: Read, D: Dfa> CompactionParser<R, D> {
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
            read_index: 0,
            field_ends: vec![0; DEFAULT_CAPACITY + 1024],
            position: Position::default(),
            error: false,
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

/// Minimum gap between write and read index at the start of a record and after a refill. 4 bytes
/// were enough on Alder Lake but leave some extra space for other cpus (only costs a few extra
/// refills)
const MIN_GAP: usize = 64;

impl<R: Read, D: Dfa> CompactionParser<R, D> {
    #[inline]
    pub fn read_record(&mut self) -> Result<Option<Record<'_>>, Error> {
        if self.error {
            return Err(Error::Parser(self.position));
        }

        // Start the compacted record with a gap to the read index: if write trails read by only a
        // few bytes the cpu falsely detects a conflict between the store (whose address is known
        // late) and the loads of the next bytes, and flushes the pipeline or stops loading ahead
        let mut record_start = self.read_index.saturating_sub(MIN_GAP);
        let mut write = record_start;
        let mut field_count = 0;

        // A record always starts in the StartRecord (equivalent to EndRecord) state so there is no
        // need to store the DFA state between records
        let mut state = self.dfa.start_state();

        let is_error = loop {
            if self.read_index == self.buffer.len() {
                // read the new input MIN_GAP bytes after the write index, so the gap does not
                // drop below MIN_GAP for the rest of the record
                // TODO: make continuing after IO errors safe
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, record_start, write, MIN_GAP)
                    .map_err(Error::IO)?;
                record_start -= shift;
                write -= shift;
                self.read_index = write + MIN_GAP;

                if read == 0 {
                    if (self.dfa.in_quoted_field(state)
                        && matches!(self.config.quote_style, QuoteStyle::Strict(_)))
                        || self.dfa.in_escape_sequence(state)
                    {
                        break true;
                    }

                    if write == record_start && field_count == 0 {
                        return Ok(None);
                    }

                    self.field_ends[field_count] = write - record_start;
                    field_count += 1;
                    break false;
                }
            }

            // keep variables used in the hotloop in registers
            let dfa = &self.dfa;
            let buffer = self.buffer.data_mut();
            let field_ends = self.field_ends.as_mut_slice();
            let mut read = self.read_index;

            let exit = loop {
                if read >= buffer.len() {
                    break Exit::Refill;
                }

                let byte = buffer[read];
                state = dfa.transition(state, byte);

                // SAFETY: write <= read < buffer.len()
                unsafe {
                    *buffer.get_unchecked_mut(write) = byte;
                }
                write += dfa.is_output(state) as usize;
                read += 1;

                if dfa.is_field_end(state) {
                    if dfa.is_error(state) {
                        // report the byte where the error occured which was not consumed
                        read -= 1;
                        break Exit::Error;
                    }

                    field_ends[field_count] = write - record_start;
                    field_count += 1;

                    if dfa.is_record_end(state) {
                        break Exit::EndRecord;
                    }

                    if field_count == field_ends.len() {
                        break Exit::FieldEndsFull;
                    }
                }
            };

            self.read_index = read;

            match exit {
                Exit::Refill => {}
                Exit::FieldEndsFull => {
                    let len = self.field_ends.len() * 2;
                    self.field_ends.resize(len, 0);
                }
                Exit::EndRecord => break false,
                Exit::Error => break true,
            }
        };

        self.position.byte = self.buffer.source_offset(self.read_index);

        if is_error {
            self.error = true;
            self.position.field = field_count;
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        Ok(Some(Record {
            buffer: &self.buffer.data()[record_start..write],
            field_ends: &self.field_ends[..field_count],
        }))
    }

    #[inline]
    pub fn read_record_branchless(&mut self) -> Result<Option<Record<'_>>, Error> {
        if self.error {
            return Err(Error::Parser(self.position));
        }

        // Start the compacted record with a gap to the read index: if write trails read by only a
        // few bytes the cpu falsely detects a conflict between the store (whose address is known
        // late) and the loads of the next bytes, and flushes the pipeline or stops loading ahead
        let mut record_start = self.read_index.saturating_sub(MIN_GAP);
        let mut write = record_start;
        let mut field_count = 0;

        // A record always starts in the StartRecord (equivalent to EndRecord) state so there is no
        // need to store the DFA state between records
        let mut state = self.dfa.start_state();

        let is_error = loop {
            if self.read_index == self.buffer.len() {
                // read the new input MIN_GAP bytes after the write index, so the gap does not
                // drop below MIN_GAP for the rest of the record
                // TODO: make continuing after IO errors safe
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, record_start, write, MIN_GAP)
                    .map_err(Error::IO)?;
                record_start -= shift;
                write -= shift;
                self.read_index = write + MIN_GAP;

                for i in 0..field_count {
                    self.field_ends[i] -= shift;
                }

                // resize field_ends if there is not enough space for the maximum possible amount
                // of fields (one per input byte)
                if self.field_ends.len() - field_count < self.buffer.len() {
                    self.field_ends.resize(self.buffer.capacity() + 1024, 0);
                }

                if read == 0 {
                    if (self.dfa.in_quoted_field(state)
                        && matches!(self.config.quote_style, QuoteStyle::Strict(_)))
                        || self.dfa.in_escape_sequence(state)
                    {
                        break true;
                    }

                    if write == record_start && field_count == 0 {
                        return Ok(None);
                    }

                    self.field_ends[field_count] = write;
                    field_count += 1;
                    break false;
                }
            }

            // keep variables used in the hotloop in registers
            let dfa = &self.dfa;
            let buffer = self.buffer.data_mut();
            let field_ends = self.field_ends.as_mut_slice();
            let mut read = self.read_index;

            let exit = loop {
                if read >= buffer.len() {
                    break Exit::Refill;
                }

                let byte = buffer[read];
                state = dfa.transition(state, byte);

                // SAFETY: write <= read < buffer.len()
                unsafe {
                    *buffer.get_unchecked_mut(write) = byte;
                }
                write += dfa.is_output(state) as usize;
                read += 1;

                // SAFETY: The refill path ensures that the field_ends buffer is always large
                // enough (there can never be more than one field per byte)
                unsafe {
                    *field_ends.get_unchecked_mut(field_count) = write;
                }
                field_count += dfa.is_field_end(state) as usize;

                if dfa.is_record_end(state) {
                    if dfa.is_error(state) {
                        // report the byte where the error occured which was not consumed
                        read -= 1;
                        field_count -= 1;
                        break Exit::Error;
                    }

                    break Exit::EndRecord;
                }
            };

            self.read_index = read;

            match exit {
                Exit::Refill => {}
                Exit::FieldEndsFull => {
                    let len = self.field_ends.len() * 2;
                    self.field_ends.resize(len, 0);
                }
                Exit::EndRecord => break false,
                Exit::Error => break true,
            }
        };

        self.position.byte = self.buffer.source_offset(self.read_index);

        if is_error {
            self.error = true;
            self.position.field = field_count;
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        for i in 0..field_count {
            self.field_ends[i] -= record_start;
        }

        Ok(Some(Record {
            buffer: &self.buffer.data()[record_start..write],
            field_ends: &self.field_ends[..field_count],
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
