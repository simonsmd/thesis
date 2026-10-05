use std::{io::Read, str::Utf8Error};

use super::{Error, Exit, INITIAL_FIELDS, Position};
use crate::{
    buffer::{Buffer, DEFAULT_CAPACITY, Refill},
    config::{Config, QuoteStyle},
    dfa::Dfa,
};

#[derive(Debug)]
pub struct ZeroCopyParser<R: Read, D: Dfa> {
    config: Config,
    dfa: D,
    reader: R,
    buffer: Buffer,
    read_index: usize,
    /// Start of every finished field, relative to `record_start`.
    field_starts: Vec<usize>,
    /// End of every finished field, relative to `record_start`.
    field_ends: Vec<usize>,
    position: Position,
    /// store if this parser encountered an error so read_record does not start parsing from an
    /// invalid state
    error: bool,
}

impl<R: Read, D: Dfa> ZeroCopyParser<R, D> {
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
            field_starts: vec![0; INITIAL_FIELDS],
            field_ends: vec![0; INITIAL_FIELDS],
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

impl<R: Read, D: Dfa> ZeroCopyParser<R, D> {
    #[inline]
    pub fn read_record(&mut self) -> Result<Option<Record<'_>>, Error> {
        if self.error {
            return Err(Error::Parser(self.position));
        }

        let mut record_start = self.read_index;
        let mut write = self.read_index;
        let mut field_start = self.read_index;
        let mut field_count = 0;
        let mut field_not_started: usize = 1;

        // A record always starts in the StartRecord (equivalent to EndRecord) state so there is no
        // need to store the DFA state between records
        let mut state = self.dfa.start_state();

        let is_error = loop {
            if self.read_index == self.buffer.len() {
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, record_start, self.read_index)
                    .map_err(Error::IO)?;
                field_start -= shift;
                record_start -= shift;
                self.read_index -= shift;
                write -= shift;

                if read == 0 {
                    if (self.dfa.in_quoted_field(state)
                        && matches!(self.config.quote_style, QuoteStyle::Strict(_)))
                        || self.dfa.in_escape_sequence(state)
                    {
                        break true;
                    }

                    if write == field_start && field_count == 0 {
                        return Ok(None);
                    }
                    
                    // EOF right after a field delimiter (field_start has not been updated since
                    // the last field)
                    if field_not_started & 1 == 1 {
                        field_start = write;
                    }

                    self.field_starts[field_count] = field_start - record_start;
                    self.field_ends[field_count] = write - record_start;
                    field_count += 1;
                    break false;
                }
            }

            // keep variables used in the hotloop in registers
            let dfa = &self.dfa;
            let buffer = self.buffer.data_mut();
            let field_starts= self.field_starts.as_mut_slice();
            let field_ends = self.field_ends.as_mut_slice();
            let mut read = self.read_index;

            let exit = loop {
                if read >= buffer.len() {
                    break Exit::Refill;
                }

                let byte = buffer[read];
                state = dfa.transition(state, byte);

                if write < read{
                    buffer[write] = byte;
                }

                // TODO: break dependency chain between this update and write update since write
                // update can use old value of field_not_started
                field_not_started &= dfa.is_discard(state);

                // update the write pointer using one adc instruction with the field_not_started
                // bit as the operand and the is_output bit in the carry flag on x86
                #[cfg(target_arch = "x86_64")]
                {
                    let mut write_u64 = 0;
                    core::arch::x86_64::_addcarry_u64(dfa.is_output(state) as u8, write as u64, field_not_started as u64, &mut write_u64);
                    write = write_u64 as usize;
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    write += field_not_started | dfa.is_output(state) as usize;
                }

                field_start += field_not_started;

                read += 1;


                if dfa.is_field_end(state) {
                    if dfa.is_error(state) {
                        // report the byte where the error occured which was not consumed
                        read -= 1;
                        break Exit::Error;
                    }

                    field_starts[field_count] = field_start - record_start;
                    field_ends[field_count] = write - record_start;
                    field_count += 1;
                    field_not_started = 1;
                    field_start = read;
                    write = read;

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
                    self.field_starts.resize(len, 0);
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
            buffer: &self.buffer.data()[record_start..self.read_index],
            field_starts: &self.field_starts[..field_count],
            field_ends: &self.field_ends[..field_count],
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    buffer: &'a [u8],
    field_starts: &'a [usize],
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
        let start = self.field_starts[index];
        Some(&self.buffer[start..end])
    }

    /// Iterate over the fields as byte slices.
    pub fn iter(&self) -> Fields<'a> {
        Fields {
            buffer: self.buffer,
            field_starts: self.field_starts.iter(),
            field_ends: self.field_ends.iter(),
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
    field_starts: std::slice::Iter<'a, usize>,
    field_ends: std::slice::Iter<'a, usize>,
}

impl<'a> Iterator for Fields<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let end = *self.field_ends.next()?;
        let start = *self.field_starts.next()?;
        Some(&self.buffer[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.field_ends.size_hint()
    }
}

impl ExactSizeIterator for Fields<'_> {}
