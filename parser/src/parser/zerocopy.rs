use std::{io::Read, str::Utf8Error};

use fearless_simd::SimdMask;

use super::{Error, Exit, Position, Sets, Stopped, compaction::StructuralBlock};
use crate::{
    buffer::{Buffer, DEFAULT_CAPACITY, Refill},
    classifier::{self, Classifier},
    config::{Config, QuoteStyle},
    dfa::Dfa,
};

/// Initial size of the field buffers.
pub const DEFAULT_FIELDS: usize = 4096;

#[derive(Debug)]
pub struct ZeroCopyParser<R: Read, D: Dfa> {
    config: Config,
    dfa: D,
    reader: R,
    buffer: Buffer,
    read_index: usize,
    /// stores `[start, end]` index per field, relative to `record_start`
    fields: Vec<[usize; 2]>,
    position: Position,
    /// The error that stopped this parser, so read_record does not start parsing from an
    /// invalid state
    error: Option<Stopped>,
    /// The masks of the last block `read_record_simd` classified, so a record starting in
    /// the same block does not classify it again
    structural_block: StructuralBlock,
    /// [`classifier::ordinary_byte`] of the config
    ordinary: u8,
}

impl<R: Read, D: Dfa> ZeroCopyParser<R, D> {
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, Config::default())
    }

    pub fn with_config(reader: R, config: Config) -> Self {
        Self::with_capacity(reader, config, DEFAULT_CAPACITY, DEFAULT_FIELDS)
    }

    /// `capacity` is the read size; the buffer only grows past it for records that do not fit.
    /// `fields` is the initial size of the field buffers, which grow on demand. Small sizes of
    /// either exercise the growth paths in tests.
    pub fn with_capacity(reader: R, config: Config, capacity: usize, fields: usize) -> Self {
        let dfa = D::new(&config);
        let ordinary = classifier::ordinary_byte(&config);
        Self {
            config,
            dfa,
            reader,
            buffer: Buffer::with_capacity(capacity),
            read_index: 0,
            fields: vec![[0; 2]; fields.max(1)],
            position: Position::default(),
            error: None,
            structural_block: StructuralBlock::NONE,
            ordinary,
        }
    }

    /// Start over on a new input, keeping the DFA and the allocated buffers.
    pub fn reset(&mut self, reader: R) {
        self.reader = reader;
        self.buffer.clear();
        self.read_index = 0;
        self.position = Position::default();
        self.error = None;
        self.structural_block = StructuralBlock::NONE;
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
        if let Some(stopped) = self.error {
            return Err(stopped.error(self.position));
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
                    .fill(&mut self.reader, record_start, self.read_index, 0)
                    .map_err(|e| {
                        self.error = Some(Stopped::IO);
                        Error::IO(e)
                    })?;
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

                    self.fields[field_count] = [field_start - record_start, write - record_start];
                    field_count += 1;
                    break false;
                }
            }

            // keep variables used in the hotloop in registers
            let dfa = &self.dfa;
            let buffer = self.buffer.data_mut();
            let mut read = self.read_index;
            // Use a start and end pointer instaed of a slice with a count which saves one
            // register (the slice length is not needed).
            let fields = self.fields.as_mut_ptr_range();
            // SAFETY: field_count < fields.len() at every entry of the core loop
            let mut field = unsafe { fields.start.add(field_count) };

            let exit = loop {
                if read >= buffer.len() {
                    break Exit::Refill;
                }

                let byte = buffer[read];
                state = dfa.transition(state, byte);

                if write < read {
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
                    core::arch::x86_64::_addcarry_u64(
                        dfa.is_output(state) as u8,
                        write as u64,
                        field_not_started as u64,
                        &mut write_u64,
                    );
                    write = write_u64 as usize;
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    write += field_not_started | dfa.is_output(state) as usize;
                }

                field_start += field_not_started;

                read += 1;

                if dfa.is_field_end(state) {
                    // SAFETY: field < fields.end, checked after every write
                    unsafe {
                        field.write([field_start - record_start, write - record_start]);
                        field = field.add(1);
                    }
                    field_not_started = 1;
                    field_start = read;
                    write = read;

                    if dfa.is_record_end(state) {
                        // an error is also a record end
                        if dfa.is_error(state) {
                            // report the byte where the error occured which was not consumed and
                            // do not count the field that was written for it
                            read -= 1;
                            // SAFETY: a field was written above
                            field = unsafe { field.sub(1) };
                            break Exit::Error;
                        }
                        break Exit::EndRecord;
                    }

                    if field == fields.end {
                        break Exit::FieldEndsFull;
                    }
                }
            };

            self.read_index = read;
            // SAFETY: field is in the same allocation as fields.start and not before it
            field_count = unsafe { field.offset_from(fields.start) } as usize;

            match exit {
                Exit::Refill => {}
                Exit::FieldEndsFull => {
                    let len = self.fields.len() * 2;
                    self.fields.resize(len, [0; 2]);
                }
                Exit::EndRecord => break false,
                Exit::Error => break true,
            }
        };

        self.position.byte = self.buffer.source_offset(self.read_index);

        if is_error {
            self.error = Some(Stopped::Parser);
            self.position.field = field_count;
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        Ok(Some(Record {
            buffer: &self.buffer.data()[record_start..self.read_index],
            fields: &self.fields[..field_count],
        }))
    }

    /// Parse a record like [`read_record`](Self::read_record) but skip chunks of input without any
    /// structural bytes using the classifier.
    #[inline]
    pub fn read_record_simd<C: Classifier, M: Sets>(
        &mut self,
        classifier: &C,
        _sets: M,
    ) -> Result<Option<Record<'_>>, Error> {
        // a bitmask of the vector fits a u64, and the vectors are aligned
        const { assert!(C::LEN <= 64 && C::LEN.is_power_of_two()) };

        if let Some(stopped) = self.error {
            return Err(stopped.error(self.position));
        }

        let mut record_start = self.read_index;
        let mut write = self.read_index;
        let mut field_count = 0;

        // Fields are parsed using three loops. First, one loop finds the start of the field (the
        // first byte that is not discarded). Second, a loop goes through the field (using simd)
        // without copying data until it finds the first character that needs to be discarded
        // (usually the end of the field). Third, if its not the end of the field, this loops
        // processes the field while copying the field data to remove any discarded bytes.
        // The loop to resume after a refill.
        let mut phase = Phase::Start;

        // A record always starts in the StartRecord (equivalent to EndRecord) state so there is no
        // need to store the DFA state between records
        let mut state = self.dfa.start_state();

        let is_error = loop {
            if self.read_index == self.buffer.len() {
                let Refill { shift, read } = self
                    .buffer
                    .fill(&mut self.reader, record_start, self.read_index, 0)
                    .map_err(|e| {
                        self.error = Some(Stopped::IO);
                        Error::IO(e)
                    })?;
                if shift > 0 {
                    // shift the field start if its already set
                    if phase != Phase::Start {
                        self.fields[field_count][0] -= shift;
                    }
                    record_start -= shift;
                    self.read_index -= shift;
                    write = write.wrapping_sub(shift);
                    for [start, end] in &mut self.fields[..field_count] {
                        *start -= shift;
                        *end -= shift;
                    }
                }

                if read == 0 {
                    if (self.dfa.in_quoted_field(state)
                        && matches!(self.config.quote_style, QuoteStyle::Strict(_)))
                        || self.dfa.in_escape_sequence(state)
                    {
                        break true;
                    }

                    let (start, end) = match phase {
                        Phase::Start => {
                            if field_count == 0 {
                                return Ok(None);
                            }
                            (self.read_index, self.read_index)
                        }
                        Phase::InPlace => (self.fields[field_count][0], self.read_index),
                        Phase::Copy => (self.fields[field_count][0], write),
                    };

                    self.fields[field_count] = [start, end];
                    field_count += 1;
                    break false;
                }
            }

            let bytes_read = self.buffer.bytes_read();
            let mut block = self.structural_block;
            if block.bytes_read != bytes_read {
                block = StructuralBlock {
                    bytes_read,
                    ..StructuralBlock::NONE
                };
            }

            // keep variables used in the hotloop in registers
            let dfa = &self.dfa;
            let ordinary = self.ordinary;
            let buffer = self.buffer.data_mut();
            let mut read = self.read_index;
            // A cursor and an end instead of the slice, its length and the count: fewer registers
            // in the core loop and no bounds check before the write
            let fields = self.fields.as_mut_ptr_range();
            // SAFETY: field_count < fields.len() at every entry of the core loop
            let mut field = unsafe { fields.start.add(field_count) };
            // the start and end of the current SIMD chunk
            let mut base = 0;
            let mut end = 0;

            let exit = 'core: {
                /// transition the DFA with the current input byte
                macro_rules! step {
                    () => {{
                        // SAFETY: every use is guarded by read < end <= buffer.len()
                        let byte = unsafe { *buffer.get_unchecked(read) };
                        state = dfa.transition(state, byte);
                        read += 1;
                        byte
                    }};
                }

                /// handle a field end: check for an error, save field end position, check for
                /// record end and if the fields array needs to be resized
                macro_rules! record_field {
                    ($end:expr) => {
                        // SAFETY: field < fields.end, checked after every write
                        unsafe {
                            (*field)[1] = $end;
                            field = field.add(1);
                        }

                        if dfa.is_record_end(state) {
                            // an error is also a record end
                            if dfa.is_error(state) {
                                // report the byte where the error occured which was not consumed
                                // and do not count the field that was written for it
                                read -= 1;
                                // SAFETY: a field was written above
                                field = unsafe { field.sub(1) };
                                break 'core Exit::Error;
                            }
                            break 'core Exit::EndRecord;
                        }

                        if field == fields.end {
                            phase = Phase::Start;
                            break 'core Exit::FieldEndsFull;
                        }
                    };
                }

                /// classify the block of read or refill (afterwards returning to `$phase`)
                macro_rules! next_block {
                    ($phase:expr) => {
                        if read >= buffer.len() {
                            phase = $phase;
                            break 'core Exit::Refill;
                        }
                        base = read & !(C::LEN - 1);
                        if base + C::LEN > buffer.len() {
                            // disable skiping by setting the mask to all ones so the whole
                            // scalar zerocopy logic does not have to be duplicated here
                            block.base = base;
                            block.in_field = u64::MAX;
                            block.in_quoted_field = u64::MAX;
                            end = buffer.len();
                        } else {
                            if base != block.base {
                                let v = C::load(&buffer[base..base + C::LEN]);
                                block.base = base;
                                if M::PER_STATE {
                                    let [in_field, in_quoted_field] = classifier.per_state(v);
                                    block.in_field = in_field.to_bitmask();
                                    block.in_quoted_field = in_quoted_field.to_bitmask();
                                } else {
                                    block.in_field = classifier.structural(v).to_bitmask();
                                }
                            }
                            end = base + C::LEN;
                        }
                    };
                }

                /// Like in the compaction parser, have a fast path for when a field end is
                /// immediatly followed by the start of an unquoted field (an ordinary byte) so only
                /// the DFA transition has to be performed and the loop to find the field start can
                /// be skipped. `$in_place` enters the in-place loop.
                macro_rules! field_start {
                    ($in_place:block) => {
                        let structural = if M::PER_STATE {
                            block.in_field | block.in_quoted_field
                        } else {
                            block.in_field
                        };
                        if read < end && (structural >> (read - base)) & 1 == 0 {
                            state = dfa.transition(state, ordinary);
                            debug_assert!(dfa.is_output(state) && !dfa.in_quoted_field(state));
                            // SAFETY: field < fields.end
                            unsafe { (*field)[0] = read };
                            $in_place
                        }
                    };
                }

                /// the mask for the current (output) state
                macro_rules! mask {
                    () => {
                        if M::PER_STATE && dfa.in_quoted_field(state) {
                            // hint to LLVM to not turn this into a cmov
                            // (see CompactionParser::read_record_simd)
                            std::hint::cold_path();
                            block.in_quoted_field
                        } else {
                            block.in_field
                        }
                    };
                }

                'field: loop {
                    'copy: {
                        'in_place: {
                            // only after a refill in the middle of a field
                            if phase != Phase::Start {
                                let resume = phase;
                                phase = Phase::Start;
                                if resume == Phase::Copy {
                                    break 'copy;
                                }
                                break 'in_place;
                            }

                            if dfa.is_field_end(state) {
                                field_start!({
                                    break 'in_place;
                                });
                            }

                            // start: find the first output byte
                            // having it in a separate loop makes the in place loop which handles
                            // most of the bytes faster
                            loop {
                                if read >= buffer.len() {
                                    phase = Phase::Start;
                                    break 'core Exit::Refill;
                                }
                                step!();

                                if dfa.is_output(state) {
                                    // SAFETY: field < fields.end
                                    unsafe { (*field)[0] = read - 1 };
                                    break 'in_place;
                                }

                                // an empty field
                                if dfa.is_field_end(state) {
                                    // SAFETY: field < fields.end
                                    unsafe { (*field)[0] = read - 1 };
                                    record_field!(read - 1);
                                    continue 'field;
                                }
                            }
                        }

                        // in place: no discarded bytes in the field so far (no copy needed)
                        loop {
                            if read >= end {
                                next_block!(Phase::InPlace);
                            }

                            // Skip to the next structural character using the classifier
                            let rest = mask!() >> (read - base);
                            if rest == 0 {
                                read = end;
                                continue;
                            }
                            read += rest.trailing_zeros() as usize;
                            step!();

                            // still no discarded bytes so stay in the in_place loop
                            if dfa.is_output(state) {
                                continue;
                            }

                            if dfa.is_field_end(state) {
                                record_field!(read - 1);

                                // stay in this loop for the fast path of an unquoted field
                                // immediately after the current field
                                field_start!({
                                    continue;
                                });
                                continue 'field;
                            }

                            // discarded byte inside the field so we need to copy the input from
                            // now on
                            write = read - 1;
                            break 'copy;
                        }
                    }

                    // copy: the field contains discarded bytes (write < read)
                    loop {
                        if read >= end {
                            next_block!(Phase::Copy);
                        }

                        if dfa.is_output(state) {
                            let rest = mask!() >> (read - base);
                            // duplicated code for the move to avoid additional latency
                            // (see CompactionParser::read_record_simd)
                            if rest == 0 {
                                let len = end - read;
                                // SAFETY: write < read, 0 < len <= C::LEN <= 64, read + len <= end
                                unsafe { move_left(buffer, read, write, len) };
                                read = end;
                                write += len;
                                continue;
                            }
                            let len = rest.trailing_zeros() as usize;
                            if len > 0 {
                                // SAFETY: write < read, len <= C::LEN <= 64 and read + len <= end
                                unsafe { move_left(buffer, read, write, len) };
                                read += len;
                                write += len;
                            }
                        }

                        let byte = step!();

                        if dfa.is_field_end(state) {
                            record_field!(write);
                            continue 'field;
                        }

                        // SAFETY: write < read <= buffer.len()
                        unsafe {
                            *buffer.get_unchecked_mut(write) = byte;
                        }
                        write += dfa.is_output(state) as usize;
                    }
                }
            };

            self.read_index = read;
            self.structural_block = block;
            // SAFETY: field is a pointer derived from fields
            field_count = unsafe { field.offset_from(fields.start) } as usize;

            match exit {
                Exit::Refill => {}
                Exit::FieldEndsFull => {
                    let len = self.fields.len() * 2;
                    self.fields.resize(len, [0; 2]);
                }
                Exit::EndRecord => break false,
                Exit::Error => break true,
            }
        };

        self.position.byte = self.buffer.source_offset(self.read_index);

        if is_error {
            self.error = Some(Stopped::Parser);
            self.position.field = field_count;
            return Err(Error::Parser(self.position));
        }

        self.position.record += 1;

        Ok(Some(Record {
            // the field positions are buffer indices
            buffer: &self.buffer.data()[..self.read_index],
            fields: &self.fields[..field_count],
        }))
    }
}

/// In which phase the [`ZeroCopyParser::read_record_simd`] function is within the current field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// field has not started (no output bytes yet)
    Start,
    /// Inside a field with no discarded bytes so far, the output is in place
    InPlace,
    /// Inside a field after a discarded byte (write < read)
    Copy,
}

/// Move `buffer[read..read + len]` to `write`, for `0 < len <= 64` and `write < read`.
///
/// Copies the data in two chunks (as large as possible depending on `len`) that may overlap.
///
/// # Safety
///
/// `read + len <= buffer.len()`, `write < read` and `0 < len <= 64`.
#[inline(always)]
unsafe fn move_left(buffer: &mut [u8], read: usize, write: usize, len: usize) {
    debug_assert!(write < read && 0 < len && len <= 64 && read + len <= buffer.len());
    let data = buffer.as_mut_ptr();
    macro_rules! chunks {
        ($n:literal) => {{
            // SAFETY: both chunks lie in buffer[read..read + len] and buffer[write..write + len]
            unsafe {
                let first = data.add(read).cast::<[u8; $n]>().read_unaligned();
                let last = data
                    .add(read + len - $n)
                    .cast::<[u8; $n]>()
                    .read_unaligned();
                data.add(write).cast::<[u8; $n]>().write_unaligned(first);
                data.add(write + len - $n)
                    .cast::<[u8; $n]>()
                    .write_unaligned(last);
            }
        }};
    }
    if len >= 32 {
        chunks!(32);
    } else if len >= 16 {
        chunks!(16);
    } else if len >= 8 {
        chunks!(8);
    } else if len >= 4 {
        chunks!(4);
    } else if len >= 2 {
        chunks!(2);
    } else {
        chunks!(1);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    buffer: &'a [u8],
    /// `[start, end]` of each field in `buffer`
    fields: &'a [[usize; 2]],
}

impl<'a> Record<'a> {
    /// Number of fields in the record.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// The raw bytes of field `index`, or `None` if out of range.
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        let [start, end] = *self.fields.get(index)?;
        Some(&self.buffer[start..end])
    }

    /// Iterate over the fields as byte slices.
    pub fn iter(&self) -> Fields<'a> {
        Fields {
            buffer: self.buffer,
            fields: self.fields.iter(),
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
    fields: std::slice::Iter<'a, [usize; 2]>,
}

impl<'a> Iterator for Fields<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        let [start, end] = *self.fields.next()?;
        Some(&self.buffer[start..end])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.fields.size_hint()
    }
}

impl ExactSizeIterator for Fields<'_> {}

#[cfg(test)]
mod tests {
    use super::move_left;

    #[test]
    fn move_left_matches_copy_within() {
        let input: Vec<u8> = (0..200).map(|i| i as u8).collect();
        for len in 1..=64 {
            for distance in 1..=70 {
                let read = 80;
                let mut expected = input.clone();
                expected.copy_within(read..read + len, read - distance);
                let mut actual = input.clone();
                // SAFETY: read + len <= 144 < 200, write < read, 0 < len <= 64
                unsafe { move_left(&mut actual, read, read - distance, len) };
                assert_eq!(actual, expected, "len {len}, distance {distance}");
            }
        }
    }
}
