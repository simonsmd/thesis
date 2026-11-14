use std::{io::Read, str::Utf8Error};

use fearless_simd::{SimdBase, SimdMask};

use super::{Error, Exit, Position, Sets};
use crate::{
    buffer::{Buffer, DEFAULT_CAPACITY, Refill},
    classifier::{self, Classifier},
    config::{Config, QuoteStyle},
    dfa::Dfa,
};

/// Initial size of the field end buffer: `read_record_branchless` writes a field end for every
/// buffered byte, so it needs one entry per byte of a default size buffer.
pub const DEFAULT_FIELDS: usize = DEFAULT_CAPACITY;

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
    /// The masks of the last block `read_record_simd` classified, so a record starting in
    /// the same block does not classify it again
    structural_block: StructuralBlock,
    /// [`classifier::ordinary_byte`] of the config
    ordinary: u8,
}

/// The structural masks of the vector at `base`, valid while the buffer has read
/// `bytes_read` bytes.
#[derive(Debug, Clone, Copy)]
pub(super) struct StructuralBlock {
    pub(super) bytes_read: usize,
    pub(super) base: usize,
    /// With [`Shared`](super::Shared) all structural bytes, with [`PerState`](super::PerState)
    /// those of an unquoted field.
    pub(super) in_field: u64,
    /// Those of a quoted field, only set and read with [`PerState`](super::PerState).
    pub(super) in_quoted_field: u64,
}

impl StructuralBlock {
    pub(super) const NONE: Self = Self {
        bytes_read: usize::MAX,
        base: usize::MAX,
        in_field: 0,
        in_quoted_field: 0,
    };
}

impl<R: Read, D: Dfa> CompactionParser<R, D> {
    pub fn new(reader: R) -> Self {
        Self::with_config(reader, Config::default())
    }

    pub fn with_config(reader: R, config: Config) -> Self {
        Self::with_capacity(reader, config, DEFAULT_CAPACITY, DEFAULT_FIELDS)
    }

    /// `capacity` is the read size; the buffer only grows past it for records that do not fit.
    /// `fields` is the initial size of the field end buffer, which grows on demand. Small sizes of
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
            field_ends: vec![0; fields.max(1)],
            position: Position::default(),
            error: false,
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
        self.error = false;
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

/// Minimum gap between write and read index at the start of a record and after a refill. This was
/// originally introduced to avoid unnecessary memory ordering machine clears for the standard
/// read_record. read_record_simd always copies whole classifier vectors (up to 64 bytes) so it
/// needs at least that gap to not overwrite unread input. read_record_simd_check writes up to 64 byte blocks
/// so use a gap of 128 to be safe for memory ordering machine clears in that instance.
const MIN_GAP: usize = 128;

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
            let mut read = self.read_index;
            // A cursor and an end instead of the slice, its length and the count: one register
            // less in the core loop and no bounds check before the write
            let fields = self.field_ends.as_mut_ptr_range();
            // SAFETY: field_count < field_ends.len() at every entry of the core loop
            let mut field_end = unsafe { fields.start.add(field_count) };

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

                    // SAFETY: field_end < fields.end, checked after every write
                    unsafe {
                        field_end.write(write - record_start);
                        field_end = field_end.add(1);
                    }

                    if dfa.is_record_end(state) {
                        break Exit::EndRecord;
                    }

                    if field_end == fields.end {
                        break Exit::FieldEndsFull;
                    }
                }
            };

            self.read_index = read;
            // SAFETY: field_end is in the same allocation as fields.start and not before it
            field_count = unsafe { field_end.offset_from(fields.start) } as usize;

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

    /// Parse a record like [`read_record`](Self::read_record) but without branching for every
    /// field.
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
                    self.field_ends.resize(self.buffer.capacity(), 0);
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

    /// Parse a record like [`read_record`](Self::read_record) but skip chunks of input without any
    /// structural bytes using the classifier.
    #[inline]
    pub fn read_record_simd_check<C: Classifier, M: Sets>(
        &mut self,
        classifier: &C,
        _sets: M,
    ) -> Result<Option<Record<'_>>, Error> {
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
            let mut read = self.read_index;
            // Use a start and end pointer instaed of a slice with a count which saves one
            // register (the slice length is not needed).
            let fields = self.field_ends.as_mut_ptr_range();
            // SAFETY: field_count < field_ends.len() at every entry of the core loop
            let mut field_end = unsafe { fields.start.add(field_count) };

            let exit = 'core: loop {
                if read >= buffer.len() {
                    break Exit::Refill;
                }

                // In an output state, identify input chunks that do not contain any structural
                // bytes using the classifier since the DFA stays in the same state and all bytes
                // are copied to the output.
                if dfa.is_output(state)
                    && let Some(bytes) = buffer.get(read..read + C::LEN)
                {
                    let v = C::load(bytes);
                    let structural = if M::PER_STATE {
                        // Only run the DFA if the input contains a byte that can change the state
                        // in the current state (either quoted or normal field).
                        if dfa.in_quoted_field(state) {
                            classifier.in_quoted_field(v).any_true()
                        } else {
                            classifier.in_field(v).any_true()
                        }
                    } else {
                        // use the same set of structural bytes for quoted and unquoted fields.
                        // Results in more false positives.
                        classifier.structural(v).any_true()
                    };
                    if !structural {
                        v.store_slice(&mut buffer[write..write + C::LEN]);
                        read += C::LEN;
                        write += C::LEN;
                        continue;
                    }
                }

                // Otherwise run the DFA over the whole chunk before checking again for the next
                // chunk
                let end = (read + C::LEN).min(buffer.len());
                while read < end {
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
                            break 'core Exit::Error;
                        }

                        // SAFETY: field_end < fields.end, checked after every write
                        unsafe {
                            field_end.write(write - record_start);
                            field_end = field_end.add(1);
                        }

                        if dfa.is_record_end(state) {
                            break 'core Exit::EndRecord;
                        }

                        if field_end == fields.end {
                            break 'core Exit::FieldEndsFull;
                        }
                    }
                }
            };

            self.read_index = read;
            // SAFETY: field_end is in the same allocation as fields.start and not before it
            field_count = unsafe { field_end.offset_from(fields.start) } as usize;

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

    /// Parse a record like [`read_record`](Self::read_record) but skip to the next structural byte
    /// within fields using the classifier instead of running the DFA for every byte.
    ///
    /// The input is classified in aligned blocks of one vector of `C` (`C::LEN` bytes) into one mask
    /// per output state.
    /// In an output state the parser copies everything up to the next set bit of the mask for the
    /// current state and continues there, otherwise it runs the DFA on the next byte (the byte
    /// after a delimiter, a closing quote, an escape or a comment line).
    ///
    /// The next load address does not depend on a mask: blocks follow each other at a fixed
    /// stride, and the jump to the next structural byte is a shift and a `tzcnt` of a mask that is
    /// already computed. A block is classified once, however many fields it holds.
    #[inline]
    pub fn read_record_simd<C: Classifier, M: Sets>(
        &mut self,
        classifier: &C,
        _sets: M,
    ) -> Result<Option<Record<'_>>, Error> {
        const {
            // the block bitmasks are u64, and `read & !(C::LEN - 1)` aligns the blocks
            assert!(C::LEN <= 64 && C::LEN.is_power_of_two());
            // copy_block needs write + C::LEN <= read, which the gap at a record start guarantees
            assert!(MIN_GAP >= C::LEN);
        };

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
            // Use a start and end pointer instaed of a slice with a count which saves one
            // register (the slice length is not needed).
            let fields = self.field_ends.as_mut_ptr_range();
            // SAFETY: field_count < field_ends.len() at every entry of the core loop
            let mut field_end = unsafe { fields.start.add(field_count) };

            macro_rules! step {
                ($core:lifetime) => {
                    step!($core, {})
                };
                // `$field_start` runs after a field end inside the record
                ($core:lifetime, $field_start:block) => {
                    // SAFETY: every use is guarded by read < end <= buffer.len()
                    let byte = unsafe { *buffer.get_unchecked(read) };
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
                            break $core Exit::Error;
                        }

                        // SAFETY: field_end < fields.end, checked after every write
                        unsafe {
                            field_end.write(write - record_start);
                            field_end = field_end.add(1);
                        }

                        if dfa.is_record_end(state) {
                            break $core Exit::EndRecord;
                        }

                        if field_end == fields.end {
                            break $core Exit::FieldEndsFull;
                        }

                        $field_start
                    }
                };
            }

            // The block of the read index. Advanced by a constant stride and not computed from
            // read, so the next block can be loaded and classified before the skips and DFA steps
            // of the current one are done.
            let mut base = read & !(C::LEN - 1);

            let exit = 'core: loop {
                // Two times the classifier chunk length needs to be left in the buffer since the
                // classifier reads one chunk and since always a full block is copied it might read
                // up to one additional chunk when copying at the end of a block
                if base + 2 * C::LEN > buffer.len() {
                    while read < buffer.len() {
                        step!('core);
                    }
                    break Exit::Refill;
                }

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

                let end = base + C::LEN;

                while read < end {
                    // Skip to the next structural byte in an output state
                    if dfa.is_output(state) {
                        let mask = if M::PER_STATE && dfa.in_quoted_field(state) {
                            // hint to LLVM to not turn this into a cmov which would make the mask
                            // and thus the read index depend on the DFA state transition so the
                            // transition and skip to the next structural byte can not run in
                            // parallel
                            std::hint::cold_path();
                            block.in_quoted_field
                        } else {
                            block.in_field
                        };
                        let rest = mask >> (read - base);

                        // skip the rest of the chunk if it does not contain any structural bytes.
                        // The code is duplicated so that LLVM does not use a cmov to calculate the
                        // number of skipped bytes which would introduce additional latency since
                        // this calculation depends on the read index.
                        if rest == 0 {
                            let skipped = end - read;
                            // SAFETY: read < end and end + C::LEN <= buffer.len()
                            unsafe { copy_block::<C>(buffer, read, write) };
                            read = end;
                            write += skipped;
                            break;
                        }

                        let skipped = rest.trailing_zeros() as usize;
                        // always copy a whole block which is faster than copying the exact number
                        // of bytes that are skipped. The additional data is overwritten or ignored
                        // anyways.
                        // SAFETY: read < end and end + C::LEN <= buffer.len()
                        unsafe { copy_block::<C>(buffer, read, write) };
                        read += skipped;
                        write += skipped;
                    }

                    // As a small optimization the DFA step for going from the EndField to the
                    // InField state when a field end is followed immediatly by an unquoted field
                    // can be skipped because there are no quotes or other structural bytes to
                    // remove so the copy can be handled by the skip loop.
                    step!('core, {
                        let structural = if M::PER_STATE {
                            block.in_field | block.in_quoted_field
                        } else {
                            block.in_field
                        };
                        if read < end && (structural >> (read - base)) & 1 == 0 {
                            state = dfa.transition(state, ordinary);
                            debug_assert!(dfa.is_output(state) && !dfa.in_quoted_field(state));
                        }
                    });
                }

                // a skip ends at the end of the block and a DFA step advances by one byte, so the
                // loop leaves the block exactly at its end
                debug_assert_eq!(read, end);
                base = end;
            };

            self.read_index = read;
            self.structural_block = block;
            // SAFETY: field_end is in the same allocation as fields.start and not before it
            field_count = unsafe { field_end.offset_from(fields.start) } as usize;

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
}

/// Copy the [`C::LEN`](Classifier::LEN) bytes at `read` to `write`.
///
/// # Safety
///
/// `read + C::LEN` must not exceed `buffer.len()`, and `write + C::LEN <= read`.
#[inline(always)]
unsafe fn copy_block<C: Classifier>(buffer: &mut [u8], read: usize, write: usize) {
    debug_assert!(write + C::LEN <= read);
    debug_assert!(read + C::LEN <= buffer.len());
    let data = buffer.as_mut_ptr();
    // SAFETY: in bounds by the contract, and the ranges do not overlap since
    // write + C::LEN <= read
    unsafe { std::ptr::copy_nonoverlapping(data.add(read), data.add(write), C::LEN) };
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
