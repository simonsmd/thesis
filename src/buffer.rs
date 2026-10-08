use std::io::{self, Read};

/// Initial buffer size, and the read size [`Buffer::fill`] aims for.
pub const DEFAULT_CAPACITY: usize = 64 * 1024;

/// What a [`Buffer::fill`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refill {
    /// How far the kept bytes moved towards the front of the buffer. Every
    /// index into the buffer that is still live has to be decreased by this.
    pub shift: usize,
    /// Number of bytes read from the source. `0` means end of input.
    pub read: usize,
}

/// The parser's input and output buffer.
///
/// The buffer only owns the allocation and the bytes read into it; where the
/// parser is reading and writing inside it is none of its business. It hands
/// out the valid bytes as a mutable slice and refills them on request.
///
/// ```text
///        keep_from            write                len()          capacity()
///            |                  |                    |                 |
///  data: [ ..| bytes to keep ...| free to overwrite ..| uninitialized ..]
/// ```
///
/// [`fill`](Buffer::fill) takes both of those positions: bytes before
/// `keep_from` are dropped, bytes in `keep_from..write` are preserved (this is
/// the record being assembled), and everything from `write` on is scratch
/// space the next read may overwrite — normally input the parser has already
/// consumed. If that space is too small the kept bytes are moved to the front
/// of the buffer, and if they alone fill it the allocation grows, so a record
/// can always be completed in one contiguous slice.
#[derive(Debug)]
pub struct Buffer {
    data: Vec<u8>,
    /// Target size of a read. A `fill` reads into all the free space it has,
    /// and makes room for at least half of this much.
    chunk: usize,
    /// `data[..filled]` holds valid bytes.
    filled: usize,
    /// Total number of bytes pulled from the source so far.
    read_total: usize,
}

impl Default for Buffer {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl Buffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `capacity` is both the initial allocation and the read size; the buffer
    /// only grows beyond it for records that do not fit.
    pub fn with_capacity(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            data: vec![0; capacity],
            chunk: capacity,
            filled: 0,
            read_total: 0,
        }
    }

    /// Current size of the allocation.
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Number of valid bytes.
    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// The valid bytes.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.data[..self.filled]
    }

    /// The valid bytes, writable so the parser can compact fields in place.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.filled]
    }

    /// Read the next chunk from `source`.
    ///
    /// Keeps `data[keep_from..write]`, and reads the new input `gap` bytes
    /// after `write`, over everything from there on. The caller must have
    /// consumed the input up to [`len`](Buffer::len) before calling this,
    /// since that is the space being read into. The `gap` bytes in between
    /// are left as they are: they are scratch space for the caller.
    ///
    /// The kept bytes may move; see [`Refill::shift`]. On return the new input
    /// starts at `write - shift + gap` and the valid bytes end `read` bytes
    /// later.
    pub fn fill<R: Read>(
        &mut self,
        source: &mut R,
        keep_from: usize,
        write: usize,
        gap: usize,
    ) -> io::Result<Refill> {
        debug_assert!(keep_from <= write && write <= self.filled);

        // Reading into a sliver of free space would mean a syscall per few
        // bytes, so make room for at least half a chunk.
        let min_free = (self.chunk / 2).max(1);

        let mut shift = 0;
        if self.data.len() - write < gap + min_free {
            // Drop everything before the record being assembled...
            if keep_from > 0 {
                self.data.copy_within(keep_from..write, 0);
                shift = keep_from;
            }
            // ...and grow if that record alone leaves no room to read into.
            if self.data.len() - (write - shift) < gap + min_free {
                let len = (self.data.len() * 2).max(write - shift + gap + self.chunk);
                self.data.resize(len, 0);
            }
        }

        let write = write - shift + gap;
        self.filled = write;

        let read = loop {
            match source.read(&mut self.data[write..]) {
                Ok(read) => break read,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };

        self.filled = write + read;
        self.read_total += read;
        Ok(Refill { shift, read })
    }

    /// Total number of bytes read from the source.
    pub fn bytes_read(&self) -> usize {
        self.read_total
    }

    /// Offset in the source of the byte at `index`.
    ///
    /// Only meaningful for input that has not been overwritten by the parser,
    /// i.e. positions at or after its write cursor.
    pub fn source_offset(&self, index: usize) -> usize {
        self.read_total - (self.filled - index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source that hands out at most `step` bytes per read, so refills
    /// happen mid-record.
    struct Trickle<'a> {
        data: &'a [u8],
        step: usize,
    }

    impl Read for Trickle<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = self.data.len().min(self.step).min(out.len());
            out[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    /// A stand-in for `read_record`: copies the input into records of
    /// `record_len` output bytes, dropping every `skip`th input byte, and
    /// refills `gap` bytes after the write position.
    fn parse(
        buffer: &mut Buffer,
        source: &mut impl Read,
        record_len: usize,
        skip: usize,
        gap: usize,
    ) -> Vec<Vec<u8>> {
        let (mut record_start, mut write, mut pos) = (0, 0, 0);
        let mut records = Vec::new();
        let mut seen = 0;

        loop {
            if pos == buffer.len() {
                let refill = buffer.fill(source, record_start, write, gap).unwrap();
                record_start -= refill.shift;
                write -= refill.shift;
                pos = write + gap;
                if refill.read == 0 {
                    break;
                }
            }

            assert_eq!(buffer.source_offset(pos), seen);
            let data = buffer.data_mut();
            data[write] = data[pos];
            if skip == 0 || seen % skip != 0 {
                write += 1;
            }
            pos += 1;
            seen += 1;

            if write - record_start == record_len {
                records.push(buffer.data()[record_start..write].to_vec());
                record_start = write;
            }
        }

        if write > record_start {
            records.push(buffer.data()[record_start..write].to_vec());
        }
        records
    }

    fn source_bytes(len: usize) -> Vec<u8> {
        (0..len as u32).map(|i| i as u8).collect()
    }

    #[test]
    fn a_record_stays_contiguous_across_refills() {
        let data = source_bytes(200);
        let mut buffer = Buffer::with_capacity(16);
        let mut source = Trickle { data: &data, step: 7 };

        let records = parse(&mut buffer, &mut source, 40, 0, 0);
        assert_eq!(records.concat(), data);
        assert!(records.iter().all(|r| r.len() == 40));
    }

    #[test]
    fn the_buffer_grows_for_a_record_larger_than_its_capacity() {
        let data = source_bytes(1000);
        let mut buffer = Buffer::with_capacity(16);
        let mut source = Trickle { data: &data, step: 5 };

        let records = parse(&mut buffer, &mut source, 1000, 0, 0);
        assert_eq!(records, vec![data]);
        assert!(buffer.capacity() >= 1000);
    }

    #[test]
    fn finished_records_are_reclaimed_without_growing() {
        // Small records through a small buffer must not grow it: each record
        // is released as soon as the next one starts.
        let data = source_bytes(4096);
        let mut buffer = Buffer::with_capacity(64);
        let mut source = Trickle { data: &data, step: 13 };

        let records = parse(&mut buffer, &mut source, 8, 0, 0);
        assert_eq!(records.concat(), data);
        assert_eq!(buffer.capacity(), 64);
    }

    #[test]
    fn discarded_bytes_free_up_space() {
        let data = source_bytes(300);
        let expected: Vec<u8> = data
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 != 0)
            .map(|(_, b)| *b)
            .collect();

        let mut buffer = Buffer::with_capacity(16);
        let mut source = Trickle { data: &data, step: 7 };

        let records = parse(&mut buffer, &mut source, 25, 3, 0);
        assert_eq!(records.concat(), expected);
    }

    #[test]
    fn a_gap_is_left_between_the_kept_bytes_and_the_new_input() {
        let data = source_bytes(1000);
        let expected: Vec<u8> = data
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 != 0)
            .map(|(_, b)| *b)
            .collect();

        // small and large records, so refills both reclaim space and grow
        for record_len in [5, 400] {
            let mut buffer = Buffer::with_capacity(16);
            let mut source = Trickle { data: &data, step: 7 };

            let records = parse(&mut buffer, &mut source, record_len, 3, 10);
            assert_eq!(records.concat(), expected);
        }
    }

    #[test]
    fn fill_leaves_the_gap_before_the_new_input() {
        let mut buffer = Buffer::with_capacity(8);
        let mut source = Trickle { data: b"abcdef", step: 3 };

        assert_eq!(buffer.fill(&mut source, 0, 0, 4).unwrap(), Refill { shift: 0, read: 3 });
        assert_eq!(&buffer.data()[4..], b"abc");
        assert_eq!(buffer.source_offset(4), 0);
        assert_eq!(buffer.fill(&mut source, 0, 2, 4).unwrap(), Refill { shift: 0, read: 3 });
        assert_eq!(&buffer.data()[6..], b"def");
        assert_eq!(buffer.source_offset(6), 3);
    }

    #[test]
    fn fill_reports_end_of_input() {
        let mut buffer = Buffer::with_capacity(8);
        let mut source = Trickle { data: b"ab", step: 8 };

        assert_eq!(buffer.fill(&mut source, 0, 0, 0).unwrap(), Refill { shift: 0, read: 2 });
        assert_eq!(buffer.data(), b"ab");
        assert_eq!(buffer.fill(&mut source, 0, 2, 0).unwrap(), Refill { shift: 0, read: 0 });
        assert_eq!(buffer.data(), b"ab");
        assert_eq!(buffer.bytes_read(), 2);
    }
}
