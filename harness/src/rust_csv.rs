//! rust-csv as comparison parsers.
//!
//! `csv-core` is driven through the same [`Buffer`] as this crate's parsers: the input is copied in
//! chunks of the same read size, so the comparison measures the state machines and record assembly
//! rather than differences in buffering. `csv/byte-record` is the full `csv::Reader` with its own
//! buffering, as an end-to-end reference point.
//!
//! Both strip a leading UTF-8 BOM, which this crate's parsers do not; the harness strips it for
//! every parser, see [`without_bom`](crate::parsers::without_bom).
//!
//! rust-csv only knows strict quoting where a quote inside an unquoted field is literal, and it
//! never reports errors: data after a closing quote is appended to the field. Where this crate's
//! parsers stop with an error the two disagree, which the correctness check reports.

use std::io::{Cursor, SeekFrom};

use csv_core::{ReadRecordResult, Reader, ReaderBuilder, Terminator as CoreTerminator};
use csv_parser::buffer::Buffer;

use crate::{
    check::Digest,
    dialect::{Dialect, QuoteStyle, Terminator},
    parsers::{
        Consumer, ErrorAt, Errors, Options, OwnedRecord, Parser, Prepared, Records, Statistics,
    },
};

pub fn parsers() -> [Parser; 2] {
    [
        Parser {
            name: "csv-core/dfa",
            supports,
            errors,
            prepare: prepare_core,
            records: core_records,
        },
        Parser {
            name: "csv/byte-record",
            supports,
            errors,
            prepare: prepare_reader,
            records: reader_records,
        },
    ]
}

fn supports(d: &Dialect, _: &[u8]) -> Result<(), &'static str> {
    if d.quote.is_some() && d.quote_style == Some(QuoteStyle::Toggle) {
        return Err("toggle quoting");
    }
    if d.escape.is_some() && d.escape_in_unquoted_fields {
        return Err("escapes in unquoted fields");
    }
    Ok(())
}

fn errors(_: &Dialect) -> Errors {
    Errors::Never
}

fn terminator(d: &Dialect) -> CoreTerminator {
    match d.terminator {
        Terminator::Crlf => CoreTerminator::CRLF,
        Terminator::Byte(b) => CoreTerminator::Any(b.0),
    }
}

fn core_reader(d: &Dialect) -> Reader {
    ReaderBuilder::new()
        .delimiter(d.delimiter.0)
        .terminator(terminator(d))
        .quote(d.quote.map_or(b'"', |q| q.0))
        .quoting(d.quote.is_some())
        .escape(d.escape.map(|e| e.0))
        .double_quote(d.double_quote_escape)
        .comment(d.comment.map(|c| c.0))
        .build()
}

/// csv-core fed from a [`Buffer`], one record at a time.
struct CoreStream<'a> {
    reader: Reader,
    source: &'a [u8],
    buffer: Buffer,
    /// Next unread byte in the buffer.
    pos: usize,
    eof: bool,
    output: Vec<u8>,
    ends: Vec<usize>,
}

impl<'a> CoreStream<'a> {
    fn new(d: &Dialect, input: &'a [u8], o: &Options) -> Self {
        Self {
            reader: core_reader(d),
            source: input,
            buffer: Buffer::with_capacity(o.capacity),
            pos: 0,
            eof: false,
            output: vec![0; 1024],
            ends: vec![0; o.fields.unwrap_or(256).max(1)],
        }
    }

    /// The next record as its unescaped bytes and the field ends within them.
    #[inline(always)]
    fn next(&mut self) -> Option<(&[u8], &[usize])> {
        let (mut out_len, mut ends_len) = (0, 0);
        loop {
            if self.pos == self.buffer.len() && !self.eof {
                // nothing is kept: csv-core carries partial records in its own output
                let len = self.buffer.len();
                let refill = self
                    .buffer
                    .fill(&mut self.source, len, len, 0)
                    .expect("slice reads cannot fail");
                self.pos = len - refill.shift;
                self.eof = refill.read == 0;
            }

            // an empty input tells csv-core the data has ended
            let (result, read, written, ended) = self.reader.read_record(
                &self.buffer.data()[self.pos..],
                &mut self.output[out_len..],
                &mut self.ends[ends_len..],
            );
            self.pos += read;
            out_len += written;
            ends_len += ended;

            match result {
                ReadRecordResult::InputEmpty => {}
                ReadRecordResult::OutputFull => {
                    let len = self.output.len() * 2;
                    self.output.resize(len, 0);
                }
                ReadRecordResult::OutputEndsFull => {
                    let len = self.ends.len() * 2;
                    self.ends.resize(len, 0);
                }
                ReadRecordResult::Record => {
                    return Some((&self.output[..out_len], &self.ends[..ends_len]));
                }
                ReadRecordResult::End => return None,
            }
        }
    }

    /// Start over on `input`, keeping csv-core's DFA and the buffers.
    fn reset(&mut self, input: &'a [u8]) {
        self.reader.reset();
        self.source = input;
        self.buffer.clear();
        self.pos = 0;
        self.eof = false;
    }

    /// Input offset of the next unread byte.
    fn offset(&self) -> usize {
        self.buffer.source_offset(self.pos)
    }
}

/// Fields of a csv-core record: its ends are relative to the start of the record's output.
struct CoreFields<'a> {
    output: &'a [u8],
    ends: std::slice::Iter<'a, usize>,
    start: usize,
}

impl<'a> CoreFields<'a> {
    fn new(output: &'a [u8], ends: &'a [usize]) -> Self {
        Self {
            output,
            ends: ends.iter(),
            start: 0,
        }
    }
}

impl<'a> Iterator for CoreFields<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<&'a [u8]> {
        let end = *self.ends.next()?;
        let field = &self.output[self.start..end];
        self.start = end;
        Some(field)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.ends.size_hint()
    }
}

impl ExactSizeIterator for CoreFields<'_> {}

struct PreparedCore<'a>(CoreStream<'a>);

impl<'a> PreparedCore<'a> {
    #[inline(always)]
    fn consume<C: Consumer>(&mut self, input: &'a [u8], consumer: &mut C) {
        self.0.reset(input);
        while let Some((output, ends)) = self.0.next() {
            consumer.record(CoreFields::new(output, ends));
        }
    }
}

impl<'a> Prepared<'a> for PreparedCore<'a> {
    fn statistics(&mut self, input: &'a [u8]) -> Result<Statistics, ErrorAt> {
        let mut statistics = Statistics::default();
        self.consume(input, &mut statistics);
        Ok(statistics)
    }

    fn digest(&mut self, input: &'a [u8]) -> (Digest, Option<ErrorAt>) {
        let mut digest = Digest::default();
        self.consume(input, &mut digest);
        (digest, None)
    }
}

fn prepare_core<'a>(d: &Dialect, input: &'a [u8], o: &Options) -> Box<dyn Prepared<'a> + 'a> {
    Box::new(PreparedCore(CoreStream::new(d, input, o)))
}

fn core_records<'a>(d: &Dialect, input: &'a [u8], o: &Options) -> Records<'a> {
    let mut stream = CoreStream::new(d, input, o);
    Box::new(std::iter::from_fn(move || {
        let fields = {
            let (output, ends) = stream.next()?;
            CoreFields::new(output, ends).map(<[u8]>::to_vec).collect()
        };
        Some(Ok(OwnedRecord {
            fields,
            end: stream.offset(),
        }))
    }))
}

fn csv_reader<R: std::io::Read>(d: &Dialect, input: R, o: &Options) -> csv::Reader<R> {
    csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .buffer_capacity(o.capacity.max(1))
        .delimiter(d.delimiter.0)
        .terminator(match d.terminator {
            Terminator::Crlf => csv::Terminator::CRLF,
            Terminator::Byte(b) => csv::Terminator::Any(b.0),
        })
        .quote(d.quote.map_or(b'"', |q| q.0))
        .quoting(d.quote.is_some())
        .escape(d.escape.map(|e| e.0))
        .double_quote(d.double_quote_escape)
        .comment(d.comment.map(|c| c.0))
        .from_reader(input)
}

/// `ByteRecord` fields with their count, since its iterator does not report one.
struct Counted<I> {
    fields: I,
    len: usize,
}

impl<'a, I: Iterator<Item = &'a [u8]>> Iterator for Counted<I> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<&'a [u8]> {
        let field = self.fields.next()?;
        self.len -= 1;
        Some(field)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, I: Iterator<Item = &'a [u8]>> ExactSizeIterator for Counted<I> {}

/// `csv::Reader` over a cursor, which it can seek back to the start: that resets its state
/// without rebuilding the reader.
struct PreparedReader<'a> {
    reader: csv::Reader<Cursor<&'a [u8]>>,
    record: csv::ByteRecord,
}

impl<'a> PreparedReader<'a> {
    #[inline(always)]
    fn consume<C: Consumer>(&mut self, input: &'a [u8], consumer: &mut C) {
        *self.reader.get_mut() = Cursor::new(input);
        // `seek` would skip this when the position already matches
        self.reader
            .seek_raw(SeekFrom::Start(0), csv::Position::new())
            .expect("seeking a cursor cannot fail");
        while self
            .reader
            .read_byte_record(&mut self.record)
            .expect("flexible byte records cannot fail")
        {
            consumer.record(Counted {
                fields: self.record.iter(),
                len: self.record.len(),
            });
        }
    }
}

impl<'a> Prepared<'a> for PreparedReader<'a> {
    fn statistics(&mut self, input: &'a [u8]) -> Result<Statistics, ErrorAt> {
        let mut statistics = Statistics::default();
        self.consume(input, &mut statistics);
        Ok(statistics)
    }

    fn digest(&mut self, input: &'a [u8]) -> (Digest, Option<ErrorAt>) {
        let mut digest = Digest::default();
        self.consume(input, &mut digest);
        (digest, None)
    }
}

fn prepare_reader<'a>(d: &Dialect, input: &'a [u8], o: &Options) -> Box<dyn Prepared<'a> + 'a> {
    let reader = csv_reader(d, Cursor::new(input), o);
    Box::new(PreparedReader {
        reader,
        record: csv::ByteRecord::new(),
    })
}

fn reader_records<'a>(d: &Dialect, input: &'a [u8], o: &Options) -> Records<'a> {
    let mut reader = csv_reader(d, input, o);
    let mut record = csv::ByteRecord::new();
    Box::new(std::iter::from_fn(move || {
        if !reader
            .read_byte_record(&mut record)
            .expect("flexible byte records cannot fail")
        {
            return None;
        }
        let fields = record.iter().map(<[u8]>::to_vec).collect();
        Some(Ok(OwnedRecord {
            fields,
            end: reader.position().byte() as usize,
        }))
    }))
}
