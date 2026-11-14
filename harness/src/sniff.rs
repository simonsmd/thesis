//! Guessing the dialect of a CSV file and describing its content.
//!
//! Everything here is built on one function, [`observe`]: it walks the input with the reference
//! DFA under a given dialect and counts what it sees (records, fields, quoted fields, errors and
//! so on). It uses the reference implementation of the DFA.
//!
//! [`detect`] guesses the dialect by trying candidates on the start of the input and keeping the
//! one under which the records look most alike. Its rules are:
//!
//! 1. The delimiter is the candidate that gives the most consistent field count per record.
//! 2. Of the quoting rules that parse the sample without an error and just as consistently, the
//!    strictest one wins. A feature the sample shows no sign of stays switched off.
//!
//! [`sniff`] does both: it detects the dialect and then describes the whole input with it.

use std::collections::HashMap;

use csv_parser::reference::{self, State, TransitionAction};
use serde_json::{Value, json};

use crate::dialect::{Byte, Dialect, QuoteStyle, Terminator};

/// Version of the output format, to tell results of different sniffer versions apart.
pub const VERSION: u32 = 1;

/// How much of the input [`detect`] looks at.
pub const DETECT_BYTES: usize = 64 * 1024;

// -- the DFA as a table ------------------------------------------------------------------------

const STATES: [State; 11] = [
    State::StartRecord,
    State::InField,
    State::InQuotedFieldOutput,
    State::InQuotedFieldDiscard,
    State::EscapeSequenceInField,
    State::EscapeSequenceInQuotedField,
    State::AfterQuote,
    State::InComment,
    State::EndField,
    State::EndRecord,
    State::Error,
];
const START: u8 = 0;
const IN_QUOTED_FIELD: u8 = 2;
const IN_COMMENT: u8 = 7;

fn index_of(state: State) -> u8 {
    STATES
        .iter()
        .position(|s| *s == state)
        .expect("STATES lists every state") as u8
}

fn in_quotes(state: u8) -> bool {
    matches!(
        STATES[state as usize],
        State::InQuotedFieldOutput
            | State::InQuotedFieldDiscard
            | State::EscapeSequenceInQuotedField
    )
}

// What happens when the DFA reads one byte. A byte can cause several of these at once.
/// The byte is part of the field's content.
const OUTPUT: u32 = 1 << 0;
const FIELD_END: u32 = 1 << 1;
const RECORD_END: u32 = 1 << 2;
const ERROR: u32 = 1 << 3;
/// The first byte of a record.
const RECORD_START: u32 = 1 << 4;
/// A terminator where no record is open: an empty line, or the `\n` of `\r\n`.
const SKIPPED_TERMINATOR: u32 = 1 << 5;
const COMMENT_START: u32 = 1 << 6;
/// A quote that opens a quoted part at the start of a field.
const QUOTE_OPENS_FIELD: u32 = 1 << 7;
/// A quote that opens a quoted part after the field has already started.
const QUOTE_OPENS_INSIDE: u32 = 1 << 8;
const QUOTE_CLOSES: u32 = 1 << 9;
/// The second quote of a doubled quote, which stands for one quote character.
const DOUBLED_QUOTE: u32 = 1 << 10;
/// The field goes on after a closing quote.
const TEXT_AFTER_QUOTE: u32 = 1 << 11;
/// A quote that is ordinary content of an unquoted field.
const PLAIN_QUOTE: u32 = 1 << 12;
const ESCAPE_IN_QUOTES: u32 = 1 << 13;
const ESCAPE_OUTSIDE_QUOTES: u32 = 1 << 14;
/// A delimiter inside quotes.
const QUOTED_DELIMITER: u32 = 1 << 15;
/// A record terminator inside quotes.
const QUOTED_TERMINATOR: u32 = 1 << 16;
/// One of the bytes a parser has to stop at in any state.
const STRUCTURAL: u32 = 1 << 17;
/// One of the bytes a parser has to stop at in the state it is in.
const STRUCTURAL_HERE: u32 = 1 << 18;

/// Events that only matter per field, to describe the field when it ends.
const FIELD_EVENTS: u32 = QUOTE_OPENS_FIELD
    | QUOTE_OPENS_INSIDE
    | QUOTE_CLOSES
    | DOUBLED_QUOTE
    | TEXT_AFTER_QUOTE
    | ESCAPE_IN_QUOTES
    | ESCAPE_OUTSIDE_QUOTES
    | QUOTED_DELIMITER
    | QUOTED_TERMINATOR;

#[derive(Clone, Copy, Default)]
struct Step {
    next: u8,
    events: u32,
}

/// For every state and input byte, the next state and what happened on the way.
struct Table {
    steps: Vec<[Step; 256]>,
    is_terminator: [bool; 256],
}

impl Table {
    fn new(dialect: &Dialect) -> Self {
        let config = dialect.config();
        let mut is_terminator = [false; 256];
        match dialect.terminator {
            Terminator::Crlf => {
                is_terminator[b'\r' as usize] = true;
                is_terminator[b'\n' as usize] = true;
            }
            Terminator::Byte(b) => is_terminator[b.0 as usize] = true,
        }
        let delimiter = dialect.delimiter.0;
        let quote = dialect.quote.map(|b| b.0);
        let escape = dialect.escape.map(|b| b.0);

        // The same sets as `csv_parser::classifier::structural_bytes*`.
        let plain_quote_allowed = dialect.quotes_in_unquoted_fields == Some(true)
            && dialect.quote_style == Some(QuoteStyle::Strict);
        let structural = |b: u8| {
            b == delimiter || is_terminator[b as usize] || Some(b) == quote || Some(b) == escape
        };
        let structural_in_field = |b: u8| {
            b == delimiter
                || is_terminator[b as usize]
                || (Some(b) == quote && !plain_quote_allowed)
                || (Some(b) == escape && dialect.escape_in_unquoted_fields)
        };
        let structural_in_quotes = |b: u8| Some(b) == quote || Some(b) == escape;

        let mut steps = vec![[Step::default(); 256]; STATES.len()];
        for (from, row) in steps.iter_mut().enumerate() {
            for byte in 0..=255u8 {
                let mut state = STATES[from];
                let mut events = 0;
                // A transition that does not consume the byte is followed by another one.
                let next = loop {
                    let (next, action) = reference::transition(&config, state, byte);
                    events |= match (state, next) {
                        (State::StartRecord, State::StartRecord) => SKIPPED_TERMINATOR,
                        (State::StartRecord, State::InComment) => COMMENT_START,
                        (State::StartRecord | State::EndField, State::InQuotedFieldDiscard) => {
                            QUOTE_OPENS_FIELD
                        }
                        (State::InField, State::InQuotedFieldDiscard) => QUOTE_OPENS_INSIDE,
                        (State::InField, State::InField) if Some(byte) == quote => PLAIN_QUOTE,
                        (State::InQuotedFieldOutput, State::AfterQuote) => QUOTE_CLOSES,
                        (State::AfterQuote, State::InQuotedFieldOutput) => DOUBLED_QUOTE,
                        (State::AfterQuote, State::InField) => TEXT_AFTER_QUOTE,
                        (_, State::EscapeSequenceInQuotedField) => ESCAPE_IN_QUOTES,
                        (_, State::EscapeSequenceInField) => ESCAPE_OUTSIDE_QUOTES,
                        (State::InQuotedFieldOutput, State::InQuotedFieldOutput)
                            if byte == delimiter =>
                        {
                            QUOTED_DELIMITER
                        }
                        (State::InQuotedFieldOutput, State::InQuotedFieldOutput)
                            if is_terminator[byte as usize] =>
                        {
                            QUOTED_TERMINATOR
                        }
                        _ => 0,
                    };
                    events |= match next {
                        State::EndField => FIELD_END,
                        State::EndRecord => FIELD_END | RECORD_END,
                        State::Error => ERROR,
                        _ => 0,
                    };
                    match action {
                        TransitionAction::Epsilon => state = next,
                        TransitionAction::Discard => break next,
                        TransitionAction::Output => {
                            events |= OUTPUT;
                            break next;
                        }
                    }
                };

                if matches!(STATES[from], State::StartRecord | State::EndRecord)
                    && events & (SKIPPED_TERMINATOR | COMMENT_START) == 0
                {
                    events |= RECORD_START;
                }
                if from as u8 != IN_COMMENT {
                    if structural(byte) {
                        events |= STRUCTURAL;
                    }
                    let here = if from as u8 == IN_QUOTED_FIELD {
                        structural_in_quotes(byte)
                    } else {
                        structural_in_field(byte)
                    };
                    if here {
                        events |= STRUCTURAL_HERE;
                    }
                }
                row[byte as usize] = Step {
                    next: index_of(next),
                    events,
                };
            }
        }
        Self {
            steps,
            is_terminator,
        }
    }
}

// -- observing ---------------------------------------------------------------------------------

/// Number of records and their total size, for one field count.
#[derive(Debug, Default, Clone, Copy)]
struct RecordsOfCount {
    records: u64,
    bytes: u64,
}

/// What one walk over the input counted. See [`Observation::report`] for what the names mean.
#[derive(Debug, Default)]
pub struct Observation {
    records: u64,
    fields: u64,
    errors: u64,
    comment_lines: u64,
    empty_lines: u64,
    lf: u64,
    crlf: u64,
    cr: u64,
    /// Records whose first byte is `#`, as a hint that the file may have comment lines.
    records_starting_with_hash: u64,
    /// The input ended inside quotes.
    unterminated_quote: bool,
    quoted_fields: u64,
    fully_quoted_fields: u64,
    /// Fully quoted fields that are not empty.
    filled_quoted_fields: u64,
    doubled_quotes: u64,
    escapes_in_quotes: u64,
    escapes_outside_quotes: u64,
    quotes_opening_inside_field: u64,
    text_after_closing_quote: u64,
    plain_quotes: u64,
    fields_with_quoted_delimiter: u64,
    fields_with_quoted_terminator: u64,
    by_field_count: HashMap<u32, RecordsOfCount>,
    /// Bytes of all records that were counted in `by_field_count`.
    record_bytes: u64,
    /// Bytes of records that ended with an error or inside quotes.
    broken_bytes: u64,
    longest_record: u64,
    /// Size of the cut off last record of an incomplete sample.
    tail_bytes: u64,
    tail_in_quotes: bool,
    detail: Option<Box<Detail>>,
}

/// The more expensive counts, only collected for the final description.
#[derive(Debug, Default)]
struct Detail {
    field_bytes: u64,
    empty_fields: u64,
    longest_field: u64,
    field_lengths: Histogram,
    record_lengths: Histogram,
    /// How often each exact field length occurs, over all fields and per column.
    length_counts: HashMap<u32, u64>,
    column_length_counts: Vec<HashMap<u32, u64>>,
    column_fields: Vec<u64>,
    column_quoted: Vec<u64>,
    /// Field lengths of the previous and the current record.
    previous: Vec<u32>,
    current: Vec<u32>,
    same_length_as_previous: u64,
    compared_to_previous: u64,
    previous_field_quoted: bool,
    quote_switches: u64,
    quoted_field_bytes: u64,
    copied_fields: u64,
    copied_field_bytes: u64,
    records_ending_with_empty_field: u64,
    /// Field counts of the first records, to find where the table starts.
    first_field_counts: Vec<u32>,
    structural: u64,
    structural_here: u64,
    structural_gaps: Histogram,
    structural_gaps_here: Histogram,
    last_structural: Option<usize>,
    last_structural_here: Option<usize>,
    non_ascii_bytes: u64,
    nul_bytes: u64,
    parts: Vec<Part>,
}

/// The counts for one of several equally sized pieces of the input.
#[derive(Debug, Default, Clone)]
struct Part {
    by_field_count: HashMap<u32, u64>,
    errors: u64,
}

/// Columns beyond this are not looked at one by one.
const MAX_COLUMNS: usize = 256;
const FIRST_RECORDS: usize = 64;

/// Counts of values by their size: slot 0 is for 0, slot `n` for values from `2^(n-1)` up to
/// `2^n - 1`. So the slots are 0, 1, 2-3, 4-7, 8-15, ... Histograms of several files can be
/// added up slot by slot.
#[derive(Debug, Default, Clone)]
struct Histogram(Vec<u64>);

impl Histogram {
    fn add(&mut self, value: usize) {
        let slot = (usize::BITS - value.leading_zeros()) as usize;
        if self.0.len() <= slot {
            self.0.resize(slot + 1, 0);
        }
        self.0[slot] += 1;
    }
}

/// How unpredictable a value is, in bits: 0 if it is always the same, `log2(n)` if `n` values
/// are equally likely (Shannon entropy).
fn entropy(counts: &HashMap<u32, u64>) -> f64 {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let bits: f64 = counts
        .values()
        .map(|&n| {
            let p = n as f64 / total as f64;
            -p * p.log2()
        })
        .sum();
    bits.max(0.0)
}

fn share(part: u64, whole: u64) -> f64 {
    if whole == 0 {
        0.0
    } else {
        part as f64 / whole as f64
    }
}

/// Rounded so that the JSON output stays short.
fn rounded(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// The state of one walk over the input.
struct Walk<'a> {
    data: &'a [u8],
    seen: Observation,
    in_record: bool,
    record_start: usize,
    record_fields: u32,
    field_length: usize,
    field_events: u32,
    /// Position of the last `\r` that was counted as a line ending.
    last_cr: Option<usize>,
    comment_start: usize,
}

impl Walk<'_> {
    fn count_terminator(&mut self, position: usize) {
        match self.data[position] {
            b'\r' => {
                self.seen.cr += 1;
                self.last_cr = Some(position);
            }
            b'\n' if position > 0 && self.last_cr == Some(position - 1) => {
                self.seen.cr -= 1;
                self.seen.crlf += 1;
            }
            b'\n' => self.seen.lf += 1,
            _ => {}
        }
    }

    fn end_field(&mut self) {
        let events = std::mem::take(&mut self.field_events);
        let length = std::mem::take(&mut self.field_length);
        let column = self.record_fields as usize;
        self.record_fields += 1;

        let seen = &mut self.seen;
        seen.fields += 1;
        let quoted = events & (QUOTE_OPENS_FIELD | QUOTE_OPENS_INSIDE) != 0;
        // quoted from its first to its last byte, the only form RFC 4180 knows
        let fully_quoted = events & QUOTE_OPENS_FIELD != 0
            && events & QUOTE_CLOSES != 0
            && events & (QUOTE_OPENS_INSIDE | TEXT_AFTER_QUOTE) == 0;
        seen.quoted_fields += quoted as u64;
        seen.fully_quoted_fields += fully_quoted as u64;
        seen.filled_quoted_fields += (fully_quoted && length > 0) as u64;
        seen.fields_with_quoted_delimiter += (events & QUOTED_DELIMITER != 0) as u64;
        seen.fields_with_quoted_terminator += (events & QUOTED_TERMINATOR != 0) as u64;

        let Some(detail) = seen.detail.as_deref_mut() else {
            return;
        };
        detail.field_bytes += length as u64;
        detail.empty_fields += (length == 0) as u64;
        detail.longest_field = detail.longest_field.max(length as u64);
        detail.field_lengths.add(length);
        let length = u32::try_from(length).unwrap_or(u32::MAX);
        *detail.length_counts.entry(length).or_default() += 1;

        if quoted {
            detail.quoted_field_bytes += length as u64;
        }
        // The content is not one unbroken piece of the input, so a parser that hands out slices
        // of the input has to copy this field.
        if events
            & (DOUBLED_QUOTE
                | ESCAPE_IN_QUOTES
                | ESCAPE_OUTSIDE_QUOTES
                | QUOTE_OPENS_INSIDE
                | TEXT_AFTER_QUOTE)
            != 0
        {
            detail.copied_fields += 1;
            detail.copied_field_bytes += length as u64;
        }
        if column > 0 && quoted != detail.previous_field_quoted {
            detail.quote_switches += 1;
        }
        detail.previous_field_quoted = quoted;

        if let Some(&previous) = detail.previous.get(column) {
            detail.compared_to_previous += 1;
            detail.same_length_as_previous += (previous == length) as u64;
        }
        detail.current.push(length);
        if column < MAX_COLUMNS {
            if detail.column_fields.len() <= column {
                detail.column_fields.resize(column + 1, 0);
                detail.column_quoted.resize(column + 1, 0);
                detail
                    .column_length_counts
                    .resize_with(column + 1, HashMap::new);
            }
            detail.column_fields[column] += 1;
            detail.column_quoted[column] += quoted as u64;
            *detail.column_length_counts[column]
                .entry(length)
                .or_default() += 1;
        }
    }

    /// Close the open record, which ends just before `end`.
    fn end_record(&mut self, end: usize) {
        let fields = std::mem::take(&mut self.record_fields);
        let bytes = (end - self.record_start) as u64;
        self.in_record = false;

        let seen = &mut self.seen;
        seen.records += 1;
        seen.record_bytes += bytes;
        seen.longest_record = seen.longest_record.max(bytes);
        let of_count = seen.by_field_count.entry(fields).or_default();
        of_count.records += 1;
        of_count.bytes += bytes;

        let Some(detail) = seen.detail.as_deref_mut() else {
            return;
        };
        detail.record_lengths.add(bytes as usize);
        std::mem::swap(&mut detail.previous, &mut detail.current);
        detail.current.clear();
        // the last field was empty: the record ends with a delimiter
        if fields > 1 && detail.previous.last() == Some(&0) {
            detail.records_ending_with_empty_field += 1;
        }
        if detail.first_field_counts.len() < FIRST_RECORDS {
            detail.first_field_counts.push(fields);
        }
        if !detail.parts.is_empty() {
            let part = self.record_start * detail.parts.len() / self.data.len();
            *detail.parts[part].by_field_count.entry(fields).or_default() += 1;
        }
    }

    /// Give up on the open record, which ends just before `end`.
    fn drop_record(&mut self, end: usize) {
        self.seen.broken_bytes += (end - self.record_start) as u64;
        self.in_record = false;
        self.record_fields = 0;
        self.field_length = 0;
        self.field_events = 0;
        if let Some(detail) = self.seen.detail.as_deref_mut() {
            detail.current.clear();
        }
    }

    fn structural(&mut self, position: usize, events: u32) {
        let Some(detail) = self.seen.detail.as_deref_mut() else {
            return;
        };
        if events & STRUCTURAL != 0 {
            detail.structural += 1;
            detail
                .structural_gaps
                .add(position - detail.last_structural.map_or(0, |last| last + 1));
            detail.last_structural = Some(position);
        }
        if events & STRUCTURAL_HERE != 0 {
            detail.structural_here += 1;
            detail
                .structural_gaps_here
                .add(position - detail.last_structural_here.map_or(0, |last| last + 1));
            detail.last_structural_here = Some(position);
        }
    }
}

/// Walk `data` with the reference DFA for `dialect` and count what is in it.
///
/// `complete` says whether `data` is the whole file. If it is only the start of one, the last
/// record is cut off somewhere and is left out of the counts.
///
/// A parser stops at the first error. This walk counts the error, skips to the end of the line
/// and goes on, so that one bad record does not hide the rest of the file.
///
/// `detail` also collects the length statistics. `parts` above 0 additionally counts records
/// separately for that many equally sized pieces of the input, to see whether the file changes
/// along the way.
pub fn observe(
    dialect: &Dialect,
    data: &[u8],
    complete: bool,
    detail: bool,
    parts: usize,
) -> Observation {
    let table = Table::new(dialect);
    let mut walk = Walk {
        data,
        seen: Observation::default(),
        in_record: false,
        record_start: 0,
        record_fields: 0,
        field_length: 0,
        field_events: 0,
        last_cr: None,
        comment_start: 0,
    };
    if detail {
        walk.seen.detail = Some(Box::new(Detail {
            parts: vec![Part::default(); if data.is_empty() { 0 } else { parts }],
            non_ascii_bytes: data.iter().filter(|b| **b >= 0x80).count() as u64,
            nul_bytes: data.iter().filter(|b| **b == 0).count() as u64,
            ..Default::default()
        }));
    }

    let mut state = START;
    let mut i = 0;
    while i < data.len() {
        let byte = data[i];
        let before = state;
        let Step { next, events } = table.steps[state as usize][byte as usize];
        state = next;

        // by far the most common case: an ordinary byte inside a field
        if events == OUTPUT {
            walk.field_length += 1;
            i += 1;
            continue;
        }

        if events & RECORD_START != 0 {
            walk.in_record = true;
            walk.record_start = i;
            walk.seen.records_starting_with_hash += (byte == b'#') as u64;
        }
        walk.field_length += (events & OUTPUT != 0) as usize;
        walk.field_events |= events & FIELD_EVENTS;
        walk.seen.doubled_quotes += (events & DOUBLED_QUOTE != 0) as u64;
        walk.seen.escapes_in_quotes += (events & ESCAPE_IN_QUOTES != 0) as u64;
        walk.seen.escapes_outside_quotes += (events & ESCAPE_OUTSIDE_QUOTES != 0) as u64;
        walk.seen.quotes_opening_inside_field += (events & QUOTE_OPENS_INSIDE != 0) as u64;
        walk.seen.text_after_closing_quote += (events & TEXT_AFTER_QUOTE != 0) as u64;
        walk.seen.plain_quotes += (events & PLAIN_QUOTE != 0) as u64;
        if events & (STRUCTURAL | STRUCTURAL_HERE) != 0 {
            walk.structural(i, events);
        }

        if events & SKIPPED_TERMINATOR != 0 {
            let second_half_of_crlf = byte == b'\n' && i > 0 && walk.last_cr == Some(i - 1);
            walk.seen.empty_lines += !second_half_of_crlf as u64;
            walk.count_terminator(i);
        }
        if events & COMMENT_START != 0 {
            walk.seen.comment_lines += 1;
            walk.comment_start = i;
        }
        if before == IN_COMMENT && state == START {
            walk.count_terminator(i);
        }
        if events & FIELD_END != 0 {
            walk.end_field();
        }
        if events & RECORD_END != 0 {
            walk.end_record(i);
            walk.count_terminator(i);
        }
        if events & ERROR != 0 {
            walk.seen.errors += 1;
            let line_end = data[i..]
                .iter()
                .position(|b| table.is_terminator[*b as usize])
                .map_or(data.len(), |n| i + n);
            if let Some(detail) = walk.seen.detail.as_deref_mut()
                && !detail.parts.is_empty()
            {
                let part = walk.record_start * detail.parts.len() / data.len();
                detail.parts[part].errors += 1;
            }
            walk.drop_record(line_end);
            if line_end < data.len() {
                walk.count_terminator(line_end);
            }
            state = START;
            i = line_end;
        }
        i += 1;
    }

    if walk.in_record {
        if !complete {
            walk.seen.tail_bytes = (data.len() - walk.record_start) as u64;
            walk.seen.tail_in_quotes = in_quotes(state);
        } else if in_quotes(state) {
            // The parsers return this record like any other, but a quote that is never closed
            // usually means that the dialect is wrong, so it must not count as a good record.
            walk.seen.unterminated_quote = true;
            walk.seen.records += 1;
            walk.end_field();
            walk.drop_record(data.len());
        } else {
            walk.end_field();
            walk.end_record(data.len());
        }
    }
    walk.seen
}

impl Observation {
    /// The field count that most records have, and how many records and bytes have it.
    fn common_field_count(&self) -> (u32, RecordsOfCount) {
        self.by_field_count
            .iter()
            .map(|(count, of_count)| (*count, *of_count))
            .max_by_key(|(count, of_count)| (of_count.records, of_count.bytes, *count))
            .unwrap_or_default()
    }

    /// The cut off end of a sample is far longer than any record before it and sits inside
    /// quotes. That is what a quote character that is not really one looks like: it opens a
    /// quoted field that swallows the rest of the input.
    fn tail_was_swallowed(&self) -> bool {
        self.tail_in_quotes && self.tail_bytes > 2 * self.longest_record
    }

    /// The share of the input that lies in records with the most common field count, from 0 to 1.
    ///
    /// Measured in bytes and not in records, so that a few huge records, which is what wrongly
    /// paired quotes produce, weigh as much as the input they cover. A dialect that puts
    /// everything into fewer than `min_columns` columns scores 0: a single column is what every
    /// wrong delimiter produces, so it says nothing about the file.
    pub fn score(&self, min_columns: u32) -> f64 {
        let (count, common) = self.common_field_count();
        if count < min_columns {
            return 0.0;
        }
        let swallowed = if self.tail_was_swallowed() {
            self.tail_bytes
        } else {
            0
        };
        share(
            common.bytes,
            self.record_bytes + self.broken_bytes + swallowed,
        )
    }

    /// Parsed without anything that hints at a wrong dialect.
    fn clean(&self) -> bool {
        self.errors == 0 && !self.unterminated_quote && !self.tail_was_swallowed()
    }

    /// The counts as JSON. The keys are documented in `scripts/csvscrape/features.py`, which
    /// reads them.
    pub fn report(&self) -> Value {
        let (common, of_common) = self.common_field_count();
        let counts = || self.by_field_count.iter().filter(|(_, c)| c.records > 0);
        let mut report = json!({
            "records": self.records,
            "fields": self.fields,
            "errors": self.errors,
            "unterminated_quote": self.unterminated_quote,
            "comment_lines": self.comment_lines,
            "empty_lines": self.empty_lines,
            "line_endings": {"lf": self.lf, "crlf": self.crlf, "cr": self.cr},
            "common_field_count": common,
            "common_field_count_share": rounded(share(of_common.records, self.records)),
            "consistent_bytes_share": rounded(self.score(2)),
            "min_field_count": counts().map(|(n, _)| *n).min().unwrap_or(0),
            "max_field_count": counts().map(|(n, _)| *n).max().unwrap_or(0),
            "distinct_field_counts": counts().count(),
            "short_records": counts().filter(|(n, _)| **n < common).map(|(_, c)| c.records).sum::<u64>(),
            "long_records": counts().filter(|(n, _)| **n > common).map(|(_, c)| c.records).sum::<u64>(),
            "longest_record": self.longest_record,
            "quoted_fields": self.quoted_fields,
            "fully_quoted_fields": self.fully_quoted_fields,
            "fields_with_quoted_delimiter": self.fields_with_quoted_delimiter,
            "fields_with_quoted_terminator": self.fields_with_quoted_terminator,
            "doubled_quotes": self.doubled_quotes,
            "escapes_in_quoted_fields": self.escapes_in_quotes,
            "escapes_in_unquoted_fields": self.escapes_outside_quotes,
            "quotes_opening_inside_field": self.quotes_opening_inside_field,
            "text_after_closing_quote": self.text_after_closing_quote,
            "quotes_in_unquoted_fields": self.plain_quotes,
        });
        let Some(detail) = &self.detail else {
            return report;
        };

        let columns = (common as usize).min(detail.column_fields.len());
        let column_fields: u64 = detail.column_fields[..columns].iter().sum();
        // per column, weighted by how many fields the column has
        let column_entropy: f64 = (0..columns)
            .map(|c| {
                entropy(&detail.column_length_counts[c])
                    * share(detail.column_fields[c], column_fields)
            })
            .sum();
        let quoting = |wanted: fn(u64, u64) -> bool| {
            (0..columns)
                .filter(|&c| wanted(detail.column_quoted[c], detail.column_fields[c]))
                .count()
        };
        let preamble = detail
            .first_field_counts
            .iter()
            .position(|n| *n == common)
            .unwrap_or(0);
        let parts: Vec<Value> = detail
            .parts
            .iter()
            .map(|part| {
                let records: u64 = part.by_field_count.values().sum();
                let (count, most) = part
                    .by_field_count
                    .iter()
                    .max_by_key(|(count, n)| (**n, **count))
                    .map_or((0, 0), |(c, n)| (*c, *n));
                json!({
                    "records": records,
                    "errors": part.errors,
                    "common_field_count": count,
                    "common_field_count_share": rounded(share(most, records)),
                })
            })
            .collect();

        let more = json!({
            "preamble_records": preamble,
            "records_ending_with_empty_field": detail.records_ending_with_empty_field,
            "field_bytes": detail.field_bytes,
            "empty_fields": detail.empty_fields,
            "longest_field": detail.longest_field,
            "field_length_histogram": detail.field_lengths.0,
            "record_length_histogram": detail.record_lengths.0,
            "field_length_entropy": rounded(entropy(&detail.length_counts)),
            "column_length_entropy": rounded(column_entropy),
            "same_length_as_previous_record": rounded(share(detail.same_length_as_previous, detail.compared_to_previous)),
            "quoted_field_bytes": detail.quoted_field_bytes,
            "quote_switches": detail.quote_switches,
            "columns_always_quoted": quoting(|quoted, fields| quoted == fields),
            "columns_never_quoted": quoting(|quoted, _| quoted == 0),
            "columns_sometimes_quoted": quoting(|quoted, fields| quoted > 0 && quoted < fields),
            "copied_fields": detail.copied_fields,
            "copied_field_bytes": detail.copied_field_bytes,
            "structural_bytes": detail.structural,
            "structural_bytes_in_state": detail.structural_here,
            "structural_gap_histogram": detail.structural_gaps.0,
            "structural_gap_in_state_histogram": detail.structural_gaps_here.0,
            "non_ascii_bytes": detail.non_ascii_bytes,
            "nul_bytes": detail.nul_bytes,
            "parts": parts,
        });
        if let (Value::Object(report), Value::Object(more)) = (&mut report, more) {
            report.extend(more);
        }
        report
    }
}

// -- detecting ---------------------------------------------------------------------------------

/// Delimiters that are tried. On a tie the earlier one wins. Tab and `\x01` come first because
/// they hardly ever occur in text: if one of them splits the records evenly, it is the delimiter,
/// however evenly the commas in the text happen to fall. The others are ordered by how common
/// they are as delimiters.
const DELIMITERS: [u8; 6] = [b'\t', 0x01, b',', b';', b'|', b':'];
const QUOTES: [u8; 2] = [b'"', b'\''];

/// How much better a delimiter has to score to replace one earlier in [`DELIMITERS`]. Also how
/// much a quote character that never wraps a whole field has to improve the score to be taken
/// as one.
const MARGIN: f64 = 0.10;
/// Quoting rules that score within this of the best one count as equally good.
const SAME_SCORE: f64 = 0.01;

/// The sample for [`detect`] is made longer until it has this many line breaks, for files with
/// very long records.
const SAMPLE_LINES: usize = 32;
const MAX_SAMPLE_BYTES: usize = 8 << 20;

/// The result of [`detect`].
#[derive(Debug, Clone)]
pub struct Detection {
    pub dialect: Dialect,
    /// [`Observation::score`] of the dialect on the sample.
    pub score: f64,
    /// How far the score is ahead of the best other delimiter. Close to 0 means that another
    /// delimiter fits the sample about as well.
    pub margin: f64,
    /// False if no delimiter splits the sample into at least two columns. The dialect then has
    /// a comma as its delimiter and the file is read as a single column.
    pub delimiter_found: bool,
    /// Less strict quoting rules to fall back on, should the dialect fail further into the file.
    fallbacks: Vec<Dialect>,
}

impl Detection {
    pub fn report(&self) -> Value {
        json!({
            "score": rounded(self.score),
            "margin": rounded(self.margin),
            "delimiter_found": self.delimiter_found,
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Quoting {
    /// Quotes only open a field; a quote elsewhere is an error.
    Strict,
    /// Like `Strict`, but a quote inside an unquoted field is ordinary content.
    StrictPlainQuotes,
    /// A quote switches quoting on and off anywhere.
    Toggle,
}

fn candidate(delimiter: u8, quote: Option<(u8, Quoting)>, escape: bool) -> Dialect {
    let (style, plain_quotes) = match quote {
        Some((_, Quoting::Strict)) => (Some(QuoteStyle::Strict), Some(false)),
        Some((_, Quoting::StrictPlainQuotes)) => (Some(QuoteStyle::Strict), Some(true)),
        Some((_, Quoting::Toggle)) => (Some(QuoteStyle::Toggle), None),
        None => (None, None),
    };
    Dialect {
        delimiter: Byte(delimiter),
        terminator: Terminator::Crlf,
        quote: quote.map(|(q, _)| Byte(q)),
        quote_style: style,
        quotes_in_unquoted_fields: plain_quotes,
        escape: escape.then_some(Byte(b'\\')),
        escape_in_unquoted_fields: false,
        double_quote_escape: quote.is_some(),
        comment: None,
    }
}

/// Guess the dialect from a sample of the input.
pub fn detect(sample: &[u8], complete: bool) -> Detection {
    let look = |dialect: &Dialect| observe(dialect, sample, complete, false, 0);
    let occurs = |byte: u8| sample.contains(&byte);
    let quotes: Vec<u8> = QUOTES
        .into_iter()
        .filter(|q| sample.iter().filter(|b| *b == q).count() >= 2)
        .collect();

    // 1. The delimiter. Quotes are read in the most forgiving way here, since the exact rules
    //    are only worked out in step 2. Each delimiter gets the score of its best reading.
    let score_of = |delimiter: u8| {
        let readings =
            std::iter::once(None).chain(quotes.iter().map(|q| Some((*q, Quoting::Toggle))));
        readings
            .map(|quote| look(&candidate(delimiter, quote, false)).score(2))
            .fold(0.0, f64::max)
    };
    let scores: Vec<(u8, f64)> = DELIMITERS
        .into_iter()
        .filter(|d| occurs(*d))
        .map(|d| (d, score_of(d)))
        .collect();
    let mut best: Option<(u8, f64)> = None;
    for &(delimiter, score) in &scores {
        let replaces = match best {
            None => score > 0.0,
            // A delimiter that splits every record alike beats an earlier one that does not:
            // `1,5;2,5` under a header `a;b` is a semicolon file. Not so for the colon, which
            // splits a column of times just as evenly.
            Some((_, best)) => {
                score > best + MARGIN || (score == 1.0 && best < 1.0 && delimiter != b':')
            }
        };
        if replaces {
            best = Some((delimiter, score));
        }
    }
    // A space only if nothing else fits and it splits every record into the same few columns.
    if best.is_none() && occurs(b' ') {
        let seen = look(&candidate(b' ', None, false));
        if seen.score(2) == 1.0 && seen.common_field_count().0 >= 3 {
            best = Some((b' ', 1.0));
        }
    }
    let runner_up = |delimiter: u8| {
        scores
            .iter()
            .filter(|(d, _)| *d != delimiter)
            .map(|(_, s)| *s)
            .fold(0.0, f64::max)
    };
    // Without a delimiter the file has one column. It can still quote lines that contain a
    // comma, so the quoting rules are worked out with a comma and one column counts as fine.
    let (delimiter, margin, min_columns) = match best {
        Some((delimiter, score)) => (delimiter, score - runner_up(delimiter), 2),
        None => (b',', 0.0, 1),
    };
    let score = |seen: &Observation| seen.score(min_columns);

    // 2. The quoting rules, strictest first. A quote character that wraps whole fields somewhere
    //    is taken to be a real one. One that only shows up inside fields (an apostrophe, inches)
    //    is ordinary text, unless reading it as a quote makes the records far more consistent.
    let wraps_fields = |quote: u8| {
        let seen = look(&candidate(delimiter, Some((quote, Quoting::Toggle)), false));
        // An empty quoted field counts for `"` only: `''` is also how some text writes a quote.
        seen.filled_quoted_fields > 0 || (quote == b'"' && seen.fully_quoted_fields > 0)
    };
    let (wrapping, other): (Vec<u8>, Vec<u8>) = quotes.iter().partition(|q| wraps_fields(**q));
    let escapes: &[bool] = if occurs(b'\\') {
        &[false, true]
    } else {
        &[false]
    };
    let with_quote = |quote: u8, quoting: Quoting| {
        escapes
            .iter()
            .map(move |&escape| candidate(delimiter, Some((quote, quoting)), escape))
    };
    let mut candidates = Vec::new();
    for &quote in &wrapping {
        for quoting in [Quoting::Strict, Quoting::StrictPlainQuotes, Quoting::Toggle] {
            candidates.extend(with_quote(quote, quoting));
        }
    }
    let unquoted = candidates.len();
    candidates.push(candidate(delimiter, None, false));
    for &quote in &other {
        candidates.extend(with_quote(quote, Quoting::Toggle));
    }

    let seen: Vec<Observation> = candidates.iter().map(look).collect();
    let top = seen[..=unquoted].iter().map(score).fold(0.0, f64::max);
    // without a quote character nothing can go wrong, so the unquoted reading is always clean
    let mut chosen = (0..=unquoted)
        .find(|&i| seen[i].clean() && score(&seen[i]) >= top - SAME_SCORE)
        .unwrap_or(unquoted);
    for i in unquoted + 1..candidates.len() {
        if seen[i].clean() && score(&seen[i]) >= score(&seen[chosen]) + MARGIN {
            chosen = i;
        }
    }
    let mut dialect = candidates[chosen].clone();
    let mut best_score = score(&seen[chosen]);

    // 3. Extras, each kept only if it makes the records more consistent.
    //    Comment lines: only `#`, and only if some records start with it. A header line that
    //    starts with `#` has as many fields as the records below it and stays a record.
    if seen[chosen].records_starting_with_hash > 0 {
        let with_comments = Dialect {
            comment: Some(Byte(b'#')),
            ..dialect.clone()
        };
        let seen = look(&with_comments);
        if seen.clean() && score(&seen) > best_score {
            (dialect, best_score) = (with_comments, score(&seen));
        }
    }
    //    A backslash that also escapes outside of quotes, as in `a\,b`.
    if occurs(b'\\') {
        let escape_everywhere = Dialect {
            escape: Some(Byte(b'\\')),
            escape_in_unquoted_fields: true,
            ..dialect.clone()
        };
        let seen = look(&escape_everywhere);
        if seen.clean() && score(&seen) >= best_score + SAME_SCORE {
            (dialect, best_score) = (escape_everywhere, score(&seen));
        }
    }
    // A file that escapes quotes with a backslash and never doubles one does not use doubling.
    if dialect.quote.is_some() && dialect.escape.is_some() && look(&dialect).doubled_quotes == 0 {
        dialect.double_quote_escape = false;
    }

    let fallbacks = candidates
        .into_iter()
        .zip(&seen)
        .skip(chosen + 1)
        .filter(|(_, seen)| seen.clean())
        .map(|(candidate, _)| Dialect {
            comment: dialect.comment,
            ..candidate
        })
        .collect();
    Detection {
        dialect,
        score: best_score,
        margin,
        delimiter_found: best.is_some(),
        fallbacks,
    }
}

/// The start of `data` that [`detect`] looks at: [`DETECT_BYTES`], or more if that holds only a
/// few lines.
fn sample_of(data: &[u8]) -> &[u8] {
    let mut length = DETECT_BYTES.min(data.len());
    let lines = |length: usize| {
        data[..length]
            .iter()
            .filter(|b| **b == b'\n' || **b == b'\r')
            .count()
    };
    while length < data.len().min(MAX_SAMPLE_BYTES) && lines(length) < SAMPLE_LINES {
        length = (length * 2).min(data.len());
    }
    &data[..length]
}

/// Detect the dialect of `data` and describe all of it with that dialect.
///
/// The dialect is detected on the start of the input. What comes after the start can correct it
/// in three ways:
///
/// - The start has no `"` but the rest does. The quoting rules are then taken from the part of
///   the input where the first one appears.
/// - The dialect runs into errors further in. The next less strict quoting rule that gets through
///   all of the input without one is used instead.
/// - No record ends with `\r` or `\r\n`. Then `\n` alone is the terminator.
pub fn sniff(data: &[u8], complete: bool, parts: usize) -> (Detection, Observation) {
    let sample = sample_of(data);
    let rest = &data[sample.len()..];
    let mut detection = detect(sample, complete && rest.is_empty());

    if detection.dialect.quote.is_none()
        && let Some(first_quote) = rest.iter().position(|b| *b == b'"')
    {
        let line_start = data[..sample.len() + first_quote]
            .iter()
            .rposition(|b| *b == b'\n')
            .map_or(0, |n| n + 1);
        let around = sample_of(&data[line_start..]);
        let there = detect(around, complete && line_start + around.len() == data.len());
        if there.dialect.quote.is_some() && there.dialect.delimiter == detection.dialect.delimiter {
            detection.dialect = Dialect {
                comment: detection.dialect.comment,
                ..there.dialect
            };
            detection.fallbacks = there.fallbacks;
        }
    }

    let mut seen = observe(&detection.dialect, data, complete, true, parts);
    if !seen.clean() {
        let fallbacks = std::mem::take(&mut detection.fallbacks);
        let passes = |dialect: &Dialect| observe(dialect, data, complete, false, 0).clean();
        if let Some(dialect) = fallbacks.into_iter().find(passes) {
            detection.dialect = dialect;
            seen = observe(&detection.dialect, data, complete, true, parts);
        }
    }
    // `crlf` ends records at `\r`, `\n` and `\r\n` alike, so it is right for every file. A file
    // where no record ends with a `\r` needs less.
    if seen.cr == 0 && seen.crlf == 0 {
        detection.dialect.terminator = Terminator::Byte(Byte(b'\n'));
        // a `\r` inside quotes is no longer a byte the parser has to stop at
        if data.contains(&b'\r') {
            seen = observe(&detection.dialect, data, complete, true, parts);
        }
    }
    (detection, seen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialect_of(text: &str) -> Dialect {
        sniff(text.as_bytes(), true, 0).0.dialect
    }

    #[test]
    fn plain_file() {
        let d = dialect_of("a,b,c\n1,2,3\n4,5,6\n");
        assert_eq!(d.delimiter, Byte(b','));
        assert_eq!(d.quote, None);
        assert_eq!(d.terminator, Terminator::Byte(Byte(b'\n')));
        assert_eq!(dialect_of("a,b\r\n1,2\r\n").terminator, Terminator::Crlf);
    }

    #[test]
    fn delimiter() {
        // the comma splits every data row evenly too, but not the header
        assert_eq!(
            dialect_of("a;b;c\n1,5;2,5;3,5\n4,5;5,5;6,5\n").delimiter,
            Byte(b';')
        );
        assert_eq!(dialect_of("a\tb\n1\t2\n").delimiter, Byte(b'\t'));
        // the colons split evenly as well, but the comma comes first
        assert_eq!(dialect_of("12:30:00,5\n12:31:00,6\n").delimiter, Byte(b','));
        assert!(!sniff(b"one\ntwo\nthree\n", true, 0).0.delimiter_found);
        // one column, where lines with a comma are quoted
        let d = dialect_of("one\n\"two, three\"\nfour\n");
        assert_eq!((d.delimiter, d.quote), (Byte(b','), Some(Byte(b'"'))));
    }

    #[test]
    fn quoting_is_as_strict_as_the_file_allows() {
        let d = dialect_of("a,b\n\"x,1\",2\n\"y\",3\n");
        assert_eq!(
            (d.quote, d.quote_style, d.quotes_in_unquoted_fields),
            (Some(Byte(b'"')), Some(QuoteStyle::Strict), Some(false))
        );

        // a quote inside an unquoted field next to real quoted fields
        let d = dialect_of("a,b\n\"x,1\",5\" pipe\n\"y\",3\n");
        assert_eq!(
            (d.quote_style, d.quotes_in_unquoted_fields),
            (Some(QuoteStyle::Strict), Some(true))
        );

        // text after a closing quote only parses if quotes toggle
        let d = dialect_of("a,b\n\"x,1\" and more,2\n\"y\",3\n");
        assert_eq!(d.quote_style, Some(QuoteStyle::Toggle));

        // apostrophes and inch marks in a file that quotes nothing
        assert_eq!(
            dialect_of("a,b\ndon't,5\" pipe\nwon't,6\" pipe\n").quote,
            None
        );
    }

    #[test]
    fn doubled_quotes_and_backslash_escapes() {
        let d = dialect_of("a,b\n\"say \"\"hi\"\", you\",2\n\"C:\\dir\",3\n");
        assert_eq!((d.escape, d.double_quote_escape), (None, true));

        let d = dialect_of("a,b\n\"say \\\"hi\\\", you\",2\n\"y\",3\n");
        assert_eq!(
            (d.escape, d.escape_in_unquoted_fields, d.double_quote_escape),
            (Some(Byte(b'\\')), false, false)
        );
    }

    #[test]
    fn comment_lines() {
        assert_eq!(
            dialect_of("# made by a tool\na,b\n1,2\n3,4\n").comment,
            Some(Byte(b'#'))
        );
        // a header that happens to start with `#` is a record
        assert_eq!(dialect_of("#id,name\n1,x\n2,y\n").comment, None);
    }

    #[test]
    fn counts() {
        let (_, seen) = sniff(
            b"a,b\r\n\"x,1\",22\r\n\r\ny,\"q\"\"q\"\r\nshort\r\n",
            true,
            0,
        );
        let report = seen.report();
        assert_eq!(report["records"], 4);
        assert_eq!(report["fields"], 7);
        assert_eq!(report["common_field_count"], 2);
        assert_eq!(report["short_records"], 1);
        assert_eq!(report["empty_lines"], 1);
        assert_eq!(report["line_endings"]["crlf"], 5);
        assert_eq!(report["line_endings"]["cr"], 0);
        assert_eq!(report["quoted_fields"], 2);
        assert_eq!(report["fields_with_quoted_delimiter"], 1);
        assert_eq!(report["doubled_quotes"], 1);
        assert_eq!(report["copied_fields"], 1);
        // lengths 1,1,3,2,1,3,5: none of 0, three of 1, three of 2-3, one of 4-7
        assert_eq!(report["field_length_histogram"], json!([0, 3, 3, 1]));
    }

    #[test]
    fn the_cut_off_end_of_a_sample_is_not_a_record() {
        let (_, seen) = sniff(b"a,b\n1,2\n3", false, 0);
        assert_eq!(seen.report()["records"], 2);
        assert_eq!(seen.report()["short_records"], 0);
    }

    #[test]
    fn errors_are_counted_and_skipped() {
        let strict = Dialect::default();
        let seen = observe(&strict, b"a,b\nx\"y,1\nc,d\n", true, true, 0);
        assert_eq!((seen.errors, seen.records), (1, 2));
    }
}
