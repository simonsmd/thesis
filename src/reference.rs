use crate::{
    config::{Config, QuoteStyle}, parser::{Cursor, Dfa, DfaResult},
};

// TODO: should be copy?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    StartRecord = 12,
    InField = 0,
    InQuotedFieldOutput = 6,
    InQuotedFieldDiscard = 18,
    EscapeSequenceInField = 24,
    EscapeSequenceInQuotedField = 30,
    AfterQuote = 36,
    InComment = 42,
    EndField = 48,
    EndRecord = 54,
    Error = 60,
}

/// 6 Bits per state so the masking for the bitshift dfa is done automatically by the shift instructions
pub const STATE_MASK: u64 = 0b11_1111;

pub fn is_output(state: u64) -> bool {
    state <= State::InQuotedFieldOutput as u64
}

pub fn is_field_emit(state: u64) -> bool {
    state >= State::EndField as u64
}

#[derive(Debug, PartialEq, Eq)]
pub enum TransitionAction {
    Discard,
    Output,
    Epsilon,
}

pub fn transition(config: &Config, state: State, input: u8) -> (State, TransitionAction) {
    use State::*;
    use TransitionAction::*;
    match state {
        StartRecord => {
            if config.terminator.equals(input) {
                (StartRecord, Discard)
            } else if config.comment == Some(input) {
                (InComment, Discard)
            } else if config.quote == Some(input) {
                (InQuotedFieldDiscard, Discard)
            } else {
                (InField, Epsilon)
            }
        }
        InField => {
            if input == config.delimiter {
                (EndField, Discard)
            } else if config.terminator.equals(input) {
                (EndRecord, Discard)
            } else if config.quote == Some(input) {
                match config.quote_style {
                    QuoteStyle::Strict(false) => (Error, Discard),
                    QuoteStyle::Strict(true) => (InField, Output),
                    QuoteStyle::Toggle => (InQuotedFieldDiscard, Discard),
                }
            } else if config.escape_in_unquoted_fields && config.escape == Some(input) {
                (EscapeSequenceInField, Discard)
            } else {
                (InField, Output)
            }
        }
        InQuotedFieldOutput => {
            if config.quote == Some(input) {
                (AfterQuote, Discard)
            } else if config.escape == Some(input) {
                (EscapeSequenceInQuotedField, Discard)
            } else {
                (InQuotedFieldOutput, Output)
            }
        }
        InQuotedFieldDiscard => (InQuotedFieldOutput, Epsilon),
        EscapeSequenceInField => (InField, Output),
        EscapeSequenceInQuotedField => (InQuotedFieldOutput, Output),
        AfterQuote => {
            if config.double_quote_escape && config.quote == Some(input) {
                (InQuotedFieldOutput, Output)
            } else if config.delimiter == input {
                (EndField, Discard)
            } else if config.terminator.equals(input) {
                (EndRecord, Discard)
            } else if matches!(config.quote_style, QuoteStyle::Strict(_)) {
                (Error, Discard)
            } else {
                (InField, Epsilon)
            }
        }
        InComment => {
            if input == b'\n' {
                (StartRecord, Discard)
            } else {
                (InComment, Discard)
            }
        }
        EndField => {
            if config.quote == Some(input) {
                (InQuotedFieldDiscard, Discard)
            } else {
                (InField, Epsilon)
            }
        }
        EndRecord => (StartRecord, Epsilon),
        Error => (Error, Discard),
    }
}

pub fn transition_next_state(config: &Config, state: State, byte: u8) -> (State, TransitionAction) {
    let mut state = state;
    loop {
        let (next, action) = transition(config, state, byte);
        if action != TransitionAction::Epsilon {
            return (next, action);
        }
        state = next;
    }
}

pub struct ReferenceDfa {
    state: State,
    config: Config,
}

impl Dfa for ReferenceDfa {
    fn new(config: &Config) -> Self {
        Self {
            state: State::StartRecord,
            config: config.clone(),
        }
    }

    #[inline]
    fn run(&mut self, buffer: &mut [u8], cursor: &mut Cursor, field_ends: &mut [usize]) -> DfaResult {
        let mut state = self.state;
        let mut read = cursor.read;
        let mut write = cursor.write;
        let mut count = cursor.field_count;
        let record_start = cursor.record_start;

        let result = loop {
            if read >= buffer.len() {
                break DfaResult::Refill;
            }

            let (next, action) = transition(&self.config, state, buffer[read]);
            state = next;

            if state == State::Error {
                break DfaResult::Error;
            }

            // TODO: is this or branch faster?
            buffer[write] = buffer[read];
            write += (action == TransitionAction::Output) as usize;
            read += (action != TransitionAction::Epsilon) as usize;

            if state == State::EndField || state == State::EndRecord {
                field_ends[count] = write - record_start;
                count += 1;

                if state == State::EndRecord {
                    break DfaResult::EndRecord;
                }

                if count == field_ends.len() {
                    break DfaResult::FieldEndsFull;
                }
            }
        };

        self.state = state;
        cursor.read = read;
        cursor.write = write;
        cursor.field_count = count;
        result
    }

    #[inline]
    fn finish(&self, config: &Config) -> DfaResult {
        let unterminated_quote = matches!(
            self.state,
            State::InQuotedFieldOutput
                | State::InQuotedFieldDiscard
                | State::EscapeSequenceInQuotedField
        );

        if unterminated_quote && matches!(config.quote_style, QuoteStyle::Strict(_)) {
            DfaResult::Error
        } else {
            DfaResult::EndRecord
        }
    }
}
