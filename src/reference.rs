use crate::{
    config::{Config, QuoteStyle}, dfa::Dfa,
};

// TODO: should be copy?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum State {
    StartRecord = 14,
    InField = 1,
    InQuotedFieldOutput = 7,
    InQuotedFieldDiscard = 20,
    EscapeSequenceInField = 26,
    EscapeSequenceInQuotedField = 32,
    AfterQuote = 38,
    InComment = 44,
    EndField = 50,
    EndRecord = 56,
    Error = 62,
}

/// 6 Bits per state so the masking for the bitshift dfa is done automatically by the shift instructions
pub const STATE_MASK: u64 = 0b11_1111;

pub fn is_output(state: u64) -> bool {
    state <= State::InQuotedFieldOutput as u64
}

pub fn is_field_emit(state: u64) -> bool {
    state >= State::EndField as u64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    config: Config,
}

impl Dfa for ReferenceDfa {
    type State = (State, TransitionAction);

    fn new(config: &Config) -> Self {
        Self {
            config: config.clone(),
        }
    }

    #[inline(always)]
    fn start_state(&self) -> Self::State {
        (State::StartRecord, TransitionAction::Discard)
    }

    #[inline(always)]
    fn transition(&self, (state, _): Self::State, byte: u8) -> Self::State {
        transition_next_state(&self.config, state, byte)
    }

    #[inline(always)]
    fn is_output(&self, (_, action): Self::State) -> bool {
        action == TransitionAction::Output
    }

    #[inline(always)]
    fn is_discard(&self, (_, action): Self::State) -> usize {
        (action == TransitionAction::Discard) as usize
    }

    #[inline(always)]
    fn is_field_end(&self, (state, _): Self::State) -> bool {
        matches!(state, State::EndField | State::EndRecord | State::Error)
    }

    #[inline(always)]
    fn is_record_end(&self, (state, _): Self::State) -> bool {
        matches!(state, State::EndRecord | State::Error)
    }

    #[inline(always)]
    fn is_error(&self, (state, _): Self::State) -> bool {
        state == State::Error
    }

    #[inline(always)]
    fn in_quoted_field(&self, (state, _): Self::State) -> bool {
        matches!(state, State::InQuotedFieldOutput | State::InQuotedFieldDiscard)
    }

    #[inline(always)]
    fn in_escape_sequence(&self, (state, _): Self::State) -> bool {
        matches!(state, State::EscapeSequenceInField | State::EscapeSequenceInQuotedField)
    }
}
