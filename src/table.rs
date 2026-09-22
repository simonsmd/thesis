use crate::{config::{Config, QuoteStyle}, parser::{Dfa, DfaResult}, reference::{State, TransitionAction, transition_next_state}};

pub struct TableDfa {
    table: [[Transition; 11]; 256],
    state: u8,
}

#[derive(Clone, Copy, Default)]
struct Transition(u8);

impl Transition {
    fn new(state: State, action: TransitionAction) -> Self {
        Self(state_to_index(state) | ((action == TransitionAction::Output) as u8) << 7)
    }

    fn state_index(&self) -> u8 {
        self.0 & 0x7f
    }

    fn has_output(&self) -> bool {
        self.0 & (1 << 7) != 0
    }
}

fn state_to_index(state: State) -> u8 {
    match state {
        State::StartRecord => 0,
        State::InField => 1,
        State::InQuotedFieldOutput => 2,
        State::InQuotedFieldDiscard => 3,
        State::EscapeSequenceInField => 4,
        State::EscapeSequenceInQuotedField => 5,
        State::AfterQuote => 6,
        State::InComment => 7,
        State::EndField => 8,
        State::EndRecord => 9,
        State::Error => 10,
    }
}

pub fn is_field_emit(state: u8) -> bool {
    state >= state_to_index(State::EndField)
}


impl TableDfa {
    fn build_transition_table(&mut self, config: &Config) {
        use State::*;
        const STATES: [State; 11] = [
            StartRecord,
            InField,
            InQuotedFieldOutput,
            InQuotedFieldDiscard,
            EscapeSequenceInField,
            EscapeSequenceInQuotedField,
            AfterQuote,
            InComment,
            EndField,
            EndRecord,
            Error,
        ];

        for b in 0..=255_u8 {
            for state in STATES {
                let (next_state, action) = transition_next_state(config, state, b);
                self.table[b as usize][state_to_index(state) as usize] = Transition::new(next_state, action);
            }
        }
    }
}

impl Dfa for TableDfa {
    fn new(config: &Config) -> Self {
        let mut dfa = Self {
            table: [[Transition::default(); 11]; 256],
            state: state_to_index(State::StartRecord)
        };

        dfa.build_transition_table(config);

        dfa
    }

    fn run(
        &mut self,
        buffer: &mut [u8],
        read_index: &mut usize,
        write_index: &mut usize,
        record_start: &mut usize,
        field_ends: &mut Vec<usize>,
    ) -> DfaResult {
        let mut state = self.state;
        let mut read = *read_index;
        let mut write = *write_index;

        let result = loop {
            if read >= buffer.len() {
                break DfaResult::Refill;
            }

            let byte = buffer[read];
            let transition = self.table[byte as usize][state as usize];
            state = transition.state_index();

            buffer[write] = byte;
            write += transition.has_output() as usize;
            read += 1;

            if is_field_emit(state) {
                // TODO: also error when calling read_record/run again when already in the error
                // state.
                if state == state_to_index(State::Error) {
                    // report the byte where the error occured which was not consumed
                    read -= 1;
                    break DfaResult::Error;
                }

                // TODO: also keep record start locally so its not loaded on every field emit
                field_ends.push(write - *record_start);

                if state == state_to_index(State::EndRecord) {
                    break DfaResult::EndRecord;
                }
            }
        };

        self.state = state;
        *read_index = read;
        *write_index = write;
        result
    }

    fn finish(&self, config: &Config) -> DfaResult {
        let unterminated_quote = self.state == state_to_index(State::InQuotedFieldOutput)
            || self.state == state_to_index(State::InQuotedFieldDiscard)
            || self.state == state_to_index(State::EscapeSequenceInQuotedField);

        if unterminated_quote && matches!(config.quote_style, QuoteStyle::Strict(_)) {
            DfaResult::Error
        } else {
            DfaResult::EndRecord
        }
    }
}
