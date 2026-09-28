use crate::{config::Config, parser::Dfa, reference::{State, TransitionAction, transition_next_state}};

pub struct TableDfa {
    /// Next state index for every input byte and current state.
    ///
    /// The entries hold nothing but the index, so a loaded entry is the index of the next lookup
    /// as is and the loop-carried chain is a single load. Whether a transition outputs its byte
    /// follows from the state it enters (see [`is_output`]).
    table: [[u8; 11]; 256],
}

/// The output states come first so [`is_output`] is a single compare, and the states that end a
/// field come last so [`is_field_emit`] is one too.
fn state_to_index(state: State) -> u8 {
    match state {
        State::InField => 0,
        State::InQuotedFieldOutput => 1,
        State::StartRecord => 2,
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

pub fn is_output(state: usize) -> bool {
    state <= state_to_index(State::InQuotedFieldOutput) as usize
}

pub fn is_field_emit(state: usize) -> bool {
    state >= state_to_index(State::EndField) as usize
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
                let next = state_to_index(next_state);
                debug_assert_eq!(
                    action == TransitionAction::Output,
                    is_output(next as usize),
                    "output must follow from the entered state: {state:?} --{b:#04x}--> {next_state:?}",
                );
                self.table[b as usize][state_to_index(state) as usize] = next;
            }
        }
    }
}

impl Dfa for TableDfa {
    type State = usize;

    fn new(config: &Config) -> Self {
        let mut dfa = Self {
            table: [[0; 11]; 256],
        };

        dfa.build_transition_table(config);

        dfa
    }

    #[inline(always)]
    fn start_state(&self) -> Self::State {
        state_to_index(State::StartRecord) as usize
    }

    #[inline(always)]
    fn transition(&self, state: Self::State, byte: u8) -> Self::State {
        self.table[byte as usize][state] as usize
    }

    #[inline(always)]
    fn is_output(&self, state: Self::State) -> bool {
        is_output(state)
    }

    #[inline(always)]
    fn is_field_end(&self, state: Self::State) -> bool {
        is_field_emit(state)
    }

    #[inline(always)]
    fn is_record_end(&self, state: Self::State) -> bool {
        state == state_to_index(State::EndRecord) as usize
    }

    #[inline(always)]
    fn is_error(&self, state: Self::State) -> bool {
        state == state_to_index(State::Error) as usize
    }

    #[inline(always)]
    fn in_quoted_field(&self, state: Self::State) -> bool {
        state == state_to_index(State::InQuotedFieldOutput) as usize || state == state_to_index(State::InQuotedFieldDiscard) as usize
    }

    #[inline(always)]
    fn in_escape_sequence(&self, state: Self::State) -> bool {
        state == state_to_index(State::EscapeSequenceInField) as usize || state == state_to_index(State::EscapeSequenceInQuotedField) as usize
    }
}
