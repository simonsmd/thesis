use crate::{
    config::Config,
    dfa::Dfa,
    reference::{State, transition_next_state},
};

pub struct BitshiftDfa {
    table: [u64; 256],
}

const fn state_to_offset(state: State) -> u64 {
    match state {
        State::InField => 1,
        State::InQuotedFieldOutput => 7,
        State::StartRecord => 14,
        State::InQuotedFieldDiscard => 20,
        State::EscapeSequenceInField => 26,
        State::EscapeSequenceInQuotedField => 32,
        State::AfterQuote => 38,
        State::InComment => 44,
        State::EndField => 50,
        State::EndRecord => 56,
        State::Error => 62,
    }
}

/// 6 bits per state so the masking of the shift amount is done automatically by the shift
/// instructions
const STATE_MASK: u64 = 0b11_1111;

pub fn is_output(state: u64) -> bool {
    state <= state_to_offset(State::InQuotedFieldOutput)
}

pub fn is_field_emit(state: u64) -> bool {
    state >= state_to_offset(State::EndField)
}

impl BitshiftDfa {
    fn build_transition_table(&mut self, config: &Config) {
        use State::*;
        // Error only has a state representation/number in the table but no valid entry in the
        // transition table for the next state because only 10 states fit into the 64 bit wide rows
        // of the bitshift transition table (6 bits per state). This does not matter because the
        // error state can be considered a field/record end since it aborts parsing anyways and so
        // the error check on the fast path is free (included in the field end check). Only on a
        // record end it costs one well predicted branch that is needed anyways for reporting the
        // correct error position without backtracking as well as one check if the parser is in the
        // error state when starting a new record which is also already there to check for previous
        // IO errors.
        const STATES: [State; 10] = [
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
        ];

        for b in 0..=255_u8 {
            let mut row: u64 = 0;

            for state in STATES {
                row |= state_to_offset(transition_next_state(config, state, b).0)
                    << state_to_offset(state);
            }

            self.table[b as usize] = row;
        }
    }
}

impl Dfa for BitshiftDfa {
    /// `(unmasked, masked)`: the unmasked state feeds the next shift so the loop-carried chain is
    /// a single `shrx`, the masked copy is what the state checks compare against.
    type State = (u64, u64);

    fn new(config: &Config) -> Self {
        let mut dfa = Self { table: [0; 256] };
        dfa.build_transition_table(config);
        dfa
    }

    #[inline(always)]
    fn start_state(&self) -> Self::State {
        (
            state_to_offset(State::StartRecord),
            state_to_offset(State::StartRecord),
        )
    }

    #[inline(always)]
    fn transition(&self, (unmasked, _): Self::State, byte: u8) -> Self::State {
        let next = self.table[byte as usize].wrapping_shr(unmasked as u32);
        (next, next & STATE_MASK)
    }

    #[inline(always)]
    fn is_output(&self, (_, masked): Self::State) -> bool {
        is_output(masked)
    }

    #[inline(always)]
    fn is_discard(&self, (unmasked, _): Self::State) -> usize {
        !unmasked as usize
    }

    #[inline(always)]
    fn is_field_end(&self, (_, state): Self::State) -> bool {
        is_field_emit(state)
    }

    #[inline(always)]
    fn is_record_end(&self, (_, state): Self::State) -> bool {
        state >= state_to_offset(State::EndRecord)
    }

    #[inline(always)]
    fn is_error(&self, (_, state): Self::State) -> bool {
        state == state_to_offset(State::Error)
    }

    #[inline(always)]
    fn is_record_start(&self, (_, state): Self::State) -> bool {
        state == state_to_offset(State::StartRecord) || state == state_to_offset(State::InComment)
    }

    #[inline(always)]
    fn in_quoted_field(&self, (_, state): Self::State) -> bool {
        state == state_to_offset(State::InQuotedFieldOutput)
            || state == state_to_offset(State::InQuotedFieldDiscard)
    }

    #[inline(always)]
    fn in_escape_sequence(&self, (_, state): Self::State) -> bool {
        state == state_to_offset(State::EscapeSequenceInField)
            || state == state_to_offset(State::EscapeSequenceInQuotedField)
    }
}
