use crate::{
    config::Config, dfa::Dfa, reference::{STATE_MASK, State, is_field_emit, is_output, transition_next_state},
};

pub struct BitshiftDfa {
    table: [u64; 256],
}

impl BitshiftDfa {
    fn build_transition_table(&mut self, config: &Config) {
        use State::*;
        // Error does not fit and needs to be handled with a branch
        // TODO: its possible to have error be a proper state by reversing the order of the
        // states so that error can loop to itself because even with the limited bits
        // available it can still transition to the first few states.
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
                row |= (transition_next_state(config, state, b).0 as u64) << (state as u32);
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
        let mut dfa = Self {
            table: [0; 256],
        };
        dfa.build_transition_table(config);
        dfa
    }

    #[inline(always)]
    fn start_state(&self) -> Self::State {
        (State::StartRecord as u64, State::StartRecord as u64)
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
        state >= State::EndRecord as u64
    }

    #[inline(always)]
    fn is_error(&self, (_, state): Self::State) -> bool {
        state == State::Error as u64
    }

    #[inline(always)]
    fn in_quoted_field(&self, (_, state): Self::State) -> bool {
        state == State::InQuotedFieldOutput as u64 || state == State::InQuotedFieldDiscard as u64
    }

    #[inline(always)]
    fn in_escape_sequence(&self, (_, state): Self::State) -> bool {
        state == State::EscapeSequenceInField as u64 || state == State::EscapeSequenceInQuotedField as u64
    }
}
