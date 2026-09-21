use crate::{
    config::{Config, QuoteStyle}, parser::{Dfa, DfaResult}, reference::{State, STATE_MASK, is_field_emit, is_output, next_state},
};

pub struct BitshiftDfa {
    table: [u64; 256],
    state: u64,
}

impl BitshiftDfa {
    fn build_transition_table(&mut self, config: &Config) {
        use State::*;
        // Error does not fit and needs to be handled with a branch
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
                row |= (next_state(config, state, b) as u64) << (state as u32);
            }

            self.table[b as usize] = row;
        }
    }
}

impl Dfa for BitshiftDfa {
    fn new(config: &Config) -> Self {
        let mut dfa = Self {
            state: State::StartRecord as u64,
            table: [0; 256],
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
        let mut unmasked_state = self.state;
        let mut read = *read_index;
        let mut write = *write_index;

        let result = loop {
            if read >= buffer.len() {
                break DfaResult::Refill;
            }

            let byte = buffer[read];
            unmasked_state = self.table[byte as usize].wrapping_shr(unmasked_state as u32);
            // keep the masked state in a seperate register so its off the loop-carried chain
            state = unmasked_state & STATE_MASK;

            buffer[write] = byte;
            write += is_output(state) as usize;
            read += 1;

            if is_field_emit(state) {
                // TODO: also error when calling read_record/run again when already in the error
                // state.
                // its possible to have error be a proper state by reversing the order of the
                // states so that error can loop to itself because even with the limited bits
                // available it can still transition to the first few states.
                if state == State::Error as u64 {
                    // report the byte where the error occured which was not consumed
                    read -= 1;
                    break DfaResult::Error;
                }

                // TODO: also keep record start locally so its not loaded on every field emit
                field_ends.push(write - *record_start);

                if state == State::EndRecord as u64 {
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
        let unterminated_quote = self.state == State::InQuotedFieldOutput as u64
            || self.state == State::InQuotedFieldDiscard as u64
            || self.state == State::EscapeSequenceInQuotedField as u64;

        if unterminated_quote && matches!(config.quote_style, QuoteStyle::Strict(_)) {
            DfaResult::Error
        } else {
            DfaResult::EndRecord
        }
    }
}
