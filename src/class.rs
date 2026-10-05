use crate::{config::{Config, RecordTerminator}, dfa::Dfa, reference::{State, transition_next_state}};

const NUM_EQUIVALENCE_CLASSES: usize =  7;

pub struct ClassDfa {
    transition_table: [[u8; 11]; NUM_EQUIVALENCE_CLASSES],
    class_table: [u8; 256],
    class_index: u8,
}

fn state_to_index(state: State) -> u8 {
    match state {
        State::StartRecord => 2,
        State::InField => 0,
        State::InQuotedFieldOutput => 1,
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


impl ClassDfa {
    // each equivalence class besides the non structural bytes has only one byte per class
    // TODO: does it make sense to have \r and \n in the same class in CRLF mode (if comments are
    // disabled)
    fn add_equivalence_class(&mut self, b: u8) {
        // class indices start 1 since the default class for all non structural bytes is 0
        self.class_index += 1;

        if self.class_index >= NUM_EQUIVALENCE_CLASSES as u8 {
            panic!("too many equivalence classes");
        }

        self.class_table[b as usize] = self.class_index;
    }

    fn build_equivalence_class_table(&mut self, config: &Config) {
        match config.terminator {
            RecordTerminator::CRLF => {
                self.add_equivalence_class(b'\n');
                self.add_equivalence_class(b'\r');
            },
            RecordTerminator::Byte(b) => self.add_equivalence_class(b),
        }

        self.add_equivalence_class(config.delimiter);

        if let Some(b) = config.quote {
            self.add_equivalence_class(b);
        }

        if let Some(b) = config.escape {
            self.add_equivalence_class(b);
        }

        if let Some(b) = config.comment {
            self.add_equivalence_class(b);
            if let RecordTerminator::Byte(b) = config.terminator && b != b'\n' {
                self.add_equivalence_class(b'\n');
            }
        }
    }
    
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
                let (next_state, _) = transition_next_state(config, state, b);
                let class = self.class_table[b as usize];
                // TODO: debug_assert that no conflicting entries are written to catch errors in
                // the equivalence class table
                self.transition_table[class as usize][state_to_index(state) as usize] = state_to_index(next_state);
            }
        }
    }
}

impl Dfa for ClassDfa {
    type State = usize;

    fn new(config: &Config) -> Self {
        let mut dfa = Self {
            transition_table: [[0; 11]; NUM_EQUIVALENCE_CLASSES],
            class_table: [0; 256],
            class_index: 0,
        };

        dfa.build_equivalence_class_table(config);
        dfa.build_transition_table(config);

        dfa
    }

    #[inline(always)]
    fn start_state(&self) -> Self::State {
        state_to_index(State::StartRecord) as usize
    }

    #[inline(always)]
    fn transition(&self, state: Self::State, byte: u8) -> Self::State {
        let class = self.class_table[byte as usize];
        self.transition_table[class as usize][state] as usize
    }

    #[inline(always)]
    fn is_output(&self, state: Self::State) -> bool {
        is_output(state)
    }

    #[inline(always)]
    fn is_discard(&self, state: Self::State) -> usize {
        !is_output(state) as usize
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

