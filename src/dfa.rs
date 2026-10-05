use crate::config::Config;

pub trait Dfa {
    /// Type for the DFA state.
    ///
    /// Some implementations use tuples (e.g. masked and unmasked state for the bitshift implementation)
    /// to optimize dependency chains in the core loops by splitting the registers used for
    /// computing the next state and checking the output bit and special states.
    ///
    /// Its best to store the state in types the size of a register. Smaller types like u8
    /// introduce a data dependency between records because the state initialization at the start
    /// of `read_record` only overwrites the register paritally. This means multiple records cannot
    /// be processed in parallel using out-of-order execution even if the record end branch is
    /// predicted correctly.
    type State: Copy;

    /// Create an instance of this Dfa with the provided configuration
    ///
    /// Used to initialize state like transition tables.
    fn new(config: &Config) -> Self;

    fn start_state(&self) -> Self::State;

    fn transition(&self, state: Self::State, byte: u8) -> Self::State;

    fn is_output(&self, state: Self::State) -> bool;

    /// Inverse of `is_output`
    ///
    /// This function is used to update the register that tracks if a field has not started yet
    /// which is then used to increment the field_start and write indices until the first
    /// emitted byte in a field in the `ZeropcopyParser` implementation.
    ///
    /// The `usize` returned from this function is used to mask the `field_not_started` variable
    /// using AND for each byte. This expression is implemented per DFA in order to make the
    /// compiler generate the optimal assembly for each DFA based on its state representation.
    fn is_discard(&self, state: Self::State) -> usize;

    /// Check if the current state marks the end of a field
    ///
    /// A record end or error is also the end of a field.
    fn is_field_end(&self, state: Self::State) -> bool;

    fn is_record_end(&self, state: Self::State) -> bool;

    fn is_error(&self, state: Self::State) -> bool;

    fn in_quoted_field(&self, state: Self::State) -> bool;

    fn in_escape_sequence(&self, state: Self::State) -> bool;
}
