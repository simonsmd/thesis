pub mod config;
pub mod dfa;
pub mod parser;
mod buffer;
pub mod reference;
pub mod table;
pub mod bitshift;
pub mod class;
pub mod skip;

pub use buffer::DEFAULT_CAPACITY;
