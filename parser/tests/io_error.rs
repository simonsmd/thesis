//! An IO error stops the parser: it is reported again on every later call.

use std::io::{self, Read};

use csv_parser::{
    parser::{CompactionParser, Error, ZeroCopyParser},
    table::TableDfa,
};

/// Yields `data` once and fails on every read after that.
struct Failing(&'static [u8]);

impl Read for Failing {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.0.is_empty() {
            return Err(io::Error::other("broken"));
        }
        let n = self.0.read(buf)?;
        Ok(n)
    }
}

#[test]
fn io_error_stops_the_parser() {
    let mut compaction = CompactionParser::<_, TableDfa>::new(Failing(b"a,b\nc,"));
    assert!(compaction.read_record().unwrap().is_some());
    assert!(matches!(compaction.read_record(), Err(Error::IO(_))));
    assert!(matches!(compaction.read_record(), Err(Error::IO(_))));

    let mut zerocopy = ZeroCopyParser::<_, TableDfa>::new(Failing(b"a,b\nc,"));
    assert!(zerocopy.read_record().unwrap().is_some());
    assert!(matches!(zerocopy.read_record(), Err(Error::IO(_))));
    assert!(matches!(zerocopy.read_record(), Err(Error::IO(_))));

    // a reset makes the parser usable again
    zerocopy.reset(Failing(b"d\n"));
    assert!(zerocopy.read_record().unwrap().is_some());
}
