use std::{error::Error, fs::File, io::BufReader};
use csv_parser::reference::ReferenceDfa;

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(std::env::args().nth(1).unwrap())?;
    let reader = BufReader::new(file);

    let mut parser = csv_parser::parser::CompactionParser::<_, ReferenceDfa>::new(reader);

    while let Some(record) = parser.read_record()? {
        let fields: Vec<&str> = record.iter_str().collect::<Result<_, _>>()?;
        println!("{}", fields.join(", "));
    }

    Ok(())
}
