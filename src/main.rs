use std::{error::Error, fs::File, io::BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(std::env::args().nth(1).unwrap())?;
    let mut reader = BufReader::new(file);

    let mut dfa = csv_parser::Dfa::default();
    let mut record = csv_parser::Record::new();

    while dfa.read_record(&mut reader, &mut record)? {
        let fields: Vec<&str> = record.iter_str().collect::<Result<_, _>>()?;
        println!("{}", fields.join(", "));
    }

    Ok(())
}
