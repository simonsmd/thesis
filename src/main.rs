use std::{error::Error, fs::File, io::BufReader};

fn main() -> Result<(), Box<dyn Error>> {
    let file = File::open(std::env::args().nth(1).unwrap())?;
    let reader = BufReader::new(file);

    let mut parser = csv_parser::Parser::new(reader);

    while let Some(record) = parser.read_record()? {
        let fields: Vec<&str> = record.iter_str().collect::<Result<_, _>>()?;
        println!("{}", fields.join(", "));
    }

    Ok(())
}
