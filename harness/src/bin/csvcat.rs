//! Print the records of a file as one parser sees them, one line per record.
//!
//! Fields are separated by ` | ` and shown with non-printable bytes escaped, so empty fields,
//! trailing whitespace and embedded terminators are visible.

use std::io::{self, Write};

use clap::Parser as _;
use harness::{
    inputs::InputArgs,
    parsers::{self, Options},
};

#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    inputs: InputArgs,
    /// The parser to use.
    #[arg(long, short, default_value = parsers::REFERENCE)]
    parser: String,
    /// Stop after this many records.
    #[arg(long, short = 'n')]
    limit: Option<usize>,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let selection = args.inputs.resolve()?;
    for skipped in &selection.skipped {
        eprintln!("skipped {}: {}", skipped.id, skipped.reason);
    }
    let parser = parsers::all()
        .into_iter()
        .find(|p| p.name == args.parser)
        .ok_or_else(|| format!("unknown parser {:?}", args.parser))?;

    let mut out = io::BufWriter::new(io::stdout().lock());
    for input in &selection.inputs {
        eprintln!("{} ({:?}): {}", input.label(), input.dialect_source, input.dialect);
        (parser.supports)(&input.dialect).map_err(|reason| format!("{} cannot parse this dialect: {reason}", parser.name))?;
        let data = std::fs::read(&input.path).map_err(|e| format!("{}: {e}", input.path.display()))?;

        for record in (parser.records)(&input.dialect, parsers::without_bom(&data), &Options::default()).take(args.limit.unwrap_or(usize::MAX)) {
            let line = match record {
                Ok(record) => {
                    let fields: Vec<String> = record.fields.iter().map(|f| f.escape_ascii().to_string()).collect();
                    fields.join(" | ")
                }
                Err(error) => format!("<{error}>"),
            };
            if writeln!(out, "{line}").is_err() {
                return Ok(()); // closed pipe, e.g. `| head`
            }
        }
    }
    Ok(())
}
