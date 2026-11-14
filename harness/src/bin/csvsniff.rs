//! Guess the dialect of one CSV file and describe its content, as one line of JSON.
//!
//! The input is a file, or standard input if no path is given. `scripts/csvscrape` runs this for
//! every file head it probes and for every file of a dataset.
//!
//! The output has `dialect` (in the form the dataset files use), `detection` (how sure the guess
//! is) and `stats` (what the file contains when parsed with that dialect).

use std::{io::Read, path::PathBuf};

use clap::Parser as _;
use harness::{dialect::Dialect, encoding::Data, sniff};
use serde_json::json;

#[derive(clap::Parser)]
struct Args {
    /// The file to look at. Standard input if left out.
    path: Option<PathBuf>,
    /// Encoding of the input as a Python codec name, e.g. `cp1252` or `utf-16-le`. The input is
    /// converted to UTF-8 first. Left out, the bytes are used as they are.
    #[arg(long)]
    encoding: Option<String>,
    /// Size of the whole file in bytes, if the input is only its start. The cut off last record
    /// of such a sample is not counted.
    #[arg(long)]
    size: Option<u64>,
    /// Do not guess the dialect, describe the input with this one: a JSON object as in the
    /// dataset files.
    #[arg(long)]
    dialect: Option<String>,
    /// Also count records separately for this many equally sized pieces of the input, to see
    /// whether the file changes along the way.
    #[arg(long, default_value_t = 0)]
    parts: usize,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let (data, input_bytes) = match &args.path {
        Some(path) => {
            let size = path
                .metadata()
                .map_err(|e| format!("{}: {e}", path.display()))?
                .len();
            (Data::map(path, args.encoding.as_deref())?, size)
        }
        None => {
            let mut raw = Vec::new();
            std::io::stdin()
                .read_to_end(&mut raw)
                .map_err(|e| e.to_string())?;
            let size = raw.len() as u64;
            (Data::from_bytes(raw, args.encoding.as_deref())?, size)
        }
    };
    let complete = args.size.is_none_or(|size| size <= input_bytes);

    let (dialect, detection, seen) = match &args.dialect {
        Some(given) => {
            let dialect: Dialect =
                serde_json::from_str(given).map_err(|e| format!("--dialect: {e}"))?;
            dialect.validate().map_err(|e| format!("--dialect: {e}"))?;
            let seen = sniff::observe(&dialect, &data, complete, true, args.parts);
            (dialect, None, seen)
        }
        None => {
            let (detection, seen) = sniff::sniff(&data, complete, args.parts);
            (detection.dialect.clone(), Some(detection.report()), seen)
        }
    };

    println!(
        "{}",
        json!({
            "version": sniff::VERSION,
            // what was parsed: the input as UTF-8, without a byte order mark
            "bytes": data.len(),
            "complete": complete,
            "dialect": dialect,
            "detection": detection,
            "stats": seen.report(),
        })
    );
    Ok(())
}
