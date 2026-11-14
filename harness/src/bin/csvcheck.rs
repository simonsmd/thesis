//! Correctness on real files: every selected parser against the reference parser.
//!
//! Our parsers must agree with the reference exactly, at every `--capacity`. rust-csv is held to
//! the same standard, so its known semantic differences (it never errors) show up as mismatches.
//! Exits with status 1 if anything disagrees.

use std::{
    fs::File,
    process::ExitCode,
    sync::atomic::{AtomicUsize, Ordering},
};

use clap::Parser as _;
use harness::{
    check::{self, Report, Verdict},
    inputs::{Input, InputArgs, Size},
    parsers::{self, Options},
    system,
};
use rayon::prelude::*;
use serde_json::json;

#[derive(clap::Parser)]
struct Args {
    #[command(flatten)]
    inputs: InputArgs,
    /// Parsers to check, as globs (repeatable, default all).
    #[arg(long, short)]
    parser: Vec<String>,
    /// Parsers to leave out, as globs (repeatable).
    #[arg(long, short = 'x')]
    exclude: Vec<String>,
    /// Buffer read sizes to run every parser with (repeatable). Small ones put refills in the
    /// middle of records.
    #[arg(long, default_values = ["64K", "1K"])]
    capacity: Vec<Size>,
    /// Files checked in parallel (default: all cores).
    #[arg(long, short)]
    jobs: Option<usize>,
    /// Also write every outcome as JSON to this file.
    #[arg(long, short)]
    output: Option<std::path::PathBuf>,
    /// Report each file, not only those with mismatches.
    #[arg(long, short)]
    verbose: bool,
}

fn main() -> Result<ExitCode, String> {
    let args = Args::parse();
    let selection = args.inputs.resolve()?;
    let parsers = parsers::select(&args.parser, &args.exclude)?;
    let reference = parsers::reference();
    let options: Vec<Options> = args.capacity.iter().map(|c| Options { capacity: c.0 as usize, fields: None }).collect();

    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global().map_err(|e| e.to_string())?;
    }
    for skipped in &selection.skipped {
        println!("skipped {}: {}", skipped.id, skipped.reason);
    }
    eprintln!(
        "checking {} files ({}) with {} parsers at {} capacities",
        selection.inputs.len(),
        Size(selection.inputs.iter().map(|i| i.size).sum()),
        parsers.len(),
        options.len()
    );

    let done = AtomicUsize::new(0);
    let total = selection.inputs.len();
    let reports: Vec<(&Input, Result<Report, String>)> = selection
        .inputs
        .par_iter()
        .map(|input| {
            let report = check_file(input, &reference, &parsers, &options);
            let n = done.fetch_add(1, Ordering::Relaxed) + 1;
            // one print per file so that parallel output does not interleave
            print!("{}", render(n, total, input, &report, args.verbose));
            (input, report)
        })
        .collect();

    let mut mismatched_files = 0;
    let mut counts: std::collections::BTreeMap<&str, [usize; 3]> = Default::default();
    for (_, report) in &reports {
        let Ok(report) = report else { continue };
        mismatched_files += (report.mismatches().count() > 0) as usize;
        for o in &report.outcomes {
            let i = match o.verdict {
                Verdict::Agree => 0,
                Verdict::Mismatch { .. } => 1,
                Verdict::Unsupported { .. } => 2,
            };
            counts.entry(o.parser).or_default()[i] += 1;
        }
    }
    let unreadable = reports.iter().filter(|(_, r)| r.is_err()).count();

    println!("\n{:<40} {:>8} {:>9} {:>12}", "parser", "agree", "mismatch", "unsupported");
    for parser in &parsers {
        if let Some([agree, mismatch, unsupported]) = counts.get(parser.name) {
            println!("{:<40} {agree:>8} {mismatch:>9} {unsupported:>12}", parser.name);
        }
    }
    println!(
        "\n{} files: {} with mismatches, {} unreadable, {} skipped",
        total,
        mismatched_files,
        unreadable,
        selection.skipped.len()
    );

    if let Some(path) = &args.output {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let files: Vec<_> = reports
            .iter()
            .map(|(input, report)| match report {
                Ok(report) => json!({"id": input.id, "path": input.path, "dialect": input.dialect, "report": report}),
                Err(error) => json!({"id": input.id, "path": input.path, "error": error}),
            })
            .collect();
        let value = json!({
            "started": system::timestamp(),
            "args": std::env::args().collect::<Vec<_>>(),
            "build": system::build(),
            "datasets": selection.datasets,
            "overrides_hash": selection.overrides_hash,
            "skipped": selection.skipped,
            "files": files,
        });
        serde_json::to_writer_pretty(file, &value).map_err(|e| e.to_string())?;
    }

    Ok(if mismatched_files > 0 || unreadable > 0 { ExitCode::FAILURE } else { ExitCode::SUCCESS })
}

fn check_file(input: &Input, reference: &parsers::Parser, parsers: &[parsers::Parser], options: &[Options]) -> Result<Report, String> {
    let data = input.map()?;
    Ok(check::check(reference, parsers, &input.dialect, &data, options))
}

fn render(n: usize, total: usize, input: &Input, report: &Result<Report, String>, verbose: bool) -> String {
    let head = format!("[{n}/{total}] {} {}", input.label(), Size(input.size));
    let report = match report {
        Ok(report) => report,
        Err(error) => return format!("{head}: cannot read: {error}\n"),
    };
    let mismatches: Vec<_> = report.mismatches().collect();
    if mismatches.is_empty() {
        return if verbose { format!("{head}: ok\n") } else { String::new() };
    }

    let mut out = format!("{head}: {} mismatches\n  dialect: {}\n  path: {}\n", mismatches.len(), input.dialect, input.path.display());
    if let Some(error) = report.reference.error {
        out += &format!("  reference stops with {error}\n");
    }
    for o in mismatches {
        if let Verdict::Mismatch { difference, .. } = &o.verdict {
            out += &format!("  {} (capacity {}): {}\n", o.parser, o.options.capacity, difference.replace('\n', "\n    "));
        }
    }
    out
}
