//! Throughput benchmarks over datasets and files.
//!
//! For every file, each selected parser runs to completion before the next one starts: warm up,
//! then take samples back to back, so caches and predictors are warm for that parser when it is
//! measured. The parser is built once per file and reset before every parse, so building its DFA
//! and buffers is not measured.
//!
//! Warming up parses the whole file in memory until `--warmup` has passed, at least once. A
//! sample is `iterations` consecutive parses, enough that it takes at least `--min-sample-time`.
//!
//! Results are JSON Lines, one object per line and tagged by `type`, appended and flushed as they
//! are measured so that an interrupted run keeps everything up to that point:
//!
//! - `run`: settings, build, machine and dataset hashes (first line)
//! - `skip`: a selected file that cannot be run, and why
//! - `file`: a file about to be measured, with its dialect
//! - `result`: one parser on one file: its statistics, or why it was not measured
//! - `sample`: one measurement
//! - `end`: the run completed

use std::{
    collections::HashMap,
    fs::{self, File},
    hint::black_box,
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use clap::{Parser as _, Subcommand};
use harness::{
    counters::{self, Counters, Usage},
    dataset::ROOT,
    inputs::{InputArgs, Size},
    parsers::{self, Options, Parser, Statistics},
    system,
};
use serde_json::{Value, json};

#[derive(clap::Parser)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Measure parsers on files.
    Run(RunArgs),
    /// Compare two result files per file and parser.
    Compare(CompareArgs),
    /// Show the files a selection resolves to, without running anything.
    List(ListArgs),
    /// Show the parser names, optionally filtered.
    Parsers(ParserArgs),
}

#[derive(clap::Args)]
struct ParserArgs {
    /// Parsers to include, as globs like `compaction/simd*/*/bitshift` (repeatable, default all).
    #[arg(long, short)]
    parser: Vec<String>,
    /// Parsers to leave out, as globs (repeatable).
    #[arg(long, short = 'x')]
    exclude: Vec<String>,
}

#[derive(clap::Args)]
struct RunArgs {
    #[command(flatten)]
    inputs: InputArgs,
    #[command(flatten)]
    parsers: ParserArgs,
    /// Parse unmeasured for at least this long before sampling, per file and parser (ms). At least
    /// one parse is made: on a file that takes longer than this, that one parse is the warmup.
    #[arg(long, default_value_t = 50)]
    warmup: u64,
    /// Measured samples per file and parser.
    #[arg(long, short, default_value_t = 3)]
    samples: usize,
    /// Repeat the parse within a sample until a sample takes at least this long (ms).
    #[arg(long, default_value_t = 20)]
    min_sample_time: u64,
    /// Read size of the parser buffers.
    #[arg(long, default_value = "64K")]
    capacity: Size,
    /// Pin to this CPU. On hybrid CPUs pick a performance core.
    #[arg(long)]
    cpu: Option<usize>,
    /// Do not read performance counters. On hybrid CPUs they need `--cpu`.
    #[arg(long)]
    no_counters: bool,
    /// Result file. Default: `bench-data/<datasets>-<time>.jsonl` in the workspace.
    #[arg(long, short)]
    output: Option<PathBuf>,
    /// Print one line per file and parser while running.
    #[arg(long, short)]
    verbose: bool,
    /// After the run, compare it to this result file like `compare` does. Without a path, the
    /// latest default-named result for the same datasets in `bench-data/`.
    #[arg(long, num_args = 0..=1, require_equals = true, value_name = "PATH")]
    baseline: Option<Option<PathBuf>>,
    /// With `--baseline`, list files whose throughput changed by more than this many percent.
    #[arg(long, default_value_t = 3.0)]
    threshold: f64,
}

#[derive(clap::Args)]
struct CompareArgs {
    base: PathBuf,
    new: PathBuf,
    #[command(flatten)]
    parsers: ParserArgs,
    /// List files whose throughput changed by more than this many percent.
    #[arg(long, default_value_t = 3.0)]
    threshold: f64,
}

#[derive(clap::Args)]
struct ListArgs {
    #[command(flatten)]
    inputs: InputArgs,
}

fn main() -> Result<(), String> {
    match Args::parse().command {
        Command::Run(args) => run(args),
        Command::Compare(args) => compare(args),
        Command::List(args) => list(args),
        Command::Parsers(args) => {
            for parser in parsers::select(&args.parser, &args.exclude)? {
                println!("{}", parser.name);
            }
            Ok(())
        }
    }
}

fn list(args: ListArgs) -> Result<(), String> {
    let selection = args.inputs.resolve()?;
    for input in &selection.inputs {
        println!(
            "{:<12} {:>10}  {:<9} {}",
            input.label(),
            Size(input.size).to_string(),
            format!("{:?}", input.dialect_source).to_lowercase(),
            input.dialect
        );
    }
    for skipped in &selection.skipped {
        println!("{:<12} skipped: {}", short(&skipped.id), skipped.reason);
    }
    let total: u64 = selection.inputs.iter().map(|i| i.size).sum();
    println!(
        "\n{} files, {}; {} skipped",
        selection.inputs.len(),
        Size(total),
        selection.skipped.len()
    );
    Ok(())
}

/// Width of the parser name column: the longest name, at least the header.
fn name_width<'a>(names: impl Iterator<Item = &'a str>) -> usize {
    names.map(str::len).max().unwrap_or(0).max("parser".len())
}

fn short(id: &str) -> &str {
    if harness::dataset::is_content_hash(id) {
        &id[..10]
    } else {
        id
    }
}

/// Appends JSON lines and flushes after each group, so partial runs are usable.
struct Output {
    out: BufWriter<File>,
}

impl Output {
    fn line(&mut self, value: Value) -> Result<(), String> {
        serde_json::to_writer(&mut self.out, &value).map_err(|e| e.to_string())?;
        self.out.write_all(b"\n").map_err(|e| e.to_string())
    }

    fn flush(&mut self) -> Result<(), String> {
        self.out.flush().map_err(|e| e.to_string())
    }
}

fn run(args: RunArgs) -> Result<(), String> {
    let selection = args.inputs.resolve()?;
    let parsers = parsers::select(&args.parsers.parser, &args.parsers.exclude)?;
    let width = name_width(parsers.iter().map(|p| p.name));
    let options = Options {
        capacity: args.capacity.0 as usize,
        fields: None,
    };
    let min_sample_time = Duration::from_millis(args.min_sample_time);

    if let Some(cpu) = args.cpu {
        counters::pin_to(cpu).map_err(|e| format!("pinning to CPU {cpu}: {e}"))?;
    }
    let mut counters = if args.no_counters {
        None
    } else if args.cpu.is_none() && counters::is_hybrid() {
        eprintln!(
            "warning: hybrid CPU without --cpu: the counters would stop on the other core type, measuring time only"
        );
        None
    } else {
        match Counters::open(args.cpu) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("warning: hardware counters unavailable ({e}), measuring time only");
                None
            }
        }
    };

    let label = if selection.datasets.is_empty() {
        "files".to_owned()
    } else {
        selection
            .datasets
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join("+")
    };
    let results_dir = Path::new(ROOT).join("bench-data");
    // loaded before running, so that a bad baseline does not surface only after the whole run
    let baseline = match &args.baseline {
        None => None,
        Some(path) => {
            let path = match path {
                Some(path) => path.clone(),
                None => latest_result(&results_dir, &label)?,
            };
            let medians = load_medians(&path)?;
            eprintln!("baseline {}", path.display());
            Some((path, medians))
        }
    };
    let path = args.output.clone().unwrap_or_else(|| {
        results_dir.join(format!("{label}-{}.jsonl", system::compact_timestamp()))
    });
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = File::create_new(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Output {
        out: BufWriter::new(file),
    };

    let build = system::build();
    if build.git_dirty {
        eprintln!("note: built from uncommitted changes in parser/ or harness/");
    }
    out.line(json!({
        "type": "run",
        "version": 1,
        "started": system::timestamp(),
        "args": std::env::args().collect::<Vec<_>>(),
        "settings": {
            "warmup_ns": args.warmup * 1_000_000,
            "samples": args.samples,
            "min_sample_time_ns": min_sample_time.as_nanos() as u64,
            "options": options,
            "cpu": args.cpu,
            "counters": counters.is_some(),
        },
        "build": build,
        "system": system::system(args.cpu),
        "datasets": selection.datasets.iter().map(|(name, hash)| json!({"name": name, "hash": hash})).collect::<Vec<_>>(),
        "overrides_hash": selection.overrides_hash,
        "parsers": parsers.iter().map(|p| p.name).collect::<Vec<_>>(),
    }))?;
    for skipped in &selection.skipped {
        out.line(json!({"type": "skip", "id": skipped.id, "path": skipped.path, "reason": skipped.reason}))?;
    }
    out.flush()?;
    eprintln!("writing {}", path.display());
    eprintln!(
        "{} files ({}), {} parsers; {} files skipped",
        selection.inputs.len(),
        Size(selection.inputs.iter().map(|i| i.size).sum()),
        parsers.len(),
        selection.skipped.len()
    );

    // per parser, one summary per file, for the table at the end
    let mut summaries: HashMap<&str, Vec<Summary>> = HashMap::new();
    // files where a parser stopped with an error or disagreed with another, for after the table
    let mut problems: Vec<Problems> = Vec::new();

    for (i, input) in selection.inputs.iter().enumerate() {
        let data = input.read()?;
        let data: &[u8] = &data;
        out.line(json!({
            "type": "file",
            "id": input.id,
            "path": input.path,
            "dataset": input.dataset,
            "size": input.size,
            // what the parsers see: the file as UTF-8, without a byte order mark
            "bytes": data.len(),
            "encoding": input.encoding,
            "bucket": input.bucket,
            "dialect": input.dialect,
            "dialect_source": input.dialect_source,
            "dialect_adjusted": input.dialect_adjusted,
        }))?;
        eprintln!(
            "[{}/{}] {} {} {}",
            i + 1,
            selection.inputs.len(),
            input.label(),
            Size(input.size),
            input.dialect
        );

        // Expected parsing result. Set by the first parser
        let mut expected: Option<(&str, Statistics)> = None;
        let mut file_problems = Problems {
            input,
            ours_run: 0,
            errors: Vec::new(),
            disagreements: Vec::new(),
        };

        for parser in &parsers {
            let result = |status: &str| json!({"type": "result", "file": input.id, "parser": parser.name, "status": status});

            if let Err(reason) = (parser.supports)(&input.dialect) {
                let mut r = result("unsupported");
                r["reason"] = reason.into();
                out.line(r)?;
                continue;
            }

            file_problems.ours_run += parser.is_ours() as usize;
            let measured = measure(
                parser,
                input,
                data,
                &options,
                &args,
                min_sample_time,
                counters.as_mut(),
            )?;
            let (statistics, iterations, samples) = match measured {
                Ok(m) => m,
                Err(error) => {
                    eprintln!("    {:<width$} {error}", parser.name);
                    match file_problems.errors.iter_mut().find(|(e, _)| *e == error) {
                        Some((_, parsers)) => parsers.push(parser.name),
                        None => file_problems
                            .errors
                            .push((error.clone(), vec![parser.name])),
                    }
                    let mut r = result("error");
                    r["error"] = json!(error);
                    out.line(r)?;
                    out.flush()?;
                    continue;
                }
            };

            let mut r = result("ok");
            r["statistics"] = json!(statistics);
            r["iterations"] = json!(iterations);
            if parser.is_ours() {
                match expected {
                    None => expected = Some((parser.name, statistics)),
                    Some((name, stats)) if stats != statistics => {
                        eprintln!(
                            "    warning: {} disagrees with {name}: {statistics:?} vs {stats:?}",
                            parser.name
                        );
                        file_problems.disagreements.push(format!(
                            "{} vs {name}: {statistics:?} vs {stats:?}",
                            parser.name
                        ));
                        r["disagrees_with"] = json!(name);
                    }
                    Some(_) => {}
                }
            }
            out.line(r)?;

            let mut throughputs = Vec::with_capacity(samples.len());
            for (index, sample) in samples.iter().enumerate() {
                let mut s = json!({
                    "type": "sample",
                    "file": input.id,
                    "parser": parser.name,
                    "sample": index,
                    "iterations": iterations,
                    "ns": sample.ns,
                });
                let fields = s.as_object_mut().unwrap();
                fields.extend(json!(sample.usage).as_object().unwrap().clone());
                if let Some(reading) = sample.reading {
                    fields.extend(json!(reading).as_object().unwrap().clone());
                }
                out.line(s)?;
                throughputs.push(data.len() as f64 * iterations as f64 / sample.ns as f64);
            }
            out.flush()?;

            let cycles_per_byte: Vec<f64> = samples
                .iter()
                .filter_map(|s| {
                    s.reading
                        .map(|r| r.cycles as f64 / (data.len() as f64 * iterations as f64))
                })
                .collect();
            let summary = Summary {
                throughput: median(&mut throughputs),
                cv: cv(&throughputs),
                cycles_per_byte: Range::of(&cycles_per_byte),
            };
            if args.verbose {
                let cpb = summary.cycles_per_byte.map_or(String::new(), |c| {
                    format!("  {:.3} c/B ({:.3}-{:.3})", c.mean, c.min, c.max)
                });
                eprintln!(
                    "    {:<width$} {:>7.3} GB/s ±{:.1}%{cpb}",
                    parser.name,
                    summary.throughput,
                    summary.cv * 100.0
                );
            }
            summaries.entry(parser.name).or_default().push(summary);
        }
        if !file_problems.errors.is_empty() || !file_problems.disagreements.is_empty() {
            problems.push(file_problems);
        }
    }

    out.line(json!({"type": "end", "finished": system::timestamp()}))?;
    out.flush()?;

    println!(
        "\n{:<width$} {:>6} {:>11} {:>10} {:>9} {:>8} {:>8}",
        "parser", "files", "GB/s (geo)", "median CV", "c/B (geo)", "c/B min", "c/B max"
    );
    for parser in &parsers {
        let Some(results) = summaries.get(parser.name) else {
            continue;
        };
        let throughputs: Vec<f64> = results.iter().map(|s| s.throughput).collect();
        let mut cvs: Vec<f64> = results.iter().map(|s| s.cv).collect();
        let per_file: Vec<f64> = results
            .iter()
            .filter_map(|s| s.cycles_per_byte.map(|c| c.mean))
            .collect();
        let cpb = match Range::of(&per_file) {
            Some(c) => format!(" {:>9.3} {:>8.3} {:>8.3}", geomean(&per_file), c.min, c.max),
            None => String::new(),
        };
        println!(
            "{:<width$} {:>6} {:>11.3} {:>9.2}%{cpb}",
            parser.name,
            results.len(),
            geomean(&throughputs),
            median(&mut cvs) * 100.0
        );
    }

    if !selection.skipped.is_empty() {
        println!("\nskipped {} files:", selection.skipped.len());
        for skipped in &selection.skipped {
            println!("  {:<12} {}", short(&skipped.id), skipped.reason);
        }
    }
    if !problems.is_empty() {
        println!("\nerrors or disagreements on {} files:", problems.len());
        for p in &problems {
            match p.input.path.display().to_string() {
                path if path == p.input.label() => println!("  {path}"),
                path => println!("  {} {path}", p.input.label()),
            }
            for (error, names) in &p.errors {
                let parsers = if names.len() == p.ours_run && names.len() > 1 {
                    format!("all {} of our parsers", names.len())
                } else {
                    names.join(", ")
                };
                println!("    {error}: {parsers}");
            }
            for disagreement in &p.disagreements {
                println!("    statistics differ: {disagreement}");
            }
        }
    }
    if let Some((base_path, base)) = &baseline {
        println!("\ncompared to {}:", base_path.display());
        match print_comparison(base, &load_medians(&path)?, |_| true, args.threshold) {
            Ok(()) => {}
            // the run is written either way, so this is not an error of the run
            Err(e) => println!("  {e}"),
        }
    }
    eprintln!("\nwrote {}", path.display());
    Ok(())
}

/// The newest result in `dir` named like a default output for `label`: `<label>-<time>.jsonl`.
fn latest_result(dir: &Path, label: &str) -> Result<PathBuf, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let is_timestamp = |t: &str| {
        let (date, time) = t.split_at_checked(8).unwrap_or_default();
        date.len() == 8
            && time.len() == 7
            && time.starts_with('-')
            && date
                .bytes()
                .chain(time.bytes().skip(1))
                .all(|b| b.is_ascii_digit())
    };
    entries
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|name| {
            let time = name
                .strip_prefix(label)
                .and_then(|n| n.strip_prefix('-'))
                .and_then(|n| n.strip_suffix(".jsonl"));
            time.is_some_and(is_timestamp)
        })
        // the timestamps sort in time order
        .max()
        .map(|name| dir.join(name))
        .ok_or_else(|| {
            format!(
                "no previous `{label}` result in {} to compare to",
                dir.display()
            )
        })
}

/// One parser on one file: its statistics, the iterations per sample and the samples; or the
/// parser's error.
type Measured = Result<(Statistics, u64, Vec<Sample>), String>;

struct Sample {
    ns: u64,
    usage: Usage,
    reading: Option<counters::Reading>,
}

/// Calibrate, warm up and sample one parser on one file. The outer error is fatal, the inner one
/// is the parser stopping with an error.
fn measure(
    parser: &Parser,
    input: &harness::inputs::Input,
    data: &[u8],
    options: &Options,
    args: &RunArgs,
    min_sample_time: Duration,
    mut counters: Option<&mut Counters>,
) -> Result<Measured, String> {
    // built once, outside the timed region: only parsing is measured, not building the DFA and
    // buffers, which can take longer than parsing a small file
    let mut prepared = (parser.prepare)(&input.dialect, data, options);
    let mut parse = || prepared.statistics(black_box(data)).map(black_box);

    // Warm up. The first parse also checks for errors; it runs cold, so the iterations per sample
    // come from the parses after it if there are any.
    let warmup = Duration::from_millis(args.warmup);
    let start = Instant::now();
    let statistics = match parse() {
        Ok(s) => s,
        Err(e) => return Ok(Err(e.to_string())),
    };
    let cold = start.elapsed();
    let mut warm = 0;
    while start.elapsed() < warmup {
        let _ = parse();
        warm += 1;
    }
    let per_parse = match warm {
        0 => cold,
        _ => (start.elapsed() - cold) / warm,
    };
    let iterations = (min_sample_time.as_nanos() / per_parse.as_nanos().max(1)).max(1) as u64;

    let mut samples = Vec::with_capacity(args.samples);
    for _ in 0..args.samples {
        if let Some(c) = counters.as_deref_mut() {
            c.start().map_err(|e| format!("starting counters: {e}"))?;
        }
        // nested inside the counters and around the timed loop, so that neither the counter
        // syscalls nor these two calls are timed
        let usage = Usage::now();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = parse();
        }
        let ns = start.elapsed().as_nanos() as u64;
        let usage = Usage::now().since(usage);
        let reading = match counters.as_deref_mut() {
            Some(c) => Some(c.stop().map_err(|e| format!("reading counters: {e}"))?),
            None => None,
        };
        samples.push(Sample { ns, usage, reading });
    }
    Ok(Ok((statistics, iterations, samples)))
}

/// Where parsers failed on one file, for the report after the table.
struct Problems<'a> {
    input: &'a harness::inputs::Input,
    /// Our parsers that were run. Only they report errors: rust-csv parses anything.
    ours_run: usize,
    /// Each distinct error, with the parsers that stopped with it.
    errors: Vec<(String, Vec<&'static str>)>,
    disagreements: Vec<String>,
}

/// One parser on one file, for the terminal output.
struct Summary {
    /// Median over the samples (GB/s).
    throughput: f64,
    /// Coefficient of variation of the throughput.
    cv: f64,
    /// Over the samples; `None` without counters.
    cycles_per_byte: Option<Range>,
}

#[derive(Clone, Copy)]
struct Range {
    mean: f64,
    min: f64,
    max: f64,
}

impl Range {
    fn of(values: &[f64]) -> Option<Range> {
        if values.is_empty() {
            return None;
        }
        Some(Range {
            mean: values.iter().sum::<f64>() / values.len() as f64,
            min: values.iter().copied().fold(f64::INFINITY, f64::min),
            max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        })
    }
}

fn geomean(values: &[f64]) -> f64 {
    (values.iter().map(|v| v.ln()).sum::<f64>() / values.len() as f64).exp()
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Coefficient of variation: standard deviation over mean.
fn cv(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() / mean
}

/// Median throughput (parsed bytes/ns = GB/s) per `(file, parser)`.
type Medians = HashMap<(String, String), f64>;

/// The [`Medians`] of a result file.
fn load_medians(path: &Path) -> Result<Medians, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut sizes: HashMap<String, f64> = HashMap::new();
    let mut throughputs: HashMap<(String, String), Vec<f64>> = HashMap::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        // a run killed mid-write leaves a partial last line
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            eprintln!("{}:{}: ignoring unparsable line", path.display(), n + 1);
            continue;
        };
        let str_field = |k: &str| value[k].as_str().unwrap_or_default().to_owned();
        match value["type"].as_str() {
            Some("file") => {
                sizes.insert(str_field("id"), value["bytes"].as_f64().unwrap_or(f64::NAN));
            }
            Some("sample") => {
                let file = str_field("file");
                let size = sizes.get(&file).copied().unwrap_or(f64::NAN);
                let bytes = size * value["iterations"].as_f64().unwrap_or(f64::NAN);
                let throughput = bytes / value["ns"].as_f64().unwrap_or(f64::NAN);
                throughputs
                    .entry((file, str_field("parser")))
                    .or_default()
                    .push(throughput);
            }
            _ => {}
        }
    }
    Ok(throughputs
        .into_iter()
        .map(|(k, mut v)| (k, median(&mut v)))
        .collect())
}

fn compare(args: CompareArgs) -> Result<(), String> {
    let base = load_medians(&args.base)?;
    let new = load_medians(&args.new)?;
    let selected = |parser: &str| {
        let included = args.parsers.parser.is_empty()
            || args.parsers.parser.iter().any(|g| parsers::glob(g, parser));
        included
            && !args
                .parsers
                .exclude
                .iter()
                .any(|g| parsers::glob(g, parser))
    };
    print_comparison(&base, &new, selected, args.threshold)
}

/// The change from `base` to `new` per parser, then the files that changed by more than
/// `threshold` percent. Only parsers for which `selected` holds are compared.
fn print_comparison(
    base: &Medians,
    new: &Medians,
    selected: impl Fn(&str) -> bool,
    threshold: f64,
) -> Result<(), String> {
    // ratios new/base per parser, over the files both measured
    let mut ratios: HashMap<&str, Vec<(&str, f64)>> = HashMap::new();
    for ((file, parser), new_tp) in new {
        if !selected(parser) {
            continue;
        }
        if let Some(base_tp) = base.get(&(file.clone(), parser.clone())) {
            ratios
                .entry(parser)
                .or_default()
                .push((file, new_tp / base_tp));
        }
    }
    if ratios.is_empty() {
        return Err("the two result files have no (file, parser) pair in common".into());
    }

    let mut names: Vec<&str> = ratios.keys().copied().collect();
    names.sort();
    let width = name_width(names.iter().copied());
    println!(
        "{:<width$} {:>6} {:>10} {:>9} {:>9}",
        "parser", "files", "geo mean", "min", "max"
    );
    for name in &names {
        let r = &ratios[name];
        let geomean = geomean(&r.iter().map(|(_, x)| *x).collect::<Vec<_>>());
        let min = r.iter().map(|(_, x)| *x).fold(f64::INFINITY, f64::min);
        let max = r.iter().map(|(_, x)| *x).fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{name:<width$} {:>6} {:>+9.2}% {:>+8.2}% {:>+8.2}%",
            r.len(),
            (geomean - 1.0) * 100.0,
            (min - 1.0) * 100.0,
            (max - 1.0) * 100.0
        );
    }

    let fraction = threshold / 100.0;
    let mut changed: Vec<(&str, &str, f64)> = names
        .iter()
        .flat_map(|name| ratios[name].iter().map(move |(file, r)| (*name, *file, *r)))
        .filter(|(_, _, r)| (r - 1.0).abs() > fraction)
        .collect();
    if !changed.is_empty() {
        changed.sort_by(|a, b| a.2.total_cmp(&b.2));
        println!("\nchanged by more than {threshold}%:");
        for (parser, file, r) in changed {
            println!(
                "  {:>+8.2}%  {:<width$} {}",
                (r - 1.0) * 100.0,
                parser,
                short(file)
            );
        }
    }
    Ok(())
}
