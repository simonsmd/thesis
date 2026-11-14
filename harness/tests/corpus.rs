//! Every parser against the reference on each file of a dataset, one test per file.
//!
//! The dataset is `CSV_DATASET` (default `test-main-uniform`). Without the corpus (a fresh
//! checkout) the tests are listed as ignored. Accepts the usual libtest arguments, e.g. a name
//! filter: `cargo test -p harness --test corpus -- 3f2a`.
//!
//! rust-csv is only compared where the reference parses without an error, since rust-csv never
//! reports one; `csvcheck` shows those cases.

use harness::{
    check,
    inputs::InputArgs,
    parsers::{self, Options},
};
use libtest_mimic::{Arguments, Failed, Trial};

fn main() {
    let args = Arguments::from_args();
    let dataset = std::env::var("CSV_DATASET").unwrap_or_else(|_| "test-main-uniform".into());

    let inputs = InputArgs {
        dataset: vec![dataset.clone()],
        file: Vec::new(),
        max_size: None,
        min_size: None,
        encoding: Vec::new(),
        file_encoding: None,
        datasets_dir: harness::dataset::default_datasets_dir(),
        corpus: harness::dataset::default_corpus_dir(),
        dialect: Default::default(),
    };
    let trials = match inputs.resolve() {
        Ok(selection) => {
            let mut trials: Vec<Trial> = selection
                .inputs
                .into_iter()
                .map(|input| {
                    let name = format!("{dataset}/{}", input.label());
                    Trial::test(name, move || check_file(&input))
                })
                .collect();
            trials.extend(selection.skipped.into_iter().map(|s| {
                Trial::test(
                    format!("{dataset}/{} ({})", &s.id[..10.min(s.id.len())], s.reason),
                    || Ok(()),
                )
                .with_ignored_flag(true)
            }));
            trials
        }
        Err(error) => {
            vec![Trial::test(format!("{dataset} ({error})"), || Ok(())).with_ignored_flag(true)]
        }
    };

    libtest_mimic::run(&args, trials).exit();
}

fn check_file(input: &harness::inputs::Input) -> Result<(), Failed> {
    let data = input.read()?;
    let report = check::check(
        &parsers::reference(),
        &parsers::all(),
        &input.dialect,
        &data,
        &[Options::default()],
    );
    let failures: Vec<String> = report
        .unexpected_mismatches()
        .map(|o| match &o.verdict {
            check::Verdict::Mismatch { difference, .. } => format!("{}: {difference}", o.parser),
            _ => unreachable!(),
        })
        .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "dialect: {}\npath: {}\n{}",
            input.dialect,
            input.path.display(),
            failures.join("\n")
        )
        .into())
    }
}
