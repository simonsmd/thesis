//! Choosing the files to run on: whole datasets and/or individual paths, with filters.

use std::{
    collections::{BTreeSet, HashSet},
    fmt, fs,
    ops::Bound,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::{
    dataset::{self, DialectSource, FileEntry, Store},
    dialect::{Byte, Dialect, QuoteStyle, Terminator},
    encoding::Data,
};

#[derive(Debug, Clone, clap::Args)]
pub struct InputArgs {
    /// Dataset from the datasets directory, e.g. `quick` (repeatable). With `--id`, only those
    /// files are run, with their dialect from these datasets.
    #[arg(long, short)]
    pub dataset: Vec<String>,

    /// A corpus file by its key (repeatable): the whole key or a unique prefix of it. It gets its
    /// dialect from the `--dataset`, which the dialect flags adjust. Without a dataset it starts
    /// from RFC 4180 and needs dialect flags.
    #[arg(long)]
    pub id: Vec<String>,

    /// A CSV file outside the corpus (repeatable). It starts from RFC 4180, which the dialect
    /// flags adjust.
    #[arg(long, short)]
    pub file: Vec<PathBuf>,

    /// Skip files larger than this, e.g. `512M`, `2G`.
    #[arg(long)]
    pub max_size: Option<Size>,

    /// Skip files smaller than this.
    #[arg(long)]
    pub min_size: Option<Size>,

    /// Only dataset files whose encoding class is one of these, e.g. `ascii`, `utf-8` (repeatable).
    #[arg(long)]
    pub encoding: Vec<String>,

    /// Encoding of `--file` inputs and of `--id` inputs without a dataset, as a Python codec name such as
    /// `cp1252` or `utf-16-le`. They are converted to UTF-8 before parsing.
    #[arg(long)]
    pub file_encoding: Option<String>,

    /// Directory with the dataset manifests and `overrides.toml`.
    #[arg(long, default_value_os_t = dataset::default_datasets_dir())]
    pub datasets_dir: PathBuf,

    /// The content-addressed file store.
    #[arg(long, default_value_os_t = dataset::default_corpus_dir())]
    pub corpus: PathBuf,

    #[command(flatten)]
    pub dialect: DialectArgs,
}

/// Dialect adjustments for files given with `--file` or `--id`.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct DialectArgs {
    /// Field delimiter for `--file` and `--id` inputs.
    #[arg(long, value_parser = parse_byte)]
    pub delimiter: Option<Byte>,
    /// Record terminator for `--file` and `--id` inputs: `crlf` or one character.
    #[arg(long, value_parser = Terminator::parse)]
    pub terminator: Option<Terminator>,
    /// Quote character for `--file` and `--id` inputs, or `none`.
    #[arg(long, value_parser = parse_optional_byte)]
    pub quote: Option<OptionalByte>,
    #[arg(long)]
    pub quote_style: Option<QuoteStyle>,
    #[arg(long)]
    pub quotes_in_unquoted_fields: Option<bool>,
    /// Escape character for `--file` and `--id` inputs, or `none`.
    #[arg(long, value_parser = parse_optional_byte)]
    pub escape: Option<OptionalByte>,
    #[arg(long)]
    pub escape_in_unquoted_fields: Option<bool>,
    #[arg(long)]
    pub double_quote_escape: Option<bool>,
    /// Comment character for `--file` and `--id` inputs, or `none`.
    #[arg(long, value_parser = parse_optional_byte)]
    pub comment: Option<OptionalByte>,
}

impl DialectArgs {
    fn is_empty(&self) -> bool {
        self.delimiter.is_none()
            && self.terminator.is_none()
            && self.quote.is_none()
            && self.quote_style.is_none()
            && self.quotes_in_unquoted_fields.is_none()
            && self.escape.is_none()
            && self.escape_in_unquoted_fields.is_none()
            && self.double_quote_escape.is_none()
            && self.comment.is_none()
    }

    fn apply(&self, dialect: &mut Dialect) {
        let d = dialect;
        if let Some(v) = self.delimiter {
            d.delimiter = v;
        }
        if let Some(v) = self.terminator {
            d.terminator = v;
        }
        if let Some(v) = self.quote {
            d.quote = v.0;
        }
        if let Some(v) = self.quote_style {
            d.quote_style = Some(v);
        }
        if let Some(v) = self.quotes_in_unquoted_fields {
            d.quotes_in_unquoted_fields = Some(v);
        }
        if let Some(v) = self.escape {
            d.escape = v.0;
        }
        if let Some(v) = self.escape_in_unquoted_fields {
            d.escape_in_unquoted_fields = v;
        }
        if let Some(v) = self.double_quote_escape {
            d.double_quote_escape = v;
        }
        if let Some(v) = self.comment {
            d.comment = v.0;
        }
    }
}

/// `\t` is accepted for a tab, since a literal one is awkward to pass in a shell.
fn parse_byte(s: &str) -> Result<Byte, String> {
    match s {
        "\\t" | "tab" => Ok(Byte(b'\t')),
        _ => Byte::parse(s),
    }
}

/// A byte or `none`, as its own type: clap reads a plain `Option<Option<Byte>>` field as a flag
/// with an optional value and panics on the type the value parser returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptionalByte(pub Option<Byte>);

fn parse_optional_byte(s: &str) -> Result<OptionalByte, String> {
    match s {
        "none" => Ok(OptionalByte(None)),
        _ => parse_byte(s).map(|b| OptionalByte(Some(b))),
    }
}

/// A byte count with an optional binary suffix: `4096`, `64K`, `1.5G`, `2GiB`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Size(pub u64);

impl FromStr for Size {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let t = s.trim();
        let t = t
            .strip_suffix("iB")
            .or_else(|| t.strip_suffix('B'))
            .unwrap_or(t);
        let (number, unit) = match t.char_indices().last() {
            Some((i, c)) if c.is_ascii_alphabetic() => (&t[..i], c.to_ascii_uppercase()),
            _ => (t, ' '),
        };
        let shift = match unit {
            ' ' => 0,
            'K' => 10,
            'M' => 20,
            'G' => 30,
            'T' => 40,
            _ => return Err(format!("unknown size unit in {s:?}")),
        };
        let number: f64 = number
            .trim()
            .parse()
            .map_err(|_| format!("invalid size {s:?}"))?;
        Ok(Size((number * (1u64 << shift) as f64) as u64))
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0 as f64;
        match self.0 {
            0..1024 => write!(f, "{} B", self.0),
            1024..1048576 => write!(f, "{:.1} KiB", n / 1024.0),
            1048576..1073741824 => write!(f, "{:.1} MiB", n / 1048576.0),
            _ => write!(f, "{:.2} GiB", n / 1073741824.0),
        }
    }
}

/// A file to run on, with a complete dialect.
#[derive(Debug, Clone)]
pub struct Input {
    /// Content hash for corpus files, else the path as given.
    pub id: String,
    pub path: PathBuf,
    pub size: u64,
    pub dataset: Option<String>,
    pub bucket: Option<u32>,
    /// The encoding to convert the file from, `None` to use its bytes as they are.
    pub encoding: Option<String>,
    pub dialect: Dialect,
    pub dialect_source: DialectSource,
    /// The dialect flags changed the dialect.
    pub dialect_adjusted: bool,
}

impl Input {
    /// A short label for progress output.
    pub fn label(&self) -> String {
        if dataset::is_content_hash(&self.id) {
            self.id[..10].to_owned()
        } else {
            self.id.clone()
        }
    }

    /// Read the file into memory, converted to UTF-8.
    pub fn read(&self) -> Result<Data, String> {
        Data::read(&self.path, self.encoding.as_deref())
    }

    /// Like [`Input::read`], but maps the file where it needs no conversion.
    pub fn map(&self) -> Result<Data, String> {
        Data::map(&self.path, self.encoding.as_deref())
    }
}

/// A file that was selected but cannot be run, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Skipped {
    pub id: String,
    pub path: PathBuf,
    pub reason: String,
}

pub struct Selection {
    pub inputs: Vec<Input>,
    pub skipped: Vec<Skipped>,
    /// `(name, manifest hash)` of each dataset used.
    pub datasets: Vec<(String, String)>,
    pub overrides_hash: String,
}

impl InputArgs {
    /// Resolve the selected files. Errors are for unusable arguments; unusable files end up in
    /// [`Selection::skipped`].
    pub fn resolve(&self) -> Result<Selection, String> {
        if self.dataset.is_empty() && self.id.is_empty() && self.file.is_empty() {
            return Err("nothing to run on: pass --dataset, --id and/or --file".into());
        }
        // an old manifest can hold a different dialect for the same file, so none is guessed
        if self.dataset.is_empty() && !self.id.is_empty() && self.dialect.is_empty() {
            return Err(
                "--id needs the --dataset to take the dialect from, or dialect flags".into(),
            );
        }

        let store = Store::open(&self.datasets_dir, &self.corpus)?;
        let datasets = self
            .dataset
            .iter()
            .map(|name| store.dataset(name))
            .collect::<Result<Vec<_>, _>>()?;
        let mut selection = Selection {
            inputs: Vec::new(),
            skipped: Vec::new(),
            datasets: datasets
                .iter()
                .map(|d| (d.name.clone(), d.hash.clone()))
                .collect(),
            overrides_hash: store.overrides_hash.clone(),
        };
        let mut seen = HashSet::new();
        let mut push = |result: Result<Input, Skipped>| match result {
            Ok(input) => {
                if seen.insert(input.id.clone()) && self.size_ok(input.size) {
                    selection.inputs.push(input);
                }
            }
            Err(skip) => selection.skipped.push(skip),
        };

        if self.id.is_empty() {
            for dataset in &datasets {
                for (key, entry) in &dataset.files {
                    if !self.encoding.is_empty() {
                        let class = entry.encoding.as_ref().map(|e| e.class.as_str());
                        if !self.encoding.iter().any(|e| Some(e.as_str()) == class) {
                            continue;
                        }
                    }
                    push(self.dataset_input(&store, key, entry, &dataset.name)?);
                }
            }
        } else if datasets.is_empty() {
            for arg in &self.id {
                // the corpus keeps files in directories named after the first two characters
                let dir = arg.get(..2).ok_or_else(|| {
                    format!("{arg}: without --dataset, give at least 2 characters")
                })?;
                let names: Vec<String> = fs::read_dir(store.corpus.join(dir))
                    .ok()
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter_map(|entry| entry.file_name().into_string().ok())
                    .collect();
                let key = find_key(&names.iter().map(String::as_str).collect(), arg)
                    .map_err(|e| format!("{e} in the corpus"))?
                    .to_owned();
                let path = store.corpus_path(&key);
                push(Ok(self.plain_input(key, &path)?));
            }
        } else {
            // with --id, the datasets only supply the dialects
            let keys: BTreeSet<&str> = datasets
                .iter()
                .flat_map(|d| d.files.keys().map(String::as_str))
                .collect();
            for arg in &self.id {
                let key = find_key(&keys, arg).map_err(|e| format!("{e} in the datasets"))?;
                let (dataset, entry) = datasets
                    .iter()
                    .find_map(|d| d.files.get(key).map(|entry| (d, entry)))
                    .expect("key is from the datasets");
                push(self.dataset_input(&store, key, entry, &dataset.name)?);
            }
        }

        for path in &self.file {
            // without this, a corpus file would silently run with the default dialect
            if let (Ok(file), Ok(corpus)) = (path.canonicalize(), store.corpus.canonicalize())
                && file.starts_with(&corpus)
            {
                return Err(format!(
                    "{}: file is in the corpus, pass it with --id",
                    path.display()
                ));
            }
            push(Ok(self.plain_input(path.display().to_string(), path)?));
        }

        Ok(selection)
    }

    fn size_ok(&self, size: u64) -> bool {
        self.max_size.is_none_or(|max| size <= max.0)
            && self.min_size.is_none_or(|min| size >= min.0)
    }

    /// A dataset file: its override or sniffed dialect, adjusted by the dialect flags.
    fn dataset_input(
        &self,
        store: &Store,
        key: &str,
        entry: &FileEntry,
        dataset: &str,
    ) -> Result<Result<Input, Skipped>, String> {
        let (mut dialect, dialect_source) = match store.dialect(key, entry) {
            Ok(known) => known,
            // an incomplete sniffed dialect can still be completed with the flags
            Err(_) if !self.dialect.is_empty() => (Dialect::default(), DialectSource::Default),
            Err(e) => {
                return Ok(Err(skipped(store, key, format!("incomplete dialect: {e}"))));
            }
        };
        let before = dialect.clone();
        self.dialect.apply(&mut dialect);
        dialect.validate().map_err(|e| format!("{key}: {e}"))?;
        let adjusted = dialect != before;

        Ok(
            corpus_input(store, key, entry, dataset, (dialect, dialect_source)).map(|mut input| {
                input.dialect_adjusted = adjusted;
                input
            }),
        )
    }

    /// A file without a known dialect: RFC 4180, adjusted by the dialect flags.
    fn plain_input(&self, id: String, path: &Path) -> Result<Input, String> {
        let size = path
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        let mut dialect = Dialect::default();
        self.dialect.apply(&mut dialect);
        dialect
            .validate()
            .map_err(|e| format!("{}: {e}", path.display()))?;

        Ok(Input {
            id,
            path: path.to_owned(),
            size,
            dataset: None,
            bucket: None,
            encoding: self.file_encoding.clone(),
            dialect_adjusted: dialect != Dialect::default(),
            dialect,
            dialect_source: DialectSource::Default,
        })
    }
}

/// The key an `--id` argument names: a key or a unique prefix of one.
fn find_key<'a>(keys: &BTreeSet<&'a str>, arg: &str) -> Result<&'a str, String> {
    // the key equal to `arg` sorts before all others that start with it
    let mut matches = keys
        .range::<str, _>((Bound::Included(arg), Bound::Unbounded))
        .take_while(|key| key.starts_with(arg));
    match (matches.next(), matches.next()) {
        (Some(key), _) if *key == arg => Ok(key),
        (Some(key), None) => Ok(key),
        (None, _) => Err(format!("{arg}: no file with this key")),
        (Some(_), Some(_)) => Err(format!(
            "{arg}: prefix matches {} files",
            2 + matches.count()
        )),
    }
}

fn skipped(store: &Store, key: &str, reason: String) -> Skipped {
    Skipped {
        id: key.to_owned(),
        path: store.corpus_path(key),
        reason,
    }
}

/// A corpus file with the given dialect, unless the corpus copy does not match the manifest.
fn corpus_input(
    store: &Store,
    key: &str,
    entry: &FileEntry,
    dataset: &str,
    (dialect, dialect_source): (Dialect, DialectSource),
) -> Result<Input, Skipped> {
    let path = store.corpus_path(key);
    match path.metadata() {
        Ok(m) if m.len() == entry.size => {}
        Ok(m) => {
            return Err(skipped(
                store,
                key,
                format!("size {} in corpus, {} in manifest", m.len(), entry.size),
            ));
        }
        Err(e) => return Err(skipped(store, key, format!("not in corpus: {e}"))),
    }

    Ok(Input {
        id: key.to_owned(),
        path,
        size: entry.size,
        dataset: Some(dataset.to_owned()),
        bucket: entry.bucket,
        encoding: entry
            .encoding
            .as_ref()
            .and_then(|e| e.name())
            .map(str::to_owned),
        dialect,
        dialect_source,
        dialect_adjusted: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_flags_parse() {
        #[derive(clap::Parser)]
        struct Args {
            #[command(flatten)]
            dialect: DialectArgs,
        }
        let args = <Args as clap::Parser>::try_parse_from([
            "x",
            "--quote",
            "none",
            "--escape",
            "\\",
            "--comment",
            "#",
        ])
        .unwrap();
        assert_eq!(args.dialect.quote, Some(OptionalByte(None)));
        assert_eq!(args.dialect.escape, Some(OptionalByte(Some(Byte(b'\\')))));
        assert_eq!(args.dialect.comment, Some(OptionalByte(Some(Byte(b'#')))));
    }

    #[test]
    fn keys_by_prefix() {
        let index = BTreeSet::from(["00140e87-401a4", "00140f11-2a", "0014", "3489bbf7"]);
        assert_eq!(find_key(&index, "0014"), Ok("0014"));
        assert_eq!(find_key(&index, "00140e8"), Ok("00140e87-401a4"));
        assert_eq!(find_key(&index, "3489bbf7"), Ok("3489bbf7"));
        assert!(find_key(&index, "00140").is_err());
        assert!(find_key(&index, "ff").is_err());
    }

    #[test]
    fn sizes() {
        assert_eq!("4096".parse(), Ok(Size(4096)));
        assert_eq!("64K".parse(), Ok(Size(65536)));
        assert_eq!("1.5G".parse(), Ok(Size(3 << 29)));
        assert_eq!("2GiB".parse(), Ok(Size(2 << 30)));
        assert_eq!("10 MB".parse(), Ok(Size(10 << 20)));
        assert!("12X".parse::<Size>().is_err());
    }
}
