//! Choosing the files to run on: whole datasets and/or individual paths, with filters.

use std::{
    collections::HashSet,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::{
    dataset::{self, DialectSource, Store},
    dialect::{Byte, Dialect, QuoteStyle, Terminator},
    encoding::Data,
};

#[derive(Debug, Clone, clap::Args)]
pub struct InputArgs {
    /// Dataset from the datasets directory, e.g. `quick` (repeatable).
    #[arg(long, short)]
    pub dataset: Vec<String>,

    /// A CSV file to parse (repeatable). A file from the corpus (named by its content hash) gets
    /// its dataset dialect; any other file starts from RFC 4180. The dialect flags adjust either.
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

    /// Encoding of `--file` inputs that are not in a dataset, as a Python codec name such as
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

/// Dialect adjustments for files given with `--file`.
#[derive(Debug, Clone, Default, clap::Args)]
pub struct DialectArgs {
    /// Field delimiter for `--file` inputs.
    #[arg(long, value_parser = parse_byte)]
    pub delimiter: Option<Byte>,
    /// Record terminator for `--file` inputs: `crlf` or one character.
    #[arg(long, value_parser = Terminator::parse)]
    pub terminator: Option<Terminator>,
    /// Quote character for `--file` inputs, or `none`.
    #[arg(long, value_parser = parse_optional_byte)]
    pub quote: Option<OptionalByte>,
    #[arg(long)]
    pub quote_style: Option<QuoteStyle>,
    #[arg(long)]
    pub quotes_in_unquoted_fields: Option<bool>,
    /// Escape character for `--file` inputs, or `none`.
    #[arg(long, value_parser = parse_optional_byte)]
    pub escape: Option<OptionalByte>,
    #[arg(long)]
    pub escape_in_unquoted_fields: Option<bool>,
    #[arg(long)]
    pub double_quote_escape: Option<bool>,
    /// Comment character for `--file` inputs, or `none`.
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
        if self.dataset.is_empty() && self.file.is_empty() {
            return Err("nothing to run on: pass --dataset and/or --file".into());
        }

        let store = Store::open(&self.datasets_dir, &self.corpus)?;
        let mut selection = Selection {
            inputs: Vec::new(),
            skipped: Vec::new(),
            datasets: Vec::new(),
            overrides_hash: store.overrides_hash.clone(),
        };
        let mut seen = HashSet::new();

        for name in &self.dataset {
            let dataset = store.dataset(name)?;
            selection
                .datasets
                .push((dataset.name.clone(), dataset.hash.clone()));

            for (hash, entry) in &dataset.files {
                if !seen.insert(hash.clone()) || !self.size_ok(entry.size) {
                    continue;
                }
                if !self.encoding.is_empty() {
                    let class = entry.encoding.as_ref().map(|e| e.class.as_str());
                    if !self.encoding.iter().any(|e| Some(e.as_str()) == class) {
                        continue;
                    }
                }

                let path = store.corpus_path(hash);
                let skip = |reason: String| Skipped {
                    id: hash.clone(),
                    path: path.clone(),
                    reason,
                };
                let (dialect, dialect_source) = match store.dialect(hash, entry) {
                    Ok(d) => d,
                    Err(e) => {
                        selection
                            .skipped
                            .push(skip(format!("incomplete dialect: {e}")));
                        continue;
                    }
                };
                match path.metadata() {
                    Ok(m) if m.len() == entry.size => {}
                    Ok(m) => {
                        selection.skipped.push(skip(format!(
                            "size {} in corpus, {} in manifest",
                            m.len(),
                            entry.size
                        )));
                        continue;
                    }
                    Err(e) => {
                        selection.skipped.push(skip(format!("not in corpus: {e}")));
                        continue;
                    }
                }

                selection.inputs.push(Input {
                    id: hash.clone(),
                    path,
                    size: entry.size,
                    dataset: Some(dataset.name.clone()),
                    bucket: entry.bucket,
                    encoding: entry
                        .encoding
                        .as_ref()
                        .and_then(|e| e.name())
                        .map(str::to_owned),
                    dialect,
                    dialect_source,
                    dialect_adjusted: false,
                });
            }
        }

        if !self.file.is_empty() {
            let index = store.index();
            for path in &self.file {
                if let Some(input) =
                    self.file_input(&store, &index, path, &mut selection.skipped)?
                    && seen.insert(input.id.clone())
                    && self.size_ok(input.size)
                {
                    selection.inputs.push(input);
                }
            }
        }

        Ok(selection)
    }

    fn size_ok(&self, size: u64) -> bool {
        self.max_size.is_none_or(|max| size <= max.0)
            && self.min_size.is_none_or(|min| size >= min.0)
    }

    fn file_input(
        &self,
        store: &Store,
        index: &std::collections::HashMap<String, dataset::FileEntry>,
        path: &Path,
        skipped: &mut Vec<Skipped>,
    ) -> Result<Option<Input>, String> {
        let size = path
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();

        let known = dataset::is_content_hash(name).then(|| {
            let entry = index.get(name);
            let dialect = match (store.overrides.get(name), entry) {
                (Some(o), _) => Some(Ok((o.dialect.clone(), DialectSource::Override))),
                (None, Some(entry)) => Some(store.dialect(name, entry)),
                (None, None) => None,
            };
            (dialect, entry)
        });
        let (id, base, entry) = match known {
            Some((Some(dialect), entry)) => (name.to_owned(), Some(dialect), entry),
            _ => (path.display().to_string(), None, None),
        };
        let bucket = entry.and_then(|e| e.bucket);
        let encoding = match entry.and_then(|e| e.encoding.as_ref()) {
            Some(known) => known.name().map(str::to_owned),
            None => self.file_encoding.clone(),
        };

        let (mut dialect, dialect_source) = match base {
            Some(Ok(known)) => known,
            // an incomplete sniffed dialect can still be completed with the flags
            Some(Err(e)) if self.dialect.is_empty() => {
                skipped.push(Skipped {
                    id,
                    path: path.to_owned(),
                    reason: format!("incomplete dialect: {e}"),
                });
                return Ok(None);
            }
            Some(Err(_)) | None => (Dialect::default(), DialectSource::Default),
        };
        let before = dialect.clone();
        self.dialect.apply(&mut dialect);
        dialect
            .validate()
            .map_err(|e| format!("{}: {e}", path.display()))?;

        Ok(Some(Input {
            id,
            path: path.to_owned(),
            size,
            dataset: None,
            bucket,
            encoding,
            dialect_adjusted: dialect != before,
            dialect,
            dialect_source,
        }))
    }
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
    fn sizes() {
        assert_eq!("4096".parse(), Ok(Size(4096)));
        assert_eq!("64K".parse(), Ok(Size(65536)));
        assert_eq!("1.5G".parse(), Ok(Size(3 << 29)));
        assert_eq!("2GiB".parse(), Ok(Size(2 << 30)));
        assert_eq!("10 MB".parse(), Ok(Size(10 << 20)));
        assert!("12X".parse::<Size>().is_err());
    }
}
