//! The datasets written by `csvscrape` and the hand-written dialect overrides.
//!
//! A dataset is `<datasets>/<name>.json`: metadata plus a map from content hash to file entry. The
//! files themselves live in the content-addressed corpus at `<corpus>/<hash[..2]>/<hash>`, shared
//! between datasets. `<datasets>/overrides.toml` holds full dialects keyed by content hash; an
//! override replaces the sniffed dialect of that file in every dataset.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use xxhash_rust::xxh3::Xxh3;

use crate::dialect::Dialect;

/// The workspace root, for defaults that point into `scripts/`.
pub const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/..");

pub fn default_datasets_dir() -> PathBuf {
    Path::new(ROOT).join("scripts/datasets")
}

pub fn default_corpus_dir() -> PathBuf {
    Path::new(ROOT).join("scripts/data/corpus")
}

/// Where a file's dialect came from.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DialectSource {
    /// `overrides.toml`.
    Override,
    /// The sniffed dialect in a dataset manifest.
    Sniffed,
    /// Neither knows the file: the RFC 4180 default.
    Default,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Encoding {
    pub class: String,
    /// The encoding named by a byte order mark.
    pub bom: Option<String>,
    /// UTF-16 without a byte order mark, recognised by its zero bytes.
    #[serde(default)]
    pub wide: Option<String>,
    /// The encoding the detector named, if it named one.
    #[serde(default)]
    pub detected: Option<String>,
}

impl Encoding {
    /// The encoding to convert the file from, strongest evidence first. `None` if nothing names
    /// one.
    pub fn name(&self) -> Option<&str> {
        self.bom.as_deref().or(self.wide.as_deref()).or(self.detected.as_deref())
    }
}

/// One file in a dataset manifest. Fields the harness does not use are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct FileEntry {
    pub size: u64,
    pub bucket: Option<u32>,
    pub repo: Option<String>,
    pub path: Option<String>,
    pub encoding: Option<Encoding>,
    /// Kept raw so that one incomplete dialect only disqualifies its own file.
    dialect: serde_json::Value,
}

impl FileEntry {
    /// The sniffed dialect, or why it does not fully determine a parser config.
    pub fn sniffed_dialect(&self) -> Result<Dialect, String> {
        let dialect = Dialect::deserialize(&self.dialect).map_err(|e| e.to_string())?;
        dialect.validate()?;
        Ok(dialect)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Dataset {
    #[serde(skip)]
    pub name: String,
    /// Hash of the manifest bytes, to tie results to the exact dataset.
    #[serde(skip)]
    pub hash: String,
    pub files: BTreeMap<String, FileEntry>,
}

/// An override: the full dialect, plus an optional `note` explaining the decision.
#[derive(Debug, Clone)]
pub struct Override {
    pub dialect: Dialect,
    pub note: Option<String>,
}

/// The datasets directory and the corpus it refers to.
pub struct Store {
    pub datasets_dir: PathBuf,
    pub corpus: PathBuf,
    pub overrides: HashMap<String, Override>,
    /// Hash of `overrides.toml`, empty if there is none.
    pub overrides_hash: String,
}

impl Store {
    pub fn open(datasets_dir: &Path, corpus: &Path) -> Result<Self, String> {
        let path = datasets_dir.join("overrides.toml");
        let (overrides, overrides_hash) = match fs::read(&path) {
            Ok(bytes) => (parse_overrides(&bytes).map_err(|e| format!("{}: {e}", path.display()))?, hash(&bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (HashMap::new(), String::new()),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        Ok(Self {
            datasets_dir: datasets_dir.to_owned(),
            corpus: corpus.to_owned(),
            overrides,
            overrides_hash,
        })
    }

    pub fn dataset(&self, name: &str) -> Result<Dataset, String> {
        let path = self.datasets_dir.join(format!("{name}.json"));
        let bytes = fs::read(&path).map_err(|e| {
            let available = self.dataset_names().join(", ");
            format!("{}: {e} (available: {available})", path.display())
        })?;
        let mut dataset: Dataset =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        dataset.name = name.to_owned();
        dataset.hash = hash(&bytes);
        Ok(dataset)
    }

    /// Names of all datasets in the directory, sorted.
    pub fn dataset_names(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(&self.datasets_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name().into_string().ok()?;
                // `<name>.review.toml` sits next to the manifests; only `.json` are datasets
                name.strip_suffix(".json").map(str::to_owned)
            })
            .collect();
        names.sort();
        names
    }

    /// Where the corpus keeps the file with this content hash.
    pub fn corpus_path(&self, hash: &str) -> PathBuf {
        self.corpus.join(&hash[..2.min(hash.len())]).join(hash)
    }

    /// The dialect of a file in the dataset: its override if there is one, else the sniffed one.
    pub fn dialect(&self, hash: &str, entry: &FileEntry) -> Result<(Dialect, DialectSource), String> {
        match self.overrides.get(hash) {
            Some(o) => Ok((o.dialect.clone(), DialectSource::Override)),
            None => entry.sniffed_dialect().map(|d| (d, DialectSource::Sniffed)),
        }
    }

    /// Every file entry of every dataset by content hash, to look up files given by path.
    ///
    /// Datasets that fail to load are left out; they are reported when selected by name.
    pub fn index(&self) -> HashMap<String, FileEntry> {
        let mut index = HashMap::new();
        for name in self.dataset_names() {
            if let Ok(dataset) = self.dataset(&name) {
                for (hash, entry) in dataset.files {
                    index.entry(hash).or_insert(entry);
                }
            }
        }
        index
    }
}

/// Parse `overrides.toml`: `[files."<hash>"]` tables, each a full [`Dialect`] plus a `note`.
///
/// Errors are fatal: an override is a deliberate decision, so a broken one must not silently fall
/// back to the sniffed dialect.
pub fn parse_overrides(bytes: &[u8]) -> Result<HashMap<String, Override>, String> {
    let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
    let mut table: toml::Table = toml::from_str(text).map_err(|e| e.to_string())?;
    let Some(files) = table.remove("files") else {
        return Ok(HashMap::new());
    };
    let toml::Value::Table(files) = files else {
        return Err("`files` must be a table".into());
    };

    files
        .into_iter()
        .map(|(hash, value)| {
            let toml::Value::Table(mut entry) = value else {
                return Err(format!("files.{hash}: must be a table"));
            };
            let note = match entry.remove("note") {
                Some(toml::Value::String(note)) => Some(note),
                Some(_) => return Err(format!("files.{hash}.note: must be a string")),
                None => None,
            };
            // `Dialect` denies unknown keys, which catches misspelled fields
            let dialect = Dialect::deserialize(toml::Value::Table(entry))
                .map_err(|e| format!("files.{hash}: {e}"))?;
            dialect.validate().map_err(|e| format!("files.{hash}: {e}"))?;
            Ok((hash, Override { dialect, note }))
        })
        .collect()
}

fn hash(bytes: &[u8]) -> String {
    let mut hasher = Xxh3::new();
    hasher.update(bytes);
    format!("{:016x}", hasher.digest())
}

/// Whether a file name looks like a corpus content hash (SHA-1 or SHA-256 hex).
pub fn is_content_hash(name: &str) -> bool {
    matches!(name.len(), 40 | 64) && name.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::Byte;

    #[test]
    fn overrides_are_full_dialects() {
        let overrides = parse_overrides(
            br#"
            [files."5c50435271b24c41a489fa9e3134b935de2a60bb"]
            delimiter = ";"
            terminator = "crlf"
            quote = '"'
            quote_style = "strict"
            quotes_in_unquoted_fields = false
            escape_in_unquoted_fields = false
            double_quote_escape = true
            note = "decimal commas"
            "#,
        )
        .unwrap();
        let o = &overrides["5c50435271b24c41a489fa9e3134b935de2a60bb"];
        assert_eq!(o.dialect.delimiter, Byte(b';'));
        assert_eq!(o.dialect.comment, None);
        assert_eq!(o.note.as_deref(), Some("decimal commas"));
    }

    #[test]
    fn incomplete_overrides_are_errors() {
        // a sparse entry from the old merge semantics
        let error = parse_overrides(b"[files.abc]\ndelimiter = \";\"\n").unwrap_err();
        assert!(error.contains("files.abc"), "{error}");
    }

    #[test]
    fn the_datasets_in_the_repo_load() {
        let dir = default_datasets_dir();
        if !dir.exists() {
            return;
        }
        let store = Store::open(&dir, &default_corpus_dir()).unwrap();
        for name in store.dataset_names() {
            let dataset = store.dataset(&name).unwrap();
            assert!(!dataset.files.is_empty(), "{name}");
        }
    }
}
