//! Converting inputs to UTF-8 before any parser sees them.
//!
//! The parsers work on bytes and expect the structural characters to be single ASCII bytes. That
//! holds for ASCII, UTF-8 and the single-byte code pages, but not for UTF-16/32 or the legacy
//! East Asian encodings, where a delimiter byte can be half of a character. So that every file is
//! handled the same way, everything that is not already ASCII or UTF-8 is converted.
//!
//! The encoding names are the ones `csvscrape` writes into the dataset files, which are Python
//! codec names (`cp1252`, `iso8859-15`, `utf-16-le`, `euc_jp`).

use std::{fs, ops::Deref, path::Path};

use encoding_rs::Encoding;

/// Python codec names that `encoding_rs` knows under a different name.
const ALIASES: &[(&str, &str)] = &[
    ("latin-1", "windows-1252"),
    ("cp874", "windows-874"),
    ("tis-620", "windows-874"),
    ("cp866", "ibm866"),
    ("cp932", "shift_jis"),
    ("shift-jis", "shift_jis"),
    ("shift-jis-2004", "shift_jis"),
    ("shift-jisx0213", "shift_jis"),
    ("euc-jis-2004", "euc-jp"),
    ("euc-jisx0213", "euc-jp"),
    ("cp949", "euc-kr"),
    ("cp936", "gbk"),
    ("gb2312", "gbk"),
    ("cp950", "big5"),
    ("big5hkscs", "big5"),
    ("iso2022-jp", "iso-2022-jp"),
    ("iso2022-jp-1", "iso-2022-jp"),
    ("iso2022-jp-2", "iso-2022-jp"),
    ("mac-roman", "macintosh"),
    ("mac-cyrillic", "x-mac-cyrillic"),
    ("utf-16", "utf-16le"),
    ("utf-16-le", "utf-16le"),
    ("utf-16-be", "utf-16be"),
];

/// Name prefixes of single-byte code pages. One that `encoding_rs` does not know (the DOS pages
/// like `cp437`, most `mac-*` pages) is converted byte by byte instead, see [`to_utf8`].
const SINGLE_BYTE_PREFIXES: &[&str] = &["cp", "iso8859", "mac", "koi8", "ptcp", "kz", "hp-roman"];

/// `data` as UTF-8, or `None` if it can be used as it is.
///
/// `encoding` of `None` means the encoding is not known; the bytes are then left alone.
///
/// Bytes that are not valid in the given encoding become U+FFFD. A single-byte code page without
/// a decoder is read as Latin-1: every byte becomes the character with the same number. The
/// non-ASCII characters are then the wrong ones, but all ASCII bytes, and with them the structure
/// of the file, stay as they are.
pub fn to_utf8(data: &[u8], encoding: Option<&str>) -> Result<Option<Vec<u8>>, String> {
    let Some(encoding) = encoding else {
        return Ok(None);
    };
    let name = encoding.trim().to_ascii_lowercase().replace('_', "-");
    if matches!(name.as_str(), "ascii" | "us-ascii" | "utf-8" | "utf8" | "utf-8-sig") {
        return Ok(None);
    }
    if let Some(big_endian) = match name.as_str() {
        "utf-32" | "utf-32-le" => Some(false),
        "utf-32-be" => Some(true),
        _ => None,
    } {
        return Ok(Some(utf32_to_utf8(data, big_endian)));
    }

    let label = ALIASES.iter().find(|(python, _)| *python == name).map_or(name.as_str(), |(_, label)| label);
    // `cp1252` is `windows-1252`, `iso8859-15` is `iso-8859-15`
    let label = match (label.strip_prefix("cp125"), label.strip_prefix("iso8859-")) {
        (Some(n), _) => format!("windows-125{n}"),
        (_, Some(n)) => format!("iso-8859-{n}"),
        _ => label.to_owned(),
    };

    if let Some(decoder) = Encoding::for_label(label.as_bytes()) {
        // also removes a byte order mark
        let (text, _, _) = decoder.decode(data);
        return Ok(Some(text.into_owned().into_bytes()));
    }
    if SINGLE_BYTE_PREFIXES.iter().any(|prefix| name.starts_with(prefix)) {
        return Ok(Some(encoding_rs::mem::decode_latin1(data).into_owned().into_bytes()));
    }
    Err(format!("no decoder for encoding {encoding:?}"))
}

fn utf32_to_utf8(data: &[u8], big_endian: bool) -> Vec<u8> {
    let mut text = String::with_capacity(data.len() / 4);
    for unit in data.chunks(4) {
        let code = match <[u8; 4]>::try_from(unit) {
            Ok(bytes) if big_endian => u32::from_be_bytes(bytes),
            Ok(bytes) => u32::from_le_bytes(bytes),
            // an incomplete last character
            Err(_) => u32::MAX,
        };
        text.push(char::from_u32(code).unwrap_or(char::REPLACEMENT_CHARACTER));
    }
    // the byte order mark
    if text.starts_with('\u{feff}') {
        text.drain(..3);
    }
    text.into_bytes()
}

/// `data` without a leading UTF-8 byte order mark.
///
/// rust-csv always strips it and this crate's parsers never do, so the harness strips it before
/// any parser sees the input: it is supporting code, not something the DFAs should be compared
/// on. Without this a mark before a quoted first field is a parse error in strict quoting.
pub fn without_bom(data: &[u8]) -> &[u8] {
    data.strip_prefix(b"\xef\xbb\xbf").unwrap_or(data)
}

/// The content of an input file as the parsers get it: UTF-8, without a byte order mark.
///
/// Dereferences to the bytes.
pub struct Data {
    bytes: Bytes,
    /// Where the content starts, after a byte order mark.
    start: usize,
}

enum Bytes {
    Read(Vec<u8>),
    Mapped(memmap2::Mmap),
}

impl Data {
    /// Read the file into memory.
    pub fn read(path: &Path, encoding: Option<&str>) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let converted = to_utf8(&raw, encoding).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self::new(Bytes::Read(converted.unwrap_or(raw))))
    }

    /// Map the file instead of reading it, for callers that hold many large files at once.
    ///
    /// A file that needs converting is still held in memory.
    pub fn map(path: &Path, encoding: Option<&str>) -> Result<Self, String> {
        let file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the input files are not modified while a harness binary runs
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
        let converted = to_utf8(&map, encoding).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self::new(converted.map_or(Bytes::Mapped(map), Bytes::Read)))
    }

    /// Bytes that are already in memory, e.g. read from standard input.
    pub fn from_bytes(raw: Vec<u8>, encoding: Option<&str>) -> Result<Self, String> {
        let converted = to_utf8(&raw, encoding)?;
        Ok(Self::new(Bytes::Read(converted.unwrap_or(raw))))
    }

    fn new(bytes: Bytes) -> Self {
        let all: &[u8] = match &bytes {
            Bytes::Read(vec) => vec,
            Bytes::Mapped(map) => map,
        };
        let start = all.len() - without_bom(all).len();
        Self { bytes, start }
    }
}

impl Deref for Data {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match &self.bytes {
            Bytes::Read(vec) => &vec[self.start..],
            Bytes::Mapped(map) => &map[self.start..],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(data: &[u8], encoding: &str) -> Vec<u8> {
        to_utf8(data, Some(encoding)).unwrap().expect("should be converted")
    }

    #[test]
    fn utf8_and_unknown_are_left_alone() {
        assert_eq!(to_utf8(b"a,\xc3\xa4\n", Some("utf-8")).unwrap(), None);
        assert_eq!(to_utf8(b"a,b\n", Some("ascii")).unwrap(), None);
        assert_eq!(to_utf8(b"a,\xe4\n", None).unwrap(), None);
    }

    #[test]
    fn converts_python_codec_names() {
        assert_eq!(convert(b"a;\xe4\n", "cp1252"), "a;ä\n".as_bytes());
        assert_eq!(convert(b"a;\xa4\n", "iso8859_15"), "a;€\n".as_bytes());
        assert_eq!(convert(b"\xff\xfea\x00,\x00\xe4\x00", "utf-16-le"), "a,ä".as_bytes());
        assert_eq!(convert(b"\x00a\x00,", "utf-16-be"), b"a,");
        assert_eq!(convert(b"a\x00\x00\x00,\x00\x00\x00", "utf-32-le"), b"a,");
        // the second byte of this Shift JIS character is `|`
        assert_eq!(convert(b"a|\x83\x7c|b", "shift_jis"), "a|ポ|b".as_bytes());
    }

    #[test]
    fn code_pages_without_a_decoder_keep_their_ascii_bytes() {
        assert_eq!(convert(b"a,\x82,b", "cp437"), "a,\u{82},b".as_bytes());
        assert!(to_utf8(b"a", Some("hz")).is_err());
    }

    #[test]
    fn data_starts_after_the_byte_order_mark() {
        let data = Data::from_bytes(b"\xef\xbb\xbfa,b".to_vec(), Some("utf-8-sig")).unwrap();
        assert_eq!(&*data, b"a,b");
    }
}
