"""The published dataset: what was drawn, and what dialect each file is in.

`download` produces bytes and a log; this produces the artifact a parser
benchmark actually consumes -- `datasets/<name>.json`, one object keyed by
content hash, carrying every file's source, its sniffed dialect in the parser's
own `Config` vocabulary, and the evidence behind it.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from . import config, dialects

ROOT = Path(__file__).parent.parent / "datasets"
OVERRIDES = ROOT / "overrides.toml"

# How much of a file the sniffers that work in memory see (CleverCSV and the
# encoding detectors). Our own sniffer and DuckDB are given the file itself.
SNIFF_CAP = 16 << 20

# Our sniffer counts records separately for this many equal pieces of a file.
# A piece whose records mostly have a different field count than the file as
# a whole is a file that changes along the way: a preamble in one dialect, two
# exports glued together, a semicolon section appended to a comma one. The
# start of a file cannot show that.
PARTS = 8

# What `--review` accepts, see `needs_review`.
REVIEW_MODES = ("disagreements", "all")


def manifest_path(name: str) -> Path:
    return ROOT / f"{name}.json"


def review_path(name: str) -> Path:
    """Where this dataset's unsettled files are scaffolded.

    Per dataset (each build rewrites it), beside its definition so the two
    artifacts of one draw sort together. Derived from `ROOT` at call time so a
    caller that repoints the root does not still write here.
    """
    return ROOT / f"{name}.review.toml"


def overrides() -> dict[str, dict[str, Any]]:
    """Hand-written dialect decisions, keyed by content hash.

    Read-only, and read here only to keep decided files out of the review
    queue. Nothing merges them into the manifest: a consumer applies them over
    the sniffed dialect at read time, which leaves both answers legible and
    means a changed override needs no regeneration at all.

    TOML rather than YAML because the values are punctuation -- `,` `;` `\\t`
    `|` `"` -- and YAML's implicit typing is at its worst exactly there. It
    also rejects a duplicate table outright, so one entry per hash is enforced
    by the parser rather than by care.
    """
    if not OVERRIDES.exists():
        return {}
    import tomllib

    with OVERRIDES.open("rb") as fh:
        return tomllib.load(fh).get("files", {})


# Fewer records than this and a file is marked `few_records`: a text without
# line breaks or a header with a row or two parses as a table of anything.
MIN_RECORDS = 5

# Above this share of delimiters written like a comma in a sentence, a file is
# marked as `prose` (see `prose_share`).
MAX_PROSE_SHARE = 0.2

# How much of the head `prose_share` and `looks_like_json_or_html` read.
TEXT_SAMPLE = 1 << 20

_JSON_OR_HTML = re.compile(r"[﻿\s]*(\{|<|\[\s*[\[{\"])")
_LETTER = r"[^\W\d_]"


def prose_share(text: str, dialect: dict[str, Any]) -> float:
    """Share of the delimiters outside quotes that follow a letter and come
    before a space.

    In a table the delimiter stands between values. In running text cut at its
    commas it follows a word, the way punctuation does. Real tables with long
    text quote it, so quoted parts are left out. A space delimiter always looks
    like this and is not measured.
    """
    delim = dialect.get("delimiter")
    if not delim or delim == " ":
        return 0.0
    quote, esc = dialect.get("quote"), dialect.get("escape")
    if quote:
        q = re.escape(quote)
        inner = f"[^{q}]"
        if esc:
            e = re.escape(esc)
            inner = f"[^{q}{e}]|{e}."
        # A doubled quote is two quoted parts in a row, so both go. The sample
        # can end inside a quoted field, which then runs to its end.
        text = re.sub(f"{q}(?:{inner})*(?:{q}|$)", "", text, flags=re.DOTALL)
    total = text.count(delim)
    if not total:
        return 0.0
    return len(re.findall(f"{_LETTER}{re.escape(delim)} ", text)) / total


def looks_like_json_or_html(text: str) -> bool:
    """Whether the file starts like JSON, HTML or XML rather than a table."""
    return _JSON_OR_HTML.match(text) is not None


def _doubts(shape: dict[str, Any], stats: dict[str, Any],
            duckdb: dict[str, Any], text: str = "",
            dialect: dict[str, Any] | None = None) -> list[str]:
    """Signs that the dialect may be wrong although no sniffer contradicts it.

    They are recorded for every file. Whether they send a file to review is
    decided by `--review`. `text` is the start of the file, `dialect` the one
    settled on.
    """
    out = []
    if stats["records"] < MIN_RECORDS:
        out.append("few_records")
    if looks_like_json_or_html(text):
        out.append("json_or_html")
    if dialect and prose_share(text, dialect) > MAX_PROSE_SHARE:
        out.append("prose")
    if not shape["delimiter_found"]:
        out.append("no_delimiter")
    if stats["errors"]:
        out.append("parse_errors")
    if stats["unterminated_quote"]:
        out.append("unterminated_quote")
    if shape["delimiter_found"] and stats["consistent_bytes_share"] < MIN_CONSISTENT:
        out.append("ragged")
    common = stats["common_field_count"]
    if any(part["records"] and part["common_field_count"] != common
           for part in stats.get("parts", [])):
        out.append("drift")
    # DuckDB's two answers for a file it could not make sense of: a pipe where
    # the file has none, and a single column where we found several.
    if (duckdb.get("Delimiter") == "|" and duckdb.get("head_has_pipe") is False) or (
            duckdb.get("n_columns") == 1 and shape["delimiter_found"]):
        out.append("duckdb_gave_up")
    return out


def _head_text(head: bytes, enc: Any) -> str:
    """The start of the file as text, for the checks that read words."""
    from . import features

    sample = head[:TEXT_SAMPLE]
    try:
        return sample.decode(features.source_encoding(enc) or "utf-8", "replace")
    except LookupError:
        return sample.decode("utf-8", "replace")


def sniff_one(job: tuple[str, str, int, tuple[str, ...]]) -> dict[str, Any]:
    """Every sniffer's verdict for one downloaded file.

    Runs in a worker process: the sniffers are CPU-bound and the GIL would
    otherwise serialise them behind each other. Takes paths rather than bytes
    so nothing large crosses the process boundary.

    Our own sniffer reads the whole file. Its dialect is the file's dialect
    (`dialects.settle`), and its counts are kept under `stats` to explain
    benchmark results with later.
    """
    key, path_str, size, which = job
    from . import features, sniffers

    path = Path(path_str)
    record: dict[str, Any] = {"content_key": key}
    with path.open("rb") as fh:
        head = fh.read(SNIFF_CAP)
        fh.seek(max(size - 1, 0))
        last = fh.read(1)

    # `size` is the real file size, never the sample's: it is how a sniffer
    # knows that a 16 MiB sample of a 2 GB file ends in a cut off record.
    sample = sniffers.Sample(head, size)
    in_memory = tuple(n for n in which if n not in ("duckdb", "custom"))
    raw = sniffers.run_sample(sample, which=in_memory) if in_memory else {}
    stats: dict[str, Any] = {}
    if "custom" in which:
        try:
            out = features.run(None, sample.enc, path=path, parts=PARTS)
            raw["custom"] = features.shape(
                out, sample.enc, last in (b"\n", b"\r")).as_dict()
            stats = out["stats"]
            record["detection"] = out["detection"]
        except Exception as exc:                            # noqa: BLE001
            raw["custom"] = {"error": f"{type(exc).__name__}: {exc}"}
    if "duckdb" in which:
        # The real path where the bytes are already UTF-8: no temp copy, and
        # no truncation for DuckDB to trip on. Other encodings go through the
        # head converted to UTF-8.
        raw["duckdb"] = sniffers.duckdb_file(path, sample)

    views = dialects.normalise(raw)
    settled = dialects.settle(
        views, head, raw.get("custom", {}).get("delimiter_found", True))
    record["sniffers"] = views
    record["dialect"] = settled["dialect"]
    record["agreement"] = settled["agreement"]
    record["contested"] = settled["contested"]
    # Failures kept apart from the views: there they would be all-None, which
    # reads exactly like a sniffer with no opinion.
    record["sniffer_errors"] = dialects.errors(raw)
    record["doubts"] = _doubts(
        raw["custom"], stats, raw.get("duckdb") or {},
        _head_text(head, sample.enc), settled["dialect"]) if stats else []

    # The harness converts the file to UTF-8 from `bom`, `wide` or `detected`,
    # in that order, before it is parsed.
    enc = sample.enc
    record["encoding"] = {
        "class": enc.encoding_class, "detected": enc.encoding,
        "bom": enc.bom, "wide": enc.wide,
        "magic": enc.magic, "control_frac": round(enc.control_frac, 5),
        "chardet": sniffers.verdict(raw, "chardet").get("encoding"),
    }
    record["stats"] = stats
    return record


def _start_context():
    """A multiprocessing context whose workers can actually start, or None.

    Never `fork`: the caller has an open requests session and live threads, and
    forking that hands a child a copy of any lock a thread happened to hold
    with nothing left alive to release it -- the same reason `probe._sniff_pool`
    refuses it. `forkserver` is preferred and `spawn` is the fallback, because
    whether forkserver works is not something `get_context` can answer: it
    needs an AF_UNIX socket, and the failure surfaces only when a worker is
    started.

    None means run in-process. That is a worse-performing pass, not a broken
    one, and it beats losing a completed download to a sniffing failure.
    """
    import multiprocessing

    for method in ("forkserver", "spawn"):
        try:
            context = multiprocessing.get_context(method)
        except ValueError:
            continue
        try:
            # Actually start one and wait for it. `get_context` succeeding says
            # nothing about whether a child can be spawned here.
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as probe:
                probe.submit(int).result()
            return context
        except Exception:                                   # noqa: BLE001
            continue
    return None


def build(name: str, members: list[dict[str, Any]], provenance: dict[str, Any],
          which: tuple[str, ...] = dialects.DIALECT_SNIFFERS + ("chardet",),
          procs: int = 0, on_event: Any = None,
          review: str = "disagreements") -> Path:
    """Sniff every file of a dataset and write `datasets/<name>.json`.

    `members` is the dataset's kept downloads; `provenance` is how it was
    drawn. Both come from `download`, which owns that bookkeeping. `review`
    decides which files are queued for a manual check, see `needs_review`.
    """
    emit = on_event or (lambda *a, **k: None)
    jobs = []
    for rec in members:
        path = config.DATA_ROOT / rec["local_path"]
        if path.exists():
            jobs.append((rec["content_key"], str(path), rec["size"], which))
    procs = procs or config.SNIFF_PROCESSES
    results: dict[str, dict[str, Any]] = {}
    started = time.monotonic()
    # `_start_context` rather than the platform default, which on Linux is
    # `forkserver` and fails where AF_UNIX sockets are refused -- after every
    # file has been downloaded, the most expensive moment to fail.
    context = _start_context() if procs > 1 and len(jobs) > 1 else None
    if context is not None:
        with ProcessPoolExecutor(max_workers=procs, mp_context=context) as pool:
            for i, out in enumerate(pool.map(sniff_one, jobs, chunksize=1)):
                results[out.pop("content_key")] = out
                emit("sniff", done=i + 1, total=len(jobs))
    else:
        # One process keeps a crashing sniffer where it leaves a traceback,
        # which is the same reason `probe` offers `--sniff-procs 0`.
        for i, job in enumerate(jobs):
            out = sniff_one(job)
            results[out.pop("content_key")] = out
            emit("sniff", done=i + 1, total=len(jobs))

    files: dict[str, Any] = {}
    for rec in members:
        key = rec["content_key"]
        entry = {
            "repo": rec["full_name"], "path": rec["path"], "ref": rec.get("ref"),
            "lfs": bool(rec["is_lfs"]), "size": rec["size"], "bucket": rec["bucket"],
        }
        # A Kaggle file is read out of its dataset's zip archive. Its position
        # there is kept so that `csvscrape fetch` can get the file again.
        if "header_offset" in rec:
            from .download import ARCHIVE_FIELDS

            entry["archive"] = {"version": rec.get("version")} | {
                name: rec[name] for name in ARCHIVE_FIELDS}
        entry.update(results.get(key, {"missing": True}))
        files[key] = entry

    ROOT.mkdir(parents=True, exist_ok=True)
    out = manifest_path(name)
    payload = {
        "dataset": provenance | {
            "name": name, "files": len(files),
            "bytes": sum(r["size"] for r in members),
            "sniffers": list(which), "sniff_cap": SNIFF_CAP, "review": review,
            "sniffed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sniff_seconds": round(time.monotonic() - started, 1),
        },
        # Sorted so a rebuild produces the same bytes and a diff shows only
        # what actually changed.
        "files": dict(sorted(files.items())),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n",
                   encoding="utf-8")
    write_review(files, name, review)
    return out


# Below this share of bytes in records with the common field count, a file is
# marked as ragged (see `_doubts`).
MIN_CONSISTENT = 0.95


def needs_review(files: dict[str, Any],
                 mode: str = "disagreements") -> dict[str, Any]:
    """Files whose dialect should be checked by hand, minus those already
    decided in `overrides.toml`.

    A file without a dialect (our sniffer failed on it) is always queued: the
    harness cannot run it. Beyond that, `mode` decides:

      * "disagreements": only files where CleverCSV or DuckDB found something
        else than our sniffer (`contested`, see `dialects.settle`).
      * "all": also files where a sniffer failed, and files with `doubts`:
        fewer than `MIN_RECORDS` records, a start like JSON or HTML, prose,
        no delimiter found, parse errors, ragged records, a field count that
        changes within the file, or a give-up answer from DuckDB.
    """
    decided = set(overrides())
    out = {}
    for key, entry in files.items():
        if key in decided or entry.get("missing"):
            continue
        flagged = entry.get("contested") or entry["dialect"]["delimiter"] is None
        if mode == "all":
            flagged = flagged or entry.get("sniffer_errors") or entry.get("doubts")
        if flagged:
            out[key] = entry
    return out


def _vote(value: Any) -> str:
    """A view's field for the review file: `none` is an answer, `?` is none."""
    if value is None:
        return "?"
    return "none" if value == dialects.NONE else repr(value)


def write_review(files: dict[str, Any], name: str,
                 mode: str = "disagreements") -> Path:
    """Scaffold `<name>.review.toml`: one commented block per unsettled file.

    Generated and disposable, and a *different file* from `overrides.toml` on
    purpose -- you copy a block across once you have decided, so the flow is
    one-way and regenerating can never overwrite a decision. That separation is
    the whole reason the tool needs no TOML writer for the file that matters.
    """
    queue = needs_review(files, mode)
    out = review_path(name)
    ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Files in dataset {name!r} whose dialect should be checked by hand",
        f"# (--review {mode}). Generated -- this file is rewritten every time the dataset",
        f"# is built. Decide one, then move its block to {OVERRIDES.name},",
        "# which is never written by the tool and is shared by every dataset.",
        "# A block is the file's full dialect: a missing key means none.",
        "#",
        f"# {len(queue)} of {len(files)} files.",
        "",
    ]
    for key, entry in sorted(queue.items()):
        d = entry["dialect"]
        lines.append(f"# {entry['repo']}/{entry['path']}  ({entry['size']:,} B)")
        for sniffer, view in (entry.get("sniffers") or {}).items():
            lines.append(f"#   {sniffer:<10} " + " ".join(
                f"{f}={_vote(view[f])}"
                for f in ("delimiter", "quote", "escape", "comment")))
        if entry.get("contested"):
            lines.append(f"#   CONTESTED   {', '.join(entry['contested'])}")
        for sniffer, err in sorted((entry.get("sniffer_errors") or {}).items()):
            lines.append(f"#   ERROR       {sniffer}: {err}")
        if entry.get("doubts"):
            lines.append(f"#   DOUBTS      {', '.join(entry['doubts'])}")
        stats = entry.get("stats") or {}
        if stats:
            lines.append(
                f"#   whole file  {stats['records']:,} records, "
                f"{stats['common_field_count']} fields on "
                f"{stats['common_field_count_share']:.1%} of them, "
                f"{stats['errors']:,} parse errors, "
                f"{stats['quoted_fields']:,} quoted fields")
        lines.append(f"# [files.\"{key}\"]")
        for field in dialects.FIELDS:
            # TOML has no null: an override is a full dialect and a missing key
            # means "none", so None stays a comment once the block is uncommented
            if d[field] is None:
                lines.append(f"# # {field}: none")
            else:
                lines.append(f"# {field} = {json.dumps(d[field])}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
