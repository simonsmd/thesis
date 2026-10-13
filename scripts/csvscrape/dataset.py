"""The published dataset: what was drawn, and what dialect each file is in.

`download` produces bytes and a log; this produces the artifact a parser
benchmark actually consumes -- `datasets/<name>.json`, one object keyed by
content hash, carrying every file's source, its sniffed dialect in the parser's
own `Config` vocabulary, and the evidence behind it.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from . import config, dialects

ROOT = Path(__file__).parent.parent / "datasets"
OVERRIDES = ROOT / "overrides.toml"

# How much of a file the in-memory sniffers see. `features.describe` decodes
# its input to str and holds the line split, so peak memory is several times
# the window, per `--sniff-procs` worker. 16 MiB covers nearly every file in
# its entirety; larger files get `WINDOWS` windows instead.
SNIFF_CAP = 16 << 20

# Above the cap the front of a file stops being evidence about the rest of it,
# so three windows are read instead of one: the opening, the middle and the
# end. Drift -- a preamble in one dialect, two concatenated exports, a
# semicolon section appended to a comma one -- is invisible to a prefix and is
# exactly what a 2 GB benchmark file is likely to contain.
WINDOWS = 3


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


def _windows(size: int) -> list[int]:
    """Byte offsets to sniff from, for a file of this size."""
    if size <= SNIFF_CAP:
        return [0]
    # Evenly spaced, last one ending at EOF. The middle windows land mid-record
    # and `sniff_one` trims to the next newline, which costs one line each.
    span = size - SNIFF_CAP
    return [round(span * i / (WINDOWS - 1)) for i in range(WINDOWS)]


def sniff_one(job: tuple[str, str, int, tuple[str, ...]]) -> dict[str, Any]:
    """Every sniffer's verdict for one downloaded file, plus the whole-file scan.

    Runs in a worker process: the sniffers are CPU-bound and the GIL would
    otherwise serialise them behind each other. Takes paths rather than bytes
    so nothing large crosses the process boundary.
    """
    key, path_str, size, which = job
    from . import sniffers

    path = Path(path_str)
    offsets = _windows(size)
    record: dict[str, Any] = {"content_key": key, "windows": len(offsets)}
    views: dict[str, dict[str, Any]] = {}

    heads: list[bytes] = []
    with path.open("rb") as fh:
        for offset in offsets:
            fh.seek(offset)
            buf = fh.read(SNIFF_CAP)
            if offset:
                # Mid-file, so the read opened inside a record. Dropping to the
                # next newline costs one line and stops every window past the
                # first from reporting a short row it invented.
                cut = buf.find(b"\n")
                buf = buf[cut + 1:] if 0 <= cut < len(buf) - 1 else buf
            heads.append(buf)
    head = heads[0]

    # `size` is the real file size throughout, never the window's. `describe`
    # reads `size <= len(head)` as "this sample reached the end" and drops its
    # last line as a fragment otherwise -- so passing the window length would
    # tell it a 16 MiB slice of a 2 GB file was the whole thing, and it would
    # describe a truncated final record as a short row.
    in_memory = tuple(n for n in which if n != "duckdb")
    raw = sniffers.run(head, size, which=in_memory) if in_memory else {}
    if "duckdb" in which:
        # The real path: no temp copy, and no truncation for DuckDB to trip on.
        raw["duckdb"] = sniffers.duckdb_path(path, has_pipe=b"|" in head)

    views = dialects.normalise(raw)
    merged = dialects.consensus(views)
    record["sniffers"] = views
    record["dialect"] = merged["dialect"]
    record["agreement"] = merged["agreement"]
    record["contested"] = merged["contested"]

    # Taken off `custom` rather than recomputed: it already ran
    # `validate.probe_encoding` over this exact window, and a second call with
    # its own fallbacks is a second opinion that could differ from the one the
    # index and the selectors were built on.
    own = sniffers.verdict(raw, "custom")
    record["encoding"] = {
        "class": own.get("encoding_class"), "detected": own.get("encoding"),
        "bom": own.get("bom"), "wide": own.get("wide_encoding"),
        "magic": own.get("magic_kind"), "control_frac": own.get("control_frac"),
        "chardet": sniffers.verdict(raw, "chardet").get("encoding"),
    }

    # The later windows exist to contradict the first one. Only the delimiter
    # is compared: it is the field every sniffer has an opinion about and the
    # one whose drift makes a file unparseable under a single config.
    if len(offsets) > 1 and "custom" in which:
        tail = [sniffers.run(buf, size, which=("custom",))
                .get("custom", {}).get("delimiter") for buf in heads[1:]]
        first = record["dialect"]["delimiter"]
        record["window_delimiters"] = [first, *tail]
        if any(d != first for d in tail if d):
            record["contested"] = sorted({*record["contested"], "delimiter"})
            record["drift"] = True

    record["consistency"] = dialects.consistency(
        path, record["dialect"]["delimiter"], record["dialect"]["quote"],
        record["dialect"]["comment"])
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
          procs: int = 0, on_event: Any = None) -> Path:
    """Sniff every file of a dataset and write `datasets/<name>.json`.

    `members` is the dataset's kept downloads; `provenance` is how it was
    drawn. Both come from `download`, which owns that bookkeeping.
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
        entry.update(results.get(key, {"missing": True}))
        files[key] = entry

    ROOT.mkdir(parents=True, exist_ok=True)
    out = manifest_path(name)
    payload = {
        "dataset": provenance | {
            "name": name, "files": len(files),
            "bytes": sum(r["size"] for r in members),
            "sniffers": list(which), "sniff_cap": SNIFF_CAP,
            "sniffed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sniff_seconds": round(time.monotonic() - started, 1),
        },
        # Sorted so a rebuild produces the same bytes and a diff shows only
        # what actually changed.
        "files": dict(sorted(files.items())),
    }
    out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n",
                   encoding="utf-8")
    write_review(files, name)
    return out


def needs_review(files: dict[str, Any]) -> dict[str, Any]:
    """Files whose dialect nothing settled, minus those already decided.

    Three ways to land here, and they are different problems: the sniffers
    split on a field, none of them found a delimiter at all, or one dialect
    does not describe the whole file. The last is the one a head sample can
    never raise by itself.
    """
    decided = set(overrides())
    out = {}
    for key, entry in files.items():
        if key in decided or entry.get("missing"):
            continue
        share = (entry.get("consistency") or {}).get("modal_share")
        if (entry.get("contested") or entry["dialect"]["delimiter"] is None
                or (share is not None and share < 0.95)):
            out[key] = entry
    return out


def write_review(files: dict[str, Any], name: str) -> Path:
    """Scaffold `<name>.review.toml`: one commented block per unsettled file.

    Generated and disposable, and a *different file* from `overrides.toml` on
    purpose -- you copy a block across once you have decided, so the flow is
    one-way and regenerating can never overwrite a decision. That separation is
    the whole reason the tool needs no TOML writer for the file that matters.
    """
    queue = needs_review(files)
    out = review_path(name)
    ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Files in dataset {name!r} whose dialect the sniffers did not",
        "# settle. Generated -- this file is rewritten every time the dataset",
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
            lines.append(f"#   {sniffer:<10} delimiter={view['delimiter']!r} "
                         f"quote={view['quote']!r} comment={view['comment']!r}")
        con = entry.get("consistency") or {}
        if con.get("modal_share") is not None:
            lines.append(f"#   whole file  {con['records']:,} records, "
                         f"{con['modal_fields']} fields on "
                         f"{con['modal_share']:.1%} of them")
        if entry.get("drift"):
            lines.append(f"#   DRIFT       windows disagree: "
                         f"{entry['window_delimiters']}")
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
