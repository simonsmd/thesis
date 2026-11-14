"""Fetch sampled files, verify them, and keep going until N usable ones land.

Validation rejects a meaningful fraction -- dead links, LFS bandwidth quotas,
archives mislabelled .csv -- so a single pass of N would deliver fewer than N.
The loop instead holds a fixed per-stratum plan and re-draws only the deficit,
so rejections are replaced within the stratum that lost them and the shape of
the draw -- log-uniform or uniform -- survives.

Every rejection is recorded with its reason rather than silently dropped: the
reject profile is itself a finding about what "a CSV file on GitHub" means.

A run builds one *dataset*, named. Several coexist -- the benchmarking corpus
and the correctness corpus are two draws over the same pool with opposite
shapes, and targeted ones per selector alongside them -- so the bookkeeping is
per dataset: its own downloads log, its own manifest, its own idea of what has
been kept and rejected. What is *not* per dataset is the bytes. The store is
content-addressed and shared, so a file both corpora draw is fetched once and
named by both manifests, which is also what makes re-running a dataset cheap.
"""
from __future__ import annotations

import hashlib
import queue
import threading
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from . import config, index, log, sample, selectors, validate
from .client import GitHub
from .util import run_workers

CORPUS = config.DATA_ROOT / "corpus"
MANIFESTS = CORPUS / "manifests"
# The dataset a plain `download` builds: the minimally-filtered pool, drawn
# log-uniformly over size, which is the benchmarking corpus.
DEFAULT_DATASET = "bench"
# What locates a Kaggle file inside its dataset's zip archive, see
# `kaggle.member_of`.
ARCHIVE_FIELDS = ("method", "crc", "csize", "header_offset")


def log_name(dataset: str) -> str:
    return f"downloads-{dataset}"


def manifest_path(dataset: str) -> Path:
    return MANIFESTS / f"{dataset}.jsonl"


def datasets() -> list[str]:
    """Dataset names that have a downloads log, whether or not it kept anything."""
    root = config.DATA_ROOT
    prefix = log_name("")
    return sorted(p.name[len(prefix):] for p in root.glob(f"{prefix}*")
                  if p.is_dir())


def _target(content_key: str) -> Path:
    return CORPUS / content_key[:2] / content_key


def outcomes(dataset: str = DEFAULT_DATASET) -> tuple[set[str], set[str]]:
    """(kept, rejected) content keys from previous runs of this dataset.

    The newest record per key decides, so a file `drop` removed after it was
    kept counts as rejected -- which also keeps it out of every later draw.
    """
    kept, rejected = set(), set()
    for rec in log.read(log_name(dataset)):
        if rec.get("state") == "ok":
            kept.add(rec["content_key"])
            rejected.discard(rec["content_key"])
        else:
            rejected.add(rec["content_key"])
            kept.discard(rec["content_key"])
    return kept, rejected


def content_id(path: Path, size: int, is_lfs: bool,
               source: str = "github") -> str:
    """The key this file's bytes hash to, in whichever scheme names it.

    Three schemes, one per way a file reaches the corpus. For LFS the pointer's
    oid *is* the sha256 of the content; for a plain GitHub blob it is git's own
    sha1 over the expected length; for Kaggle it is the CRC32 and length the
    bundle's central directory recorded, which is the only content identity
    that source exposes. One function so the local and the just-downloaded case
    cannot drift apart in any of the three.
    """
    with path.open("rb") as fh:
        chunks = iter(lambda: fh.read(1 << 20), b"")
        if source == "kaggle":
            from . import kaggle
            from .zipindex import U32_MAX

            crc = 0
            length = 0
            for chunk in chunks:
                crc = zlib.crc32(chunk, crc)
                length += len(chunk)
            return kaggle.content_key(crc & U32_MAX, length)
        if is_lfs:
            digest = hashlib.sha256()
            for chunk in chunks:
                digest.update(chunk)
            return digest.hexdigest()
        return validate.git_blob_sha1(size, chunks)


def _already_here(dest: Path, row: dict[str, Any],
                  source: str = "github") -> bool:
    """Is the file the index asked for already in the store?

    The store is content-addressed and shared across datasets, so the second
    corpus to draw a file finds it there. Confirmed by a local rehash rather
    than trusting the filename.
    """
    if not dest.exists() or dest.stat().st_size != row["size"]:
        return False
    return content_id(dest, row["size"], bool(row["is_lfs"]),
                      source) == row["content_key"]


def _rejection(path: Path, row: dict[str, Any], verify: Any) -> str | None:
    """Why these (identity-checked) bytes cannot join this dataset, or None.

    Runs on reused files too: the selector a second corpus draws under is not
    the one that admitted the file the first time.
    """
    reason = validate.check(path)
    if reason:
        return reason
    if verify is not None:
        why = verify(path, row)
        if why:
            return f"verify/{why}"
    return None


def fetch_one(client: Any, row: dict[str, Any], verify: Any = None,
              source: str = "github") -> dict[str, Any]:
    """Download one file from whichever source the corpus is drawn from.

    The two paths differ only in how bytes are addressed and how identity is
    checked; everything after -- base validation, the selector's whole-file
    gate, the content-addressed store -- is shared, which is what keeps a
    corpus definition meaning the same thing whichever pool it came from.
    """
    if source == "kaggle":
        return _kaggle_fetch_one(client, row, verify)
    return _github_fetch_one(client, row, verify)


def _kaggle_fetch_one(kg: Any, row: dict[str, Any],
                      verify: Any = None) -> dict[str, Any]:
    """Read one member out of its dataset's bundle.

    The version recorded at index time is pinned, and a pin that no longer
    resolves is the drift itself: Kaggle does not reliably retain superseded
    versions, so there is no older bundle to fall back to. Those are rejected
    as `version_drift` rather than silently re-fetched at whatever the dataset
    says today -- the row's size, bucket, delimiter and encoding all describe
    the version that was indexed, so admitting the new bytes would put a
    mislabelled file in the corpus. The redraw loop refills the stratum, so the
    shape of the draw survives the loss.
    """
    from . import kaggle

    key = row["content_key"]
    dest = _target(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    base = {
        "content_key": key, "repo_id": row["repo_id"],
        "full_name": row["full_name"], "path": row["path"],
        "bucket": row["bucket"], "stratum": row.get("stratum", row["bucket"]),
        "size": row["size"], "ref": row.get("ref"),
        "version": row.get("version"), "is_lfs": False,
        "source": "kaggle", "ts": int(time.time()),
        # Where the file sits in its dataset's archive, so that `fetch` can
        # get it again from the dataset definition alone.
        **{name: row[name] for name in ARCHIVE_FIELDS},
    }

    if _already_here(dest, row, "kaggle"):
        reason = _rejection(dest, row, verify)
        if reason:
            return base | {"state": "rejected", "reason": reason}
        return base | {
            "state": "ok", "bytes": 0, "reused": True, "sha256": None,
            "local_path": str(dest.relative_to(config.DATA_ROOT)),
        }

    url, status = kg.signed_url_cached(row["full_name"], row.get("version"))
    if not url:
        return base | {"state": "rejected", "reason": "version_drift",
                       "http": status}

    data = kaggle.read_member(kg, url, kaggle.member_of(row))
    if data is None:
        # `read_member` already checked the CRC, so this is a failed transfer
        # or an archive whose layout no longer matches what was indexed --
        # which, with the version pinned, means the pin did not hold.
        return base | {"state": "rejected", "reason": "fetch_failed"}

    try:
        tmp.write_bytes(data)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return base | {"state": "error", "reason": f"io({exc.errno})"}

    reason = _rejection(tmp, row, verify)
    if reason:
        tmp.unlink(missing_ok=True)
        return base | {"state": "rejected", "reason": reason}

    tmp.replace(dest)
    return base | {
        "state": "ok", "bytes": len(data), "sha256": None, "reused": False,
        "local_path": str(dest.relative_to(config.DATA_ROOT)),
    }


def _github_fetch_one(gh: GitHub, row: dict[str, Any],
                      verify: Any = None) -> dict[str, Any]:
    """Download, verify identity, validate content. Never raises."""
    key = row["content_key"]
    dest = _target(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    base = {
        "content_key": key, "repo_id": row["repo_id"], "full_name": row["full_name"],
        "path": row["path"], "bucket": row["bucket"],
        "stratum": row.get("stratum", row["bucket"]), "size": row["size"],
        # The commit the file was read at, so the dataset definition can be
        # re-fetched exactly.
        "ref": row["ref"], "is_lfs": bool(row["is_lfs"]), "ts": int(time.time()),
    }

    if _already_here(dest, row):
        reason = _rejection(dest, row, verify)
        if reason:
            return base | {"state": "rejected", "reason": reason}
        return base | {
            "state": "ok", "bytes": 0, "reused": True,
            "sha256": key if row["is_lfs"] else None,
            "local_path": str(dest.relative_to(config.DATA_ROOT)),
        }

    # Pinned to the commit the tree scan saw. Unpinned, a repo pushed to since
    # the scan would serve different bytes and fail the identity check below,
    # biasing the corpus towards dormant repos.
    host = ("https://media.githubusercontent.com/media" if row["is_lfs"]
            else "https://raw.githubusercontent.com")
    url = f"{host}/{row['full_name']}/{row['ref']}/{_quote(row['path'])}"

    try:
        with tmp.open("wb") as sink:
            result = gh.stream(url, sink)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        return base | {"state": "error", "reason": f"io({exc.errno})"}

    if result is None:
        tmp.unlink(missing_ok=True)
        # LFS content is billed to the *repo owner's* bandwidth quota, so an
        # over-quota repo fails here through no fault of ours.
        return base | {"state": "rejected",
                       "reason": "fetch_failed" + ("_lfs_quota?" if row["is_lfs"] else "")}

    written, sha256 = result
    size = tmp.stat().st_size

    # Identity, with no tolerance. The fetch is pinned, so a mismatch means
    # this is not the indexed file at all: its row's size, bucket and dialect
    # describe something else, and the content-addressed store would file it
    # under a key it does not hash to.
    #
    # `stream` already hashed the bytes as they went past, so the LFS case
    # costs nothing; a plain blob needs git's framing and one re-read.
    actual = sha256 if row["is_lfs"] else content_id(tmp, size, is_lfs=False)
    if actual != key:
        tmp.unlink(missing_ok=True)
        return base | {
            "state": "rejected",
            "reason": "lfs_oid_mismatch" if row["is_lfs"] else "content_mismatch",
            "indexed_size": row["size"], "actual_size": size,
        }

    # The selector's verifier is the only stage that sees the whole file, so
    # anything the head sample cannot settle belongs there -- including
    # whether an *unprobed* file is text at all.
    reason = _rejection(tmp, row, verify)
    if reason:
        tmp.unlink(missing_ok=True)
        return base | {"state": "rejected", "reason": reason}

    tmp.replace(dest)
    return base | {
        "state": "ok", "bytes": written, "sha256": sha256, "reused": False,
        "local_path": str(dest.relative_to(config.DATA_ROOT)),
    }


def _quote(path: str) -> str:
    from requests.utils import quote

    return quote(path)


def run(
    gh: Any,
    n: int,
    seed: int = 0,
    workers: int = 8,
    per_repo: int = 2,
    per_owner: int = 5,
    max_rounds: int = 12,
    select: str | None = "main",
    utf8: bool = False,
    mode: str = sample.LOG_UNIFORM,
    dataset: str = DEFAULT_DATASET,
    max_bucket: int | None = None,
    min_size: int | None = None,
    max_size: int | None = None,
    sniff: bool = True,
    sniffers_which: tuple[str, ...] | None = None,
    sniff_procs: int = 0,
    source: str = "github",
    on_event: Any = None,
    review: str = "disagreements",
) -> dict[str, Any]:
    emit = on_event or (lambda *a, **k: None)
    if min_size is not None and max_size is not None and min_size > max_size:
        return {"error": f"--min-size {min_size:,} is above "
                         f"--max-size {max_size:,}: nothing can be drawn"}
    CORPUS.mkdir(parents=True, exist_ok=True)
    kept, rejected = outcomes(dataset)

    selector = selectors.resolve(select, utf8=utf8)
    verify = selector.verify
    # Built once and reused by every round below. Lazy: the plan collects only
    # per-stratum counts and each draw collects only the stratum it is filling.
    pool = sample.candidates(select=selector, source=source)
    if max_bucket is not None:
        # A byte ceiling expressed as an octave: under a log-uniform draw equal
        # counts per bucket means the top one or two carry almost all the
        # bytes, so capping the axis is what actually bounds the download. It
        # bounds a uniform draw too, just far less dramatically -- most of that
        # pool is small already.
        pool = pool.filter(pl.col("bucket") <= max_bucket)
    # Exact byte bounds. An octave they cut part-way just has less supply,
    # which the water-filling below already copes with: the log-uniform shape
    # holds across the octaves wholly inside the range.
    if min_size is not None:
        pool = pool.filter(pl.col("size") >= min_size)
    if max_size is not None:
        pool = pool.filter(pl.col("size") <= max_size)
    # The per-stratum target, computed once and held fixed across the rounds
    # below: topping up with a freshly water-filled quota would spread the
    # replacements evenly instead of refilling the specific strata whose files
    # were rejected, and the shape of the draw is exactly what that would lose.
    available = sample.supply(sample.candidates(rejected, selector, pool=pool,
                                               source=source),
                              mode, sizes=(mode == sample.UNIFORM))
    quota = sample.allocate({s: n_rows for s, (n_rows, _) in available.items()}, n)
    if not quota:
        if selector.needs_probe:
            # This selector picks on probe metadata, so it can only draw files
            # that have some. An empty pool is then almost always coverage
            # rather than the filter, and saying which beats "no candidates".
            cov = sample.coverage(source)
            probed = int(cov["probed"].sum()) if cov.height else 0
            total = int(cov["unique_files"].sum()) if cov.height else 0
            if probed < total:
                return {"error": f"pool is {probed:,}/{total:,} probed "
                                 f"({probed / max(total, 1):.1%}) -- "
                                 f"{selector.name!r} selects on probe metadata, "
                                 f"so only probed files are drawable. Run "
                                 f"`csvscrape probe`, or pick a selector that "
                                 f"does not need it (`csvscrape selectors`)."}
        return {"error": f"no candidates for selector {selector.name!r} "
                         f"-- run compact first, or widen the selector"}
    emit("plan", strata=len(quota), files=sum(quota.values()), mode=mode,
         dataset=dataset, selector=selector.name, source=source,
         bytes=sample.projected_bytes(quota, mode, available))

    # Counted from the current members rather than every `ok` ever logged, so
    # a stratum that lost a file to `drop` shows a deficit and is refilled.
    have = Counter()
    for rec in members(dataset):
        # Older records carry only the size bucket, which is what a
        # log-uniform stratum is anyway.
        have[rec.get("stratum", rec["bucket"])] += 1

    reasons: Counter = Counter()
    stats = {"dataset": dataset, "selector": selector.name, "mode": mode,
             "source": source,
             "kept": len(kept), "rejected": 0, "rounds": 0, "bytes": 0,
             "reused": 0}
    stopping = threading.Event()
    # Stop/accept handshake for `util.run_workers`.
    accepting = threading.Event()
    accepting.set()

    with log.Writer(log_name(dataset), rotate=500) as dl_log:
        for rnd in range(max_rounds):
            deficit = {b: quota[b] - have.get(b, 0) for b in quota
                       if quota[b] - have.get(b, 0) > 0}
            if not deficit:
                break
            rows, short = sample.draw(
                deficit, seed=seed + rnd, per_repo=per_repo, per_owner=per_owner,
                exclude=kept | rejected, select=selector, pool=pool, mode=mode,
            )
            if rows.is_empty():
                break
            stats["rounds"] = rnd + 1
            emit("round", n=rnd + 1, wanted=sum(deficit.values()), drawn=rows.height)

            work: queue.Queue = queue.Queue()
            for row in rows.to_dicts():
                work.put(row)
            lock = threading.Lock()

            def worker() -> None:
                while not stopping.is_set():
                    try:
                        row = work.get_nowait()
                    except queue.Empty:
                        return
                    record = fetch_one(gh, row, verify, source)
                    with lock:
                        if not accepting.is_set():
                            # Interrupted, and this fetch missed the grace
                            # period. An unrecorded file is an undownloaded
                            # one: the next run draws it again.
                            return
                        dl_log.write(record)
                        if record["state"] == "ok":
                            kept.add(record["content_key"])
                            have[record["stratum"]] += 1
                            stats["kept"] += 1
                            stats["bytes"] += record.get("bytes", 0)
                            stats["reused"] += int(record.get("reused", False))
                        else:
                            rejected.add(record["content_key"])
                            stats["rejected"] += 1
                            reasons[record["reason"]] += 1
                        emit("download", kept=stats["kept"], target=n,
                             rejected=stats["rejected"], name=record["full_name"],
                             state=record["state"], reason=record.get("reason", ""))

            # Interrupted: what was in flight is recorded; draw no more rounds.
            if not run_workers(
                    [threading.Thread(target=worker, daemon=True)
                     for _ in range(workers)],
                    stats, stopping=stopping, accepting=accepting, lock=lock):
                break

    stats["reasons"] = dict(reasons.most_common())
    stats["shortfall"] = {b: quota[b] - have.get(b, 0) for b in quota
                          if have.get(b, 0) < quota[b]}
    stats["plan"] = quota
    stats["manifest"] = str(_write_manifest(dataset))

    # Sniffing runs here rather than inside `fetch_one`: the sniffers are
    # CPU-bound and the fetch workers are threads, and a pass over the *store*
    # is re-runnable without the network -- `csvscrape download` on a complete
    # dataset re-sniffs and rewrites the manifest without fetching anything.
    # Nothing is cached, so a re-run pays it again; `--no-sniff` skips it.
    if sniff:
        from . import dataset as dataset_mod

        kept_rows = members(dataset)
        provenance = {
            "selector": selector.name, "sampling": mode, "utf8": utf8,
            "seed": seed, "count": n, "max_bucket": max_bucket,
            "min_size": min_size, "max_size": max_size,
            "source": source,
            "index_watermark": index.heads_watermark(source),
        }
        stats["definition"] = str(dataset_mod.build(
            dataset, kept_rows, provenance, procs=sniff_procs, on_event=emit,
            review=review,
            **({"which": sniffers_which} if sniffers_which else {})))
        stats["review"] = str(dataset_mod.review_path(dataset))
    return stats


def fetch(client: Any, dataset: str, workers: int = 8,
          on_file: Any = None) -> dict[str, Any]:
    """Get the files of an existing dataset definition into the store.

    This is how a dataset is rebuilt on another machine from
    `datasets/<name>.json` alone: no index, no sampling. A file that is
    already in the store is checked against its content hash and kept. A file
    that can no longer be fetched is reported and not replaced by another one,
    so the result is either the same dataset or a list of what is missing.

    `on_file(record)` is called after every file.
    """
    import json
    from concurrent.futures import ThreadPoolExecutor

    from . import dataset as dataset_mod

    definition = json.loads(
        dataset_mod.manifest_path(dataset).read_text(encoding="utf-8"))
    source = definition["dataset"].get("source", "github")
    CORPUS.mkdir(parents=True, exist_ok=True)

    def one(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        key, entry = item
        row = {
            "content_key": key, "repo_id": None, "full_name": entry["repo"],
            "path": entry["path"], "size": entry["size"],
            "bucket": entry["bucket"], "ref": entry.get("ref"),
            "is_lfs": bool(entry.get("lfs")),
        }
        if source == "kaggle":
            archive = entry.get("archive")
            if not archive:
                return row | {"state": "rejected",
                              "reason": "no_archive_position_in_definition"}
            row |= archive
        return fetch_one(client, row, None, source)

    stats: dict[str, Any] = {"dataset": dataset, "source": source,
                             "files": len(definition["files"]), "reused": 0,
                             "fetched": 0, "bytes": 0, "failed": []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(one, definition["files"].items()):
            if rec["state"] != "ok":
                stats["failed"].append(rec)
            elif rec.get("reused"):
                stats["reused"] += 1
            else:
                stats["fetched"] += 1
                stats["bytes"] += rec["bytes"]
            if on_file:
                on_file(rec)
    return stats


def members(dataset: str = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """What is currently in this dataset: newest record per key, kept ones only.

    Replaying the log rather than reading the manifest, so that this and the
    manifest cannot disagree -- the manifest is written from exactly this.
    """
    latest: dict[str, dict] = {}
    for rec in log.read(log_name(dataset)):
        if rec.get("state") == "ok":
            # Older records carry a download-time `shape`, which nothing reads;
            # `dataset.build` describes each file itself.
            rec.pop("shape", None)
            latest[rec["content_key"]] = rec
        else:
            latest.pop(rec["content_key"], None)
    return sorted(latest.values(), key=lambda r: (r["bucket"], r["content_key"]))


def drop(dataset: str, keys: list[str], reason: str) -> Path:
    """Remove files from a dataset by hand; the next `download` refills them.

    Recorded as a rejection like any other, with the reason prefixed
    `manual/`, so the reject profile still accounts for every file that left
    and the draw never picks it again. The stratum is carried over from the
    kept record: that is what the refill counts against.
    """
    current = {r["content_key"]: r for r in members(dataset)}
    missing = [k for k in keys if k not in current]
    if missing:
        raise SystemExit(f"not in dataset {dataset!r}: {', '.join(missing)}")
    with log.Writer(log_name(dataset)) as dl_log:
        for key in keys:
            rec = current[key]
            dl_log.write({
                "content_key": key, "repo_id": rec.get("repo_id"),
                "full_name": rec["full_name"], "path": rec["path"],
                "bucket": rec["bucket"],
                "stratum": rec.get("stratum", rec["bucket"]),
                "size": rec["size"], "state": "rejected",
                "reason": f"manual/{reason}", "ts": int(time.time()),
            })
    return _write_manifest(dataset)


def _write_manifest(dataset: str = DEFAULT_DATASET) -> Path:
    """Flat manifest of this dataset's corpus, newest record wins.

    One per dataset, because the store underneath is shared: the manifest is
    the only thing that says which of the files on disk belong to this draw.
    """
    import json

    path = manifest_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in members(dataset):
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return path


def summary(stats: dict[str, Any]) -> str:
    """Human-readable outcome, with every rejection reason accounted for."""
    if "error" in stats:
        return stats["error"]
    from .index import human, pow2

    reused = f" · {stats['reused']:,} already in the store" if stats.get("reused") else ""
    out = [
        f"{stats['dataset']}: {stats['kept']:,} files · {stats['mode']} draw over "
        f"{stats['selector']!r} in {stats.get('source', 'github')} · "
        f"{human(stats['bytes'])} fetched this run{reused} · "
        f"{stats['rounds']} round(s)",
        f"manifest: {stats['manifest']}",
    ]
    # Called out on its own line: drift is not a file failing a quality gate
    # but the index going stale, and it says whether to re-run `kaggle-bundle`.
    drift = stats["reasons"].get("version_drift", 0) if stats.get("reasons") else 0
    if drift:
        out.append(f"\n{drift:,} rejected as version_drift: the dataset was "
                   f"republished since indexing and Kaggle no longer serves "
                   f"the indexed version. Re-run `kaggle-bundle` to refresh.")
    if stats["rejected"]:
        out.append(f"\nrejected {stats['rejected']:,}:")
        width = max(len(r) for r in stats["reasons"])
        for reason, count in stats["reasons"].items():
            out.append(f"    {reason:<{width}}  {count:>5,}")
    if stats["shortfall"]:
        out.append("\nshort of plan (pool exhausted):")
        for stratum, missing in sorted(stats["shortfall"].items()):
            want = stats["plan"][stratum]
            where = ("the whole pool  " if stratum == sample.WHOLE_POOL else
                     f"bucket {stratum:>2}  {pow2(stratum)} – {pow2(stratum + 1):<9}")
            out.append(f"    {where}  {want - missing:>4}/{want:<4} (-{missing})")
    return "\n".join(out)
