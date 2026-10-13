"""Compaction: raw logs -> a single Parquet index of candidate CSV files.

The tree logs keep every kept blob of every repo. Exploding and filtering that
on every question would be the one genuinely expensive step in the pipeline,
and it is a streaming transformation rather than a query -- so it runs once,
here, and everything downstream reads Parquet instead. The probe verdicts are
folded in as columns at the same time.

Memory is the constraint at full scale, which is why the build is staged and
bucketed (see `build`) rather than one query.

Rebuilding is always safe: the logs are the source of truth and this file is
derived, so widening INDEX_EXTENSIONS or changing a rule means re-running this,
never re-scraping.
"""
from __future__ import annotations

import math
import resource
import shutil
import sys
import time
from pathlib import Path

import polars as pl

from . import config, log

INDEX_PATH = config.DATA_ROOT / "index" / "files.parquet"
# The highest heads-log rec_id folded into INDEX_PATH. `probe` reads the index
# for everything up to this point and only the log beyond it, which is what
# keeps "what have I already probed" from re-parsing millions of ndjson records
# on every invocation.
WATERMARK_PATH = config.DATA_ROOT / "index" / "heads_watermark.txt"
REPOS_PATH = config.DATA_ROOT / "index" / "repos.parquet"


def _ext_filter(column: pl.Expr) -> pl.Expr:
    lowered = column.str.to_lowercase()
    predicate = lowered.str.ends_with(config.INDEX_EXTENSIONS[0])
    for ext in config.INDEX_EXTENSIONS[1:]:
        predicate = predicate | lowered.str.ends_with(ext)
    return predicate


def _lfs_frame() -> pl.LazyFrame:
    """Resolved pointers, one row per pointer blob, newest record winning.

    Keyed by the blob, not by where it was found. A pointer blob is exactly
    `version .../oid sha256:.../size N`, so its sha1 determines the oid, the
    real size, and whether it is a pointer at all. Keyed by content, one
    resolution serves every repo holding the blob, and a record can never go
    stale against a path whose file has since changed.
    """
    empty = pl.DataFrame(
        schema={
            "blob_sha": pl.String, "oid": pl.String,
            "state": pl.String, "real_size": pl.Int64,
        }
    ).lazy()
    parts = log.sealed("lfs")
    if not parts:
        return empty
    return log.newest_per(
        pl.scan_ndjson(parts), ["blob_sha"]
    ).select(
        pl.col("blob_sha"),
        pl.col("oid"),
        pl.col("state"),
        pl.col("size").cast(pl.Int64).alias("real_size"),
    )


def _shard(i: int, path: str) -> pl.LazyFrame:
    """One sealed trees shard with a stable per-record key.

    `rec_id` is stamped by the writer and is exact. Older logs without it fall
    back to the shard number and row index; the shard number is part of the
    fallback so that the key is the same whether the record was reached
    through `_trees` or through this shard alone.

    Older logs also lack `ext` and `modes`, and a missing column fails the
    whole scan, so both are supplied as nulls. `modes` explodes in parallel
    with `paths`, so it needs a list of nulls of the same length per row
    (`_projection` keeps an unknown mode).
    """
    frame = pl.scan_ndjson([path])
    names = frame.collect_schema().names()
    if "ext" not in names:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("ext"))
    if "modes" not in names and "paths" in names:
        frame = frame.with_columns(
            pl.col("paths").list.eval(pl.lit(None, dtype=pl.String)).alias("modes")
        )
    frame = frame.with_row_index("_row")
    fallback = pl.format(f"row:{i:05d}:{{}}", pl.col("_row"))
    key = pl.coalesce(pl.col("rec_id"), fallback) if "rec_id" in names else fallback
    return frame.with_columns(key.alias("_key"))


def _shards() -> list[pl.LazyFrame]:
    """Every sealed trees shard, oldest first, each keyed by `_shard`."""
    return [_shard(i, path) for i, path in enumerate(log.sealed("trees"))]


def _trees() -> pl.LazyFrame:
    """The whole trees log with a stable per-record key.

    A concatenation of per-shard scans rather than one multi-file scan, so
    the keys are identical to what `_shard` hands the staging loop, which
    reads the log one shard at a time.
    """
    return pl.concat(_shards(), how="diagonal_relaxed")


def _winning_keys() -> pl.Series:
    """Record keys of the newest successful scan of each repo.

    A repo re-scanned after a transient failure appears twice; the writers are
    dumb appenders, so uniqueness is enforced on read. Deduping the wide frame
    would buffer every path/sha/size array in the log (`unique` is blocking),
    so this pre-pass projects a few scalar columns and the blocking step sees
    one narrow row per repo.
    """
    trees = _trees()
    has_rec_id = "rec_id" in trees.collect_schema().names()
    meta = (
        trees
        .select("repo_id", "ts", "state", "_key", *(["rec_id"] if has_rec_id else []))
        .filter(pl.col("state") == "ok")
        .collect()
    )
    return log.newest_per(meta.lazy(), ["repo_id"]).collect()["_key"]


# Dedup is a global operation over every distinct content in the pool, and a
# single hash table across all of it does not fit in memory at full scale.
# Splitting the keyspace into N buckets bounds that table to a fraction of the
# pool. The staging pass writes each bucket to its own files, so raising N
# costs nothing but file count: every dedup pass reads exactly its own rows.
#
# The split is on a hash of the key rather than its leading characters, so it
# covers the whole keyspace whatever the keys look like (Kaggle keys are not
# bare hex).
DEDUP_SHARDS = 64


def _winning(keep: pl.Series, frame: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """The tree records whose keys are in `keep`, in scan order.

    A semi join rather than `is_in(keep.implode())`: combined with the row
    index `_shard` adds, that filter made the streaming engine hold every
    decoded shard at once, where the join stays flat.
    """
    return (_trees() if frame is None else frame).join(
        keep.rename("_key").to_frame().lazy(),
        on="_key", how="semi", maintain_order="left",
    )


def _projection(keep: pl.Series, lfs: pl.LazyFrame,
                frame: pl.LazyFrame | None = None) -> pl.LazyFrame:
    """Tree records exploded to one row per (repo, path) CSV occurrence.

    `lfs` is the resolved-pointer frame, already collected: the streaming
    engine builds its join hash table on whichever side it estimates smaller,
    and an ndjson scan has no row estimate, so given a lazy frame it
    materialises the exploded tree side instead. `build` collects it once
    rather than once per shard.
    """
    size = pl.coalesce(pl.col("real_size"), pl.col("raw_size"))
    return (
        _winning(keep, frame)
        # A repo with no blobs has empty arrays. Polars 1.x explodes those to
        # one all-null row and 2.0 will explode them to nothing; the ext
        # filter drops the null row either way, so the 2.0 behaviour is named
        # here to keep the output identical across the change.
        .explode(["paths", "shas", "sizes", "modes"], empty_as_null=False)
        .rename({"paths": "path", "shas": "blob_sha", "sizes": "raw_size",
                 "modes": "mode"})
        .filter(_ext_filter(pl.col("path")))
        # Symlinks leave the pool here rather than in a selector, because no
        # corpus wants one. Git stores a symlink as a blob whose bytes are the
        # target path (mode 120000), so it would index as a tiny "CSV"
        # containing a path; what it points at is either indexed under its
        # own row or outside the tree. A null mode (older logs) is kept.
        .filter(pl.col("mode").ne_missing("120000"))
        # On the blob, so one resolution serves every occurrence of it,
        # including occurrences that were never candidates themselves.
        .join(lfs, on="blob_sha", how="left")
        .with_columns(
            size.alias("size"),
            (pl.col("state_right").fill_null("plain") == "lfs").alias("is_lfs"),
            pl.coalesce(pl.col("oid"), pl.col("blob_sha")).alias("content_key"),
            # Pointer-sized with no resolution record at all was never checked,
            # so it may be an LFS file we failed to read and must not be counted
            # as a 130-byte CSV. Checked-and-genuine carries state='plain'.
            (
                pl.col("state_right").is_null()
                & pl.col("raw_size").is_between(
                    config.LFS_POINTER_MIN, config.LFS_POINTER_MAX
                )
            ).alias("lfs_suspect"),
        )
        .filter(pl.col("size") >= config.INDEX_MIN_BYTES)
        .with_columns(
            # log2(0) is -inf, and casting that to Int32 raises and fails the
            # whole sink. Empty and 1-byte files share bucket 0; `size` stays
            # exact for anything wanting only the empties.
            pl.when(pl.col("size") < 1)
            .then(pl.lit(0))
            .otherwise(pl.col("size").log(2).floor())
            .cast(pl.Int32)
            .alias("bucket"),
            pl.col("full_name").str.split("/").list.first().alias("owner"),
            # Error records carry no `ref` and never survive the winning-keys
            # join; "HEAD" keeps a null out of any URL regardless.
            pl.coalesce(pl.col("ref"), pl.lit("HEAD")).alias("ref"),
        )
        .select(
            "repo_id", "owner", "full_name", "path", "blob_sha", "content_key",
            "size", "raw_size", "bucket", "is_lfs", "lfs_suspect", "tier",
            "ext", "truncated", "ref",
        )
    )


# The columns that describe the representative occurrence, as opposed to the
# content itself. `size`/`raw_size`/`bucket` are properties of the bytes and
# agree across a group by construction; they ride along for a single sort.
# `ref` is the snapshot *this* occurrence was read from, so it travels with
# full_name/path.
_REPRESENTATIVE = [
    "repo_id", "owner", "full_name", "path", "blob_sha", "size", "raw_size",
    "bucket", "tier", "ext", "truncated", "ref",
]


def _dedup(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Collapse occurrence rows to one row per distinct content.

    Identical bytes in several repos are one candidate. Doing the collapse
    once, here, is what lets the readers stay lazy.

    The representative is the occurrence with the lowest (full_name, path),
    chosen by a per-group sort rather than by `first()`: the streaming
    group-by does not honour input order, so "first" would change between
    rebuilds of the same logs, and with it the owner attribution `sample`
    caps on.

    `is_lfs` and `lfs_suspect` are aggregated across the group instead of
    read off the representative: a small file confirmed plain in one repo and
    never checked in another is known genuine. `occurrences` and `n_repos`
    keep how duplicated the content is.
    """
    return (
        frame.group_by("content_key")
        .agg(
            pl.struct(_REPRESENTATIVE)
            .sort_by(["full_name", "path"])
            .first()
            .alias("_rep"),
            pl.col("is_lfs").any(),
            pl.col("lfs_suspect").all(),
            pl.len().alias("occurrences"),
            pl.col("repo_id").n_unique().alias("n_repos"),
        )
        .unnest("_rep")
        .select(
            "content_key", "repo_id", "owner", "full_name", "path", "blob_sha",
            "size", "raw_size", "bucket", "is_lfs", "lfs_suspect", "tier",
            "ext", "truncated", "ref", "occurrences", "n_repos",
        )
    )


def heads_watermark() -> str | None:
    """Highest heads rec_id already folded into the index, if any."""
    try:
        return WATERMARK_PATH.read_text().strip() or None
    except OSError:
        return None


def _metadata_rows() -> tuple[pl.LazyFrame | None, str | None]:
    """Probe verdicts, flattened, one row per *record* -- plus the watermark.

    Returned alongside the highest rec_id consumed so `build` can record what
    it folded in. A wrong watermark costs work, never a wrong answer: too old
    and `probe` re-reads log records it did not need to, too new and it
    re-probes files whose records were skipped. The log stays the source of
    truth and readers dedupe.

    Undeduped on purpose: `build` dedups a bucket at a time. See `_stage_meta`.
    """
    from . import probe

    parts = log.sealed(probe.HEADS_LOG)
    if not parts:
        return None, None
    frame = probe.metadata_rows()
    heads = pl.scan_ndjson(parts)
    high = (heads.select(pl.col("rec_id").max()).collect().item()
            if "rec_id" in heads.collect_schema().names() else None)
    return frame, high


def _stage_meta(rows: pl.LazyFrame, path: Path) -> None:
    """Copy the probe records to one Parquet file, so they can be read by bucket.

    The point is to get the records out of ndjson and onto something a bucket
    can be filtered off cheaply. This copy streams (it is only a projection);
    the dedup happens in `build`'s bucket loop, where each pass sees a
    fraction of the log. Deduping the whole log in one pass does not fit in
    memory at full scale, and re-reading typed, columnar Parquet once per
    bucket is cheap.
    """
    rows.sink_parquet(path)


def _with_meta(frame: pl.LazyFrame, path: Path, b: int, shards: int) -> pl.LazyFrame:
    """Attach the probe verdicts for one content-key bucket to `frame`.

    The bucket expression is the one `build` staged with, so the records for
    every key in `frame` are in this bucket and no other: the left join sees
    the whole of each key's history and `dedupe` picks the same winner it
    would have picked over the entire log.

    A pool that was never probed has no file here, and then the index simply
    has no probe columns -- the same frame comes back untouched, which is what
    the readers' `if "fetched" in names` fallbacks already expect.
    """
    from . import probe

    if not path.exists():
        return frame
    rows = pl.scan_parquet(path).filter(
        (pl.col("content_key").hash(seed=0) % shards) == b
    )
    return frame.join(probe.dedupe(rows), on="content_key", how="left",
                      maintain_order="left")


def size_bucket(size: int) -> int:
    """The octave a file of `size` bytes belongs to: floor(log2(size)).

    The scalar twin of the Polars expression in `_projection`, and it must
    stay equal to it. `bit_length` rather than log2 keeps it exact at every
    size.
    """
    return 0 if size < 1 else int(size).bit_length() - 1


def _phase(name: str) -> None:
    """One stderr line per phase: elapsed and peak RSS so far.

    `compact` is the one step that can exhaust a machine's memory, and a
    process the OOM killer took leaves nothing behind. Printing each phase as
    it begins says where it died.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"compact: {name} (t={time.monotonic() - _T0:.0f}s, "
          f"peak rss {peak:,.0f} MB)", file=sys.stderr, flush=True)


_T0 = 0.0


def build(out: Path = INDEX_PATH, shards: int = DEDUP_SHARDS) -> Path:
    """Compact the logs into the Parquet index, in three streaming passes.

    Phase 1 stages the exploded occurrence rows, one trees shard at a time,
    each written out already split by dedup bucket. Phase 2 dedups one bucket
    at a time, so the hash table never spans the whole pool and each pass
    reads only its own rows, joins that bucket's probe verdicts on, and writes
    the result split by size bucket. Phase 3 concatenates those splits in
    size-bucket order, a straight streaming copy that leaves the index
    clustered by bucket.

    The probe verdicts are bucketed by the same expression, and joined inside
    phase 2 rather than over the finished index, for the same reason the
    occurrences are: one bucket of them fits and the whole log does not.

    One shard at a time is what bounds the peak. Polars decodes a whole zstd
    shard per thread, and a multi-file scan keeps one in flight per core,
    which is many GB before any work is done. Per shard, the threads
    parallelise inside the shard instead and the peak is a few times one
    shard's size.
    """
    if not log.sealed("trees"):
        raise SystemExit(
            "no sealed tree records yet: run `python -m csvscrape trees` first.\n"
            "A shard becomes readable when it rotates; a run that has not "
            "reached ROTATE_RECORDS yet has nothing for this to compact."
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    global _T0
    _T0 = time.monotonic()
    tree_shards = _shards()
    _phase(f"winning keys over {len(tree_shards)} tree shards")
    # Computed once and reused by both sinks below: it is a full pass over the
    # trees log, narrow but not free.
    keep = _winning_keys()
    _phase("lfs pointers")
    lfs = _lfs_frame().collect().lazy()
    staged = out.parent / "_staged"
    parts = out.parent / "_parts"
    meta_path = out.parent / "_meta.parquet"
    shutil.rmtree(staged, ignore_errors=True)
    shutil.rmtree(parts, ignore_errors=True)
    meta_path.unlink(missing_ok=True)
    parts.mkdir(parents=True)
    try:
        bucket = (pl.col("content_key").hash(seed=0) % shards).alias("_bucket")
        # Probe verdicts become columns of the index rather than a join every
        # reader repeats, which lets readers push predicates into Parquet
        # instead of building a hash table over every content key. Staged
        # here, consumed per bucket in phase 2.
        _phase("staging probe metadata")
        meta_rows, high = _metadata_rows()
        if meta_rows is not None:
            _stage_meta(meta_rows, meta_path)
        for i, frame in enumerate(tree_shards):
            _phase(f"staging shard {i + 1}/{len(tree_shards)}")
            _projection(keep, lfs, frame).with_columns(bucket).sink_parquet(
                pl.PartitionBy(staged / f"shard-{i:05d}", key="_bucket",
                               include_key=False),
                mkdir=True,
            )
        for b in range(shards):
            _phase(f"dedup bucket {b + 1}/{shards}")
            files = sorted(staged.glob(f"*/_bucket={b}/*.parquet"))
            if not files:
                continue
            # Each dedup part is written split by size bucket, and the merge
            # below reads those splits in bucket order, so the index comes out
            # clustered by bucket without ever sorting the whole pool. Readers
            # that filter on a bucket (`draw`, the extremes in `status`) then
            # skip the other row groups on their statistics.
            deduped = _dedup(pl.scan_parquet(files, hive_partitioning=False))
            _with_meta(deduped, meta_path, b, shards).sink_parquet(
                pl.PartitionBy(parts / f"part-{b:03d}", key="bucket",
                               include_key=True),
                mkdir=True,
            )
        buckets = sorted({int(d.name.split("=")[1]) for d in parts.glob("*/bucket=*")})
        if buckets:
            merged = pl.concat([
                pl.scan_parquet(sorted(parts.glob(f"*/bucket={n}/*.parquet")),
                                hive_partitioning=False)
                for n in buckets
            ])
        else:
            # Nothing survived the filters: write an index with the same
            # columns (metadata included) and no rows, so readers report an
            # empty pool rather than failing or seeing a different schema.
            empty = _projection(keep, lfs, tree_shards[0]).collect_schema()
            merged = _with_meta(_dedup(pl.LazyFrame(schema=empty)),
                                meta_path, 0, shards)
        _phase("merge")
        merged.sink_parquet(out)
        WATERMARK_PATH.write_text(high or "")
    finally:
        shutil.rmtree(staged, ignore_errors=True)
        shutil.rmtree(parts, ignore_errors=True)
        meta_path.unlink(missing_ok=True)

    # Repo-level roll-up, so `status` can report how many repos were scanned
    # without re-reading the whole log.
    _phase("repos roll-up")
    (
        _winning(keep)
        .select("repo_id", "full_name", "tier", "ext", "n_blobs", "n_kept",
                "truncated", "capped")
        .sink_parquet(REPOS_PATH)
    )
    _phase("done")
    return out


def files() -> pl.LazyFrame:
    if not INDEX_PATH.exists():
        raise SystemExit("no index yet: run `python -m csvscrape compact`")
    return pl.scan_parquet(INDEX_PATH)


def compact_strings(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Copy a filtered frame's strings out of the buffers they were read into.

    A string column Polars filters or joins is a set of views into the buffers
    of the morsels it came from, and the buffers live as long as any view
    does. A filter the Parquet reader cannot evaluate itself -- a hash
    residue, a join -- runs after the read, so a 1% selection still pins the
    other 99% of every string column it touched. (A predicate the reader does
    evaluate comes out compact already.) `strip_prefix("")` is the cheapest
    expression that rebuilds the strings; `str.slice(0)` and `cast(Binary)`
    are zero-copy and change nothing.
    """
    names = [n for n, t in frame.collect_schema().items() if t == pl.String]
    return frame.with_columns(pl.col(n).str.strip_prefix("") for n in names)


def pool() -> pl.DataFrame:
    """Per-bucket fill level: the dashboard the adaptive scan loop reads.

    The index holds one row per distinct content. `repos` is distinct
    *representative* repos: the collapse keeps one source per content, and
    `n_repos` on each row says how many others exist.
    """
    return (
        files()
        .filter(~pl.col("lfs_suspect"))
        .group_by("bucket")
        .agg(
            pl.col("occurrences").sum().alias("occurrences"),
            pl.len().alias("unique_files"),
            pl.col("repo_id").n_unique().alias("repos"),
            pl.col("is_lfs").sum().alias("lfs"),
            pl.col("size").min().alias("min_size"),
            pl.col("size").max().alias("max_size"),
        )
        .sort("bucket")
        # Streaming, explicitly: the default engine loads every projected
        # column of the index before aggregating, which is linear in the pool.
        .collect(engine="streaming")
    )


# -- reporting -------------------------------------------------------------

_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")
_BLOCKS = "▏▎▍▌▋▊▉█"


def human(n: float) -> str:
    """Binary units, because the buckets are powers of two."""
    for unit in _UNITS:
        if n < 1024 or unit == _UNITS[-1]:
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n} B"


def pow2(exponent: int) -> str:
    """Exact power-of-two label: '64 B', '1 KiB', '512 MiB'.

    Bucket bounds are exact by construction, so the decimal place `human` needs
    for arbitrary sizes only costs column width here.
    """
    unit = min(exponent // 10, len(_UNITS) - 1)
    return f"{2 ** (exponent - unit * 10)} {_UNITS[unit]}"


def calibrate(target: int = 250) -> str:
    """Project a small scan out to "how many repos do I need?".

    Yield is reported per discovery *size* tier, because the tier is what
    determines it: `lfs`-tier repos carry the top buckets almost exclusively,
    while `hi` repos fill the middle. The extension a repo was discovered
    through lives in `ext` and is deliberately not mixed in here -- it predicts
    dialect, not file size, and the answer this function exists to give is
    "which kind of repo do I need more of", which only the size tier can
    answer. Extrapolation is linear and therefore optimistic -- duplicate
    content rises as coverage grows, so treat the repo counts as a lower bound
    rather than an estimate.
    """
    if not REPOS_PATH.exists():
        return "no index yet: run `compact` first"

    repos = pl.read_parquet(REPOS_PATH)
    # A repo whose tree came back truncated, or whose blob list hit the cap,
    # contributed an unknown fraction of its files. Counting it as a full
    # observation understates yield per repo and makes the projection below
    # pessimistic in the one population that matters -- the large monorepos
    # the priority ordering exists to reach.
    partial = repos.filter(
        pl.col("truncated").fill_null(False) | pl.col("capped").fill_null(False)
    )["repo_id"]
    complete = repos.filter(~pl.col("repo_id").is_in(partial.implode()))
    scanned = complete.group_by("tier").agg(pl.col("repo_id").n_unique().alias("repos"))
    got = (
        files()
        .filter(~pl.col("lfs_suspect"))
        .filter(~pl.col("repo_id").is_in(partial.implode()))
        .group_by("tier", "bucket")
        .agg(pl.len().alias("found"))
        .collect(engine="streaming")
        .join(scanned, on="tier")
        .with_columns((pl.col("found") / pl.col("repos")).alias("rate"))
    )
    if got.is_empty():
        return "nothing scanned yet"

    best = (
        got.sort("rate", descending=True)
        .group_by("bucket")
        .first()
        .sort("bucket")
    )
    totals = {r["tier"]: r["repos"] for r in scanned.to_dicts()}

    out = [
        "calibration from " + ", ".join(f"{n:,} {t}-tier repos" for t, n in totals.items()),
        f"target {target} unique files per bucket; linear extrapolation (optimistic)",
        f"excluding {partial.len():,} repos with a truncated or capped tree "
        f"(their file lists are incomplete)",
        "",
        f"  {'bucket':>6}  {'range':>17}  {'have':>5}  {'best tier':>9}"
        f"  {'per 1k repos':>12}  {'repos needed':>12}",
    ]
    need_overall = 0
    for row in best.to_dicts():
        rate = row["rate"]
        needed = math.ceil(target / rate) if rate > 0 else None
        need_overall = max(need_overall, needed or 0)
        out.append(
            f"  {row['bucket']:>6}  {pow2(row['bucket']) + ' – ' + pow2(row['bucket'] + 1):>17}"
            f"  {row['found']:>5}  {row['tier']:>9}  {rate * 1000:>12,.0f}"
            f"  {needed and f'{needed:,}' or '-':>12}"
        )

    hours = need_overall / config.CORE_PER_HOUR
    out += [
        "",
        f"  binding bucket needs ~{need_overall:,} repos of its tier",
        f"  at {config.CORE_PER_HOUR:,} tree calls/hour that is ~{hours:,.0f}h "
        f"({hours / 24:,.1f} days) on one token",
    ]
    return "\n".join(out)


def _clip(text: str, width: int, keep: str = "head") -> str:
    """Trim to width. Repos keep their head, paths keep their filename end."""
    if len(text) <= width:
        return text
    return text[: width - 1] + "…" if keep == "head" else "…" + text[-(width - 1):]


def _bar(value: int, peak: int, width: int = 30) -> str:
    """Horizontal bar, linear in count.

    Linear rather than log on purpose: the goal is a log-uniform corpus, so a
    flat profile means the target is met and any dip is a bucket still to fill.
    A log count axis would flatten exactly the shortfall worth seeing.
    """
    if value <= 0 or peak <= 0:
        return ""
    eighths = round(value / peak * width * 8)
    full, rest = divmod(max(eighths, 1), 8)
    return _BLOCKS[-1] * full + (_BLOCKS[rest - 1] if rest else "")


def _extreme(descending: bool) -> dict | None:
    """The largest (or smallest) drawable file in the index.

    Two narrow passes rather than `sort().head(1)`: the sort needs every
    column of every row in memory to return one of them, where a scalar
    min/max streams over one column and the equality filter that follows is
    pushed into the Parquet row groups.
    """
    drawable = files().filter(~pl.col("lfs_suspect"))
    size = pl.col("size").max() if descending else pl.col("size").min()
    edge = drawable.select(size).collect(engine="streaming").item()
    if edge is None:
        return None
    rows = (
        drawable.filter(pl.col("size") == edge).head(1)
        .collect(engine="streaming")
    )
    return rows.to_dicts()[0] if rows.height else None


def _unfetched(pool: pl.LazyFrame) -> tuple[int, list[tuple[str, int]]]:
    """Files with no bytes: how many were never tried, and how the rest failed.

    The unprobed part of the pool is two populations that call for different
    things: files the probe has not reached yet (a run that has not finished)
    and files it reached and could not read (some of which never will be).

    `via` is what separates them: every record the probe writes carries it,
    failures included, so a null `via` is a file no record was ever written
    for. `http` cannot do this, since a dropped connection and a file never
    drawn both have no status. An index without `via` falls back to `http`.

    Counted over the *index*, so one row is one file rather than one record.
    """
    names = pool.collect_schema().names()
    if "fetched" not in names:
        return 0, []
    http = pl.col("http") if "http" in names else pl.lit(None, dtype=pl.Int64)
    tried = (pl.col("via").is_not_null() if "via" in names
             else http.is_not_null())
    counts = (
        pool.filter(~pl.col("fetched").fill_null(False))
        .group_by(http.alias("http"), tried.alias("tried"))
        .agg(pl.len().alias("n"))
        .collect(engine="streaming")
    )
    if counts.is_empty():
        return 0, []
    untried = int(counts.filter(~pl.col("tried"))["n"].sum())
    failed = [
        (f"http {int(r['http'])}" if r["http"] is not None else "no status",
         int(r["n"]))
        for r in counts.filter(pl.col("tried"))
        .sort("n", descending=True).to_dicts()
    ]
    return untried, failed


def _coverage_note(pool: pl.LazyFrame) -> str | None:
    """The line under the probed count: what is left, and whether it will move.

    A 404 is retired -- the path does not resolve at the pinned ref and no
    number of passes will change that -- so the files behind it are not a
    to-do and saying "N unprobed" of them invites a run that can only fail
    again. Everything else is still in the queue, and that number is what the
    next `probe` will draw. See `probe.retired_keys`.
    """
    untried, failed = _unfetched(pool)
    total = untried + sum(n for _, n in failed)
    if not total:
        return None
    parts = [f"{untried:,} remaining" if untried else "every file attempted"]
    if not failed:
        # Nothing has failed, so a count for the next run would only repeat
        # the one above it.
        return "  " + parts[0]
    shown, rest = failed[:6], sum(n for _, n in failed[6:])
    codes = "  ".join(f"{label}:{n:,}" for label, n in shown)
    if rest:
        codes += f"  other:{rest:,}"
    parts.append(f"{total - untried:,} failed ({codes})")
    outstanding = untried + sum(n for label, n in failed if label != "http 404")
    parts.append(f"{outstanding:,} for the next probe run" if outstanding
                 else "nothing left to probe")
    return "  " + " · ".join(parts)


def render() -> str:
    """Human-readable fill report: headline, histogram, extremes."""
    stats = pool()
    if stats.is_empty():
        return "index is empty -- run `discover`, `trees`, then `compact`"

    from . import sample

    cov = {int(r["bucket"]): r for r in sample.coverage().to_dicts()}
    repos = pl.read_parquet(REPOS_PATH) if REPOS_PATH.exists() else None
    drawable = ~pl.col("lfs_suspect")
    totals = (
        files()
        .select(
            drawable.sum().alias("n"),
            pl.col("size").filter(drawable).sum().alias("bytes"),
            pl.col("lfs_suspect").sum().alias("suspect"),
        )
        .collect(engine="streaming")
        .row(0)
    )

    out = []
    scanned = f"{repos.height:,} repos scanned · " if repos is not None else ""
    # Suspects are excluded from every figure, so a scan whose pointer fetches
    # failed would otherwise look identical to one that resolved them all.
    suspect = f" · {totals[2]:,} unresolved lfs excluded" if totals[2] else ""
    out.append(
        f"{scanned}{stats['occurrences'].sum():,} csv files · "
        f"{totals[0]:,} unique · {stats['lfs'].sum():,} lfs · "
        f"{human(totals[1])} deduplicated{suspect}"
    )
    # How much dialect metadata exists. This bounds the selectors that filter
    # on a verdict, not what `main` can draw. See `sample.coverage`.
    probed_total = sum(r["probed"] for r in cov.values())
    share = probed_total / max(totals[0], 1)
    out.append(f"{probed_total:,} probed ({share:.1%} of unique) · "
               + ("ready to sample" if share >= 1.0
                  else f"{totals[0] - probed_total:,} unprobed"))
    note = _coverage_note(files().filter(drawable))
    if note:
        out.append(note)
    out.append("")

    peak = int(stats["unique_files"].max())
    by_bucket = {int(r["bucket"]): r for r in stats.to_dicts()}
    lo, hi = min(by_bucket), max(by_bucket)

    out.append(f"  {'bucket':>6}  {'range':>17}  {'unique':>7} {'probed':>7} "
               f"{'%':>4} {'repos':>5} {'lfs':>4}")
    for bucket in range(lo, hi + 1):
        row = by_bucket.get(bucket)
        span = f"{pow2(bucket)} – {pow2(bucket + 1)}"
        if row is None:
            # Empty buckets are drawn, never skipped: a gap in coverage is the
            # signal that decides where the next scan budget goes.
            out.append(f"  {bucket:>6}  {span:>17}  {0:>7} {'-':>7} {'-':>4} "
                       f"{'-':>5} {'-':>4}  ·")
            continue
        got = cov.get(bucket, {}).get("probed", 0)
        pct = f"{100 * got / row['unique_files']:.0f}" if row["unique_files"] else "-"
        out.append(
            f"  {bucket:>6}  {span:>17}  {row['unique_files']:>7} {got:>7} {pct:>4} "
            f"{row['repos']:>5} {row['lfs']:>4}  {_bar(row['unique_files'], peak)}"
        )

    out.append("")
    for label, descending in (("smallest", False), ("largest", True)):
        row = _extreme(descending)
        if row:
            tag = " (lfs)" if row["is_lfs"] else ""
            repo = _clip(row["full_name"], 24)
            path = _clip(row["path"], 24, keep="tail")
            out.append(f"  {label:<9} {human(row['size']):>10}  {repo:<24}  {path}{tag}")
    return "\n".join(out)


# -- kaggle: a parallel index over the same schema --------------------------
#
# Same columns as `files.parquet`, including every probe verdict, because that
# is what lets one set of selectors and one `draw` serve both pools. What is
# *not* shared is the pool itself: a dataset is built from one source or the
# other, never a union, so the two indexes are never concatenated.

KAGGLE_INDEX = config.DATA_ROOT / "index" / "kaggle_files.parquet"
KAGGLE_DATASETS = config.DATA_ROOT / "index" / "kaggle_datasets.parquet"


def _kaggle_bundles() -> pl.LazyFrame:
    from . import kaggle

    parts = log.sealed(kaggle.BUNDLES_LOG)
    if not parts:
        raise SystemExit(
            "no kaggle bundle log: run `python -m csvscrape kaggle-bundle`")
    frame = pl.scan_ndjson(parts)
    # A revisit that grew the probe prefix writes an `extended` record with
    # only the new `n_probed` for that archive version, so the prefix is the
    # furthest any record reached. See `kaggle.revisits`.
    extended = (
        frame.filter(pl.col("state") == "extended")
        .group_by("ref", "version")
        .agg(pl.col("n_probed").max().alias("extended"))
    )
    # The newest record decides, then only `ok` is kept: a dataset whose
    # archive a later record found changed, or gone, has no readable
    # coordinates left to index.
    return (
        log.newest_per(frame.filter(pl.col("state") != "extended"), ["ref"])
        .filter(pl.col("state") == "ok")
        .join(extended, on=["ref", "version"], how="left", nulls_equal=True)
        .with_columns(pl.max_horizontal("n_probed", "extended").alias("n_probed"))
        .drop("extended")
    )


def build_kaggle(out: Path | None = None) -> Path:
    """Compact the bundle logs into an index with the same schema as `build`.

    Content identity is `(crc32, uncompressed size)` from each bundle's central
    directory, formatted by `kaggle.content_key`. Identical bytes always agree
    on it, so it never misses a duplicate. It can falsely merge two distinct
    files only within a group sharing an exact byte length: the expected count
    is `sum(C(n_i, 2)) / 2**32` over same-size groups, which is negligible.

    `occurrences` and `n_repos` carry the republication count, which on Kaggle
    is the headline fact about the small buckets: teaching datasets are
    republished thousands of times, and collapsing them here is what stops a
    log-uniform draw's bottom octaves being a dozen copies of iris.
    """
    from . import kaggle, probe

    # Resolved here rather than as a default argument, which would bind the
    # module global at import and miss a data root repointed afterwards.
    out = KAGGLE_INDEX if out is None else out
    out.parent.mkdir(parents=True, exist_ok=True)
    KAGGLE_DATASETS.parent.mkdir(parents=True, exist_ok=True)
    columns = ["names", "keys", "sizes", "crcs", "csizes", "methods", "offsets",
               "position"]
    exploded = (
        _kaggle_bundles()
        # Each member's place in the bundle's hash order, so it can be told
        # whether it fell inside the probe prefix (see `in_probe_cap`).
        .with_columns(pl.int_ranges(pl.col("names").list.len()).alias("position"))
        # `empty_as_null` pinned because the default changes in Polars 2.0. A
        # dataset with no kept files has empty arrays, which become one
        # all-null row that the filter below drops.
        .explode(columns, empty_as_null=True)
        .rename({"names": "path", "keys": "content_key", "sizes": "size",
                 "crcs": "crc", "csizes": "csize", "methods": "method",
                 "offsets": "header_offset"})
        .filter(pl.col("path").is_not_null()
                & (pl.col("size") >= config.INDEX_MIN_BYTES))
        .with_columns(
            pl.lit("kaggle").alias("source"),
            pl.col("ref").alias("full_name"),
            # The GitHub index keys repos by a numeric id; here the ref is the
            # identity, and `download`'s diversity caps read this column.
            pl.col("ref").alias("repo_id"),
            pl.col("ref").str.split("/").list.first().alias("owner"),
            pl.col("size").alias("raw_size"),
            # Same guard as `_projection`: log2(0) is -inf, and the cast to
            # Int32 raises and fails the whole sink.
            pl.when(pl.col("size") < 1)
            .then(pl.lit(0))
            .otherwise(pl.col("size").cast(pl.Int64).log(2).floor())
            .cast(pl.Int32)
            .alias("bucket"),
            pl.lit(False).alias("is_lfs"),
            pl.lit(False).alias("lfs_suspect"),
            pl.lit(None, dtype=pl.String).alias("blob_sha"),
            pl.lit("kaggle").alias("tier"),
            # The version this row was read at, standing where the GitHub index
            # records a commit sha. `download` pins on it and rejects a drift.
            pl.col("version").cast(pl.Int64).alias("version"),
            pl.col("version").cast(pl.String).alias("ref_pin"),
            # True when the archive held more kept files than the *recording*
            # cap admitted, so the row comes from a sample of its dataset.
            # (Unprobed is a different thing, read off `fetched`.)
            pl.col("capped").fill_null(False).alias("truncated"),
            # Inside the first `n_probed` members, so the bundle pass queued a
            # head for it. The rest are past the probe cap and nothing
            # schedules them, so `render_kaggle` counts them apart.
            (pl.col("position") < pl.col("n_probed")).alias("in_probe_cap"),
            pl.col("path").str.extract(r"\.([^./]+)$").str.to_lowercase().alias("ext"),
        )
    )
    collapsed = (
        exploded
        .group_by("content_key")
        .agg(
            pl.len().alias("occurrences"),
            pl.col("full_name").n_unique().alias("n_repos"),
            # Per content, not per row: one copy inside some dataset's probe
            # prefix is enough for the head to have been queued, and the head
            # joins on content_key -- so it describes every copy.
            pl.col("in_probe_cap").any(),
            # One representative row per distinct content, chosen
            # deterministically so rebuilds name the same source.
            pl.all().exclude("in_probe_cap").sort_by("full_name").first(),
        )
        .with_columns(pl.col("ref_pin").alias("ref"))
        .select(
            "source", "owner", "full_name", "repo_id", "path", "content_key",
            "size", "raw_size", "bucket", "is_lfs", "lfs_suspect", "blob_sha",
            "tier", "ext", "truncated", "in_probe_cap", "ref", "occurrences",
            "n_repos",
            # Kaggle-only, and the reason a member can be fetched at all: where
            # its bytes sit inside the bundle, and how they are framed.
            "crc", "csize", "method", "header_offset", "version", "license",
        )
    )
    rows = probe.metadata_rows(kaggle.HEADS_LOG)
    if rows is not None:
        collapsed = collapsed.join(probe.dedupe(rows), on="content_key",
                                   how="left", maintain_order="left")
    collapsed.sink_parquet(out)

    # Dataset-level roll-up, so `kaggle-status` can report coverage without
    # re-reading the bundle log -- the same role `REPOS_PATH` plays for GitHub.
    # One row per dataset the pass has read, indexed or not, so the ones it
    # lost are counted rather than simply absent.
    states = kaggle.outcomes()
    (
        pl.LazyFrame({"ref": list(states), "outcome": list(states.values())},
                     schema={"ref": pl.String, "outcome": pl.String})
        .join(_kaggle_bundles()
              .select("ref", "owner", "license", "total_bytes", "zip_bytes",
                      "n_entries", "n_kept", "n_recorded", "n_probed",
                      "capped", "version"),
              on="ref", how="left")
        .sink_parquet(KAGGLE_DATASETS)
    )
    return out


def kaggle_files() -> pl.LazyFrame:
    if not KAGGLE_INDEX.exists():
        raise SystemExit("no kaggle index: run `python -m csvscrape kaggle-compact`")
    return pl.scan_parquet(KAGGLE_INDEX)


# Where each source's pool lives, so everything downstream takes a name rather
# than a path and the two cannot be mixed up at a call site.
SOURCES = {"github": (INDEX_PATH, files), "kaggle": (KAGGLE_INDEX, kaggle_files)}


def pool_for(source: str = "github") -> pl.LazyFrame:
    """The index for one source. The only place a source name becomes a file."""
    if source not in SOURCES:
        raise SystemExit(f"unknown source {source!r}; "
                         f"choose from {', '.join(sorted(SOURCES))}")
    return SOURCES[source][1]()


def _kaggle_coverage(pool: pl.LazyFrame) -> list[str]:
    """The lines under the probed count: why each unprobed file is unprobed.

    Unlike GitHub, an untried file is outstanding work only inside the probe
    prefix, where `kaggle.revisits` retries failed or dropped heads. A file
    past the cap waits for `--per-dataset` to be raised. Content with any copy
    inside a prefix counts as queued.
    """
    names = pool.collect_schema().names()
    if "fetched" not in names:
        return []
    inside = (pl.col("in_probe_cap").fill_null(True) if "in_probe_cap" in names
              else pl.lit(True))
    past = int(pool.filter(~pl.col("fetched").fill_null(False) & ~inside)
               .select(pl.len()).collect(engine="streaming").item())
    untried, failed = _unfetched(pool.filter(inside))
    out = []
    if past:
        out.append(f"  {past:,} past the probe cap -- raise --per-dataset "
                   f"to reach them")
    if failed:
        codes = "  ".join(f"{label}:{n:,}" for label, n in failed[:6])
        rest = sum(n for _, n in failed[6:])
        if rest:
            codes += f"  other:{rest:,}"
        out.append(f"  {sum(n for _, n in failed):,} inside it failed "
                   f"({codes}) -- the next kaggle-bundle retries them")
    if untried:
        out.append(f"  {untried:,} inside it never attempted -- dropped by a "
                   f"stopped run; the next kaggle-bundle retries them")
    return out


def render_kaggle() -> str:
    """The same fill report as `status`, over the Kaggle pool."""
    frame = kaggle_files()
    names = frame.collect_schema().names()
    probed = (pl.col("fetched").fill_null(False)
              if "fetched" in names else pl.lit(False))
    stats = (
        frame.group_by("bucket")
        .agg(pl.col("occurrences").sum().alias("occurrences"),
             pl.len().alias("unique_files"),
             pl.col("full_name").n_unique().alias("repos"),
             probed.sum().alias("probed"))
        .sort("bucket").collect(engine="streaming")
    )
    if stats.is_empty():
        return "kaggle index is empty -- run `kaggle-discover`, `kaggle-bundle`, then `kaggle-compact`"
    totals = (frame.select(pl.len().alias("n"),
                           pl.col("size").sum().alias("bytes"),
                           pl.col("occurrences").sum().alias("occ"),
                           pl.col("full_name").n_unique().alias("sets"))
              .collect(engine="streaming").row(0))

    out = []
    if KAGGLE_DATASETS.exists():
        # Per listed file, before dedup: which cap a file stopped at is a fact
        # about its dataset, not its content. Files past the recording cap are
        # not in the index at all; files past the probe cap are, unprobed.
        sets = pl.read_parquet(KAGGLE_DATASETS)
        capped = int(sets["capped"].fill_null(False).sum())
        listed = int(sets["n_kept"].sum())
        recorded = int(sets["n_recorded"].sum())
        queued = int(sets["n_probed"].sum())
        counts = dict(sets.group_by("outcome").len().iter_rows())
        indexed = counts.get("indexed", 0) + counts.get("reread", 0)
        out.append(f"{sets.height:,} datasets read · {indexed:,} indexed · "
                   f"{listed:,} csv files listed")
        again = [(counts.get(k, 0), label) for k, label in (
            ("reread", "read unpinned from an archive not the size listed"),
            ("drifted", "whose archive changed since"),
            ("retry", "unanswered"))]
        if any(n for n, _ in again):
            out.append("  " + " · ".join(f"{n:,} {label}" for n, label in again if n)
                       + " -- the next kaggle-bundle reads them again, pinned")
        lost = [(counts.get(k, 0), label) for k, label in (
            ("not_found", "not found (404: the version is no longer kept)"),
            ("private", "private or deleted (403)"),
            ("no_bundle", "not an archive"))]
        if any(n for n, _ in lost):
            out.append("  lost: " + " · ".join(f"{n:,} {label}"
                                                for n, label in lost if n))
        out.append(f"  {recorded:,} recorded · {listed - recorded:,} past the "
                   f"recording cap, in {capped:,} datasets")
        out.append(f"  {queued:,} queued for a head · {recorded - queued:,} "
                   f"recorded past the probe cap")
    out.append(
        f"{totals[2]:,} csv files · {totals[0]:,} unique content · "
        f"{totals[3]:,} datasets · {human(totals[1])} deduplicated")
    # Republication is the Kaggle pool's defining property, so the dedup ratio
    # is reported rather than left to be inferred from two numbers.
    if totals[0]:
        out.append(f"{totals[2] / totals[0]:.2f} copies per distinct file")
    probed_total = int(stats["probed"].sum())
    share = probed_total / max(totals[0], 1)
    out.append(f"{probed_total:,} probed ({share:.1%} of unique) · "
               + ("ready to sample" if share >= 1.0
                  else f"{totals[0] - probed_total:,} unprobed"))
    out.extend(_kaggle_coverage(frame))
    out.append("")

    peak = int(stats["unique_files"].max())
    by_bucket = {int(r["bucket"]): r for r in stats.to_dicts()}
    out.append(f"  {'bucket':>6}  {'range':>17}  {'unique':>7} {'probed':>7} "
               f"{'%':>4} {'sets':>5}")
    for bucket in range(min(by_bucket), max(by_bucket) + 1):
        row = by_bucket.get(bucket)
        span = f"{pow2(bucket)} – {pow2(bucket + 1)}"
        if row is None:
            out.append(f"  {bucket:>6}  {span:>17}  {0:>7} {'-':>7} {'-':>4} "
                       f"{'-':>5}  ·")
            continue
        pct = (f"{100 * row['probed'] / row['unique_files']:.0f}"
               if row["unique_files"] else "-")
        out.append(f"  {bucket:>6}  {span:>17}  {row['unique_files']:>7} "
                   f"{row['probed']:>7} {pct:>4} {row['repos']:>5}  "
                   f"{_bar(row['unique_files'], peak)}")
    return "\n".join(out)
