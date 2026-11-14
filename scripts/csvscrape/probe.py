"""Pool-wide metadata pass: read each file's head and describe it.

A head costs the same whatever the file's size, so describing a 1.6 GB LFS
object costs the same as a 2 KB one. Files smaller than the head
(config.HEAD_BYTES) are read in full, and each record carries a `complete`
flag saying whether it was.

Three stages, parallelised separately because each is bounded by something
different and none of them may wait on another:

  * the batched read (`--workers`). Nearly every file's head comes from a
    GraphQL query aliasing many blobs across a few repositories. GitHub charges
    one rate-limit point per query whatever it carries, so the budget -- 5,000
    points an hour -- paces this to 1.32 requests a second and the throughput
    of the whole pass is that times `config.GRAPHQL_BATCH_BYTES`.
  * the range request (`--raw-workers`), for LFS objects, blobs GraphQL
    declines, and files past `GRAPHQL_BLOB_MAX`. Metered per IP at ~113,000
    requests an hour behind a burst of roughly 40,000, which makes it the
    scarcest budget in the pipeline and the reason so little is routed to it.
  * sniffing (`--sniff-procs`), which is CPU-bound and mostly pure Python, so
    it goes to a process pool. See config.SNIFF_PROCESSES.

A bounded queue between the fetch stages and the sniff stage
(`config.SNIFF_QUEUE`) is the only backpressure: when the sniffers fall behind
the fetchers block on `put` rather than reading the draw into memory.

Caching the fetched bytes is optional (`--cache`) and lives under its own root
(`--cache-dir`, or CSVSCRAPE_CACHE) so it can sit on separate storage. It is
worth turning on for a run you might want to re-analyse: recomputing metadata
from cached bytes is a local pass, while recomputing without them is a full
network sweep. It is worth leaving off when the pool is large enough that the
bytes dominate everything else on disk.
"""
from __future__ import annotations

import collections
import multiprocessing
import pathlib
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import polars as pl
import zstandard

from . import config, index, log, sniffers, validate
from .blobstore import BlobStore
from .client import GitHub
from .util import interleave, run_workers

HEADS_LOG = "heads"

# How long a stage waits on the queue between it and the next before looking
# up. Short enough that a Ctrl-C is noticed promptly, and it doubles as the
# unit for `sniff_blocked_s` / `fetch_blocked_s`: every expiry is one thread
# idle for this long, waiting on the other side of the pipeline.
_STALL = 0.2


def _scan_heads(parts: list[str]) -> pl.LazyFrame:
    """Scan heads shards with the record's types named rather than guessed.

    `scan_ndjson` infers from the first rows and gives any field that is null
    across all of them the Null dtype. Many Shape fields (`comment`, `escape`,
    `magic_kind`) and `http` are null on most records, so the first record
    that carries a value would fail the scan with a SchemaError. Naming the
    types removes the dependency on what happens to be at the front of the log.

    Only columns some shard actually carries are named, so the older-log
    fallbacks below still fire; the rest of the schema is `_log_schema`'s.
    """
    from .features import Shape

    declared: dict[str, Any] = {
        "content_key": pl.String, "rec_id": pl.String, "ts": pl.Int64,
        "agreement": pl.Int64, "bucket": pl.Int64, "size": pl.Int64,
        "head_len": pl.Int64, "message": pl.String,
        "head_blob": pl.String,
        # Only present in older logs.
        "tail_len": pl.Int64, "tail_blob": pl.String,
        # From the dataclass that writes it, so the two cannot drift apart.
        "custom": pl.Struct({name: _shape_dtype(name)
                             for name in Shape.__annotations__}),
        # Only the field `metadata` reads is named. The other sniffers' columns
        # are never selected, so projection pushdown never materialises them.
        "charset_normalizer": pl.Struct({"encoding": pl.String}),
        "chardet": pl.Struct({"encoding": pl.String}),
    } | PROBE_FIELDS
    schema = _log_schema(parts)
    schema |= {k: v for k, v in declared.items() if k in schema}
    return pl.scan_ndjson(parts, schema=schema)


def _merge_dtype(have: Any, found: Any) -> Any:
    """The wider of two readings of one column, across shards.

    Null loses to anything, because Null here means "every record in that
    shard left the field out" rather than a type. Two structs merge field by
    field for the same reason: a shard whose CSVs all had a comment character
    types `custom.comment` as String and one whose CSVs had none types it
    Null, and taking either shard's struct whole would drop the other's
    fields.
    """
    if have is None or have == pl.Null:
        return found
    if isinstance(have, pl.Struct) and isinstance(found, pl.Struct):
        fields = {f.name: f.dtype for f in have.fields}
        for field in found.fields:
            fields[field.name] = _merge_dtype(fields.get(field.name), field.dtype)
        return pl.Struct(fields)
    return have


def _log_schema(parts: list[str]) -> dict[str, Any]:
    """The column types over every shard, not just the first one.

    `scan_ndjson` given a list infers the schema from the first file alone,
    and a column absent from *that* file is silently dropped from every file.
    A shard whose records are all failures has no `custom` or `agreement`, so
    reading it first would null every later shard's verdicts. Reading each
    shard's schema costs milliseconds.
    """
    merged: dict[str, Any] = {}
    for part in parts:
        for name, dtype in pl.scan_ndjson(part).collect_schema().items():
            merged[name] = _merge_dtype(merged.get(name), dtype)
    return merged


def _fresh_log() -> pl.LazyFrame | None:
    """Heads records not yet folded into the index.

    `compact` records the highest rec_id it consumed; everything at or below
    that lives in the index as columns, so only what came after needs parsing.
    rec_id is `{shard:05d}:{seq:08d}`, which sorts lexicographically and names
    its own shard -- so whole files below the watermark are skipped without
    being opened.

    Reading the entire log on every invocation would cost gigabytes at full
    scale.
    """
    parts = [pathlib.Path(p) for p in log.sealed(HEADS_LOG)]
    if not parts:
        return None
    mark = index.heads_watermark()
    if mark:
        shard = mark.split(":", 1)[0]
        parts = [p for p in parts if p.name[len("part-"):len("part-") + 5] >= shard]
        if not parts:
            return None
    frame = _scan_heads([str(p) for p in parts])
    if mark and "rec_id" in frame.collect_schema().names():
        frame = frame.filter(pl.col("rec_id") > mark)
    return frame


def _indexed_flags() -> pl.LazyFrame | None:
    """`fetched`/`http` for content already folded into the index."""
    if not index.INDEX_PATH.exists():
        return None
    names = pl.scan_parquet(index.INDEX_PATH).collect_schema().names()
    if "fetched" not in names:
        return None
    return pl.scan_parquet(index.INDEX_PATH).select(
        pl.col("content_key"),
        pl.col("fetched"),
        *( [pl.col("http")] if "http" in names else
           [pl.lit(None, dtype=pl.Int64).alias("http")] ),
    )


def _keys_where(condition) -> pl.LazyFrame:
    """Content keys matching a predicate over (fetched, http).

    Drawn from the index for everything compacted and from the log tail for the
    rest, so the answer is current without re-reading what the index already
    holds.
    """
    empty = pl.LazyFrame({"content_key": []}, schema={"content_key": pl.String})
    sides = []
    for frame in (_indexed_flags(), _fresh_log()):
        if frame is None:
            continue
        names = frame.collect_schema().names()
        if "fetched" not in names:
            continue
        http = pl.col("http") if "http" in names else pl.lit(None, dtype=pl.Int64)
        sides.append(
            frame.filter(condition(pl.col("fetched").fill_null(False), http))
            .select(pl.col("content_key").cast(pl.String))
        )
    if not sides:
        return empty
    return pl.concat(sides, how="vertical_relaxed").unique()


def settled(fetched: pl.Expr, http: pl.Expr) -> pl.Expr:
    """Whether (fetched, http) say a key needs no further fetch.

    The one expression behind both readers of those two columns -- the
    anti-join set in `retired_keys` and the filter `candidates` applies to the
    index directly -- so the two cannot drift apart. `settles` is the scalar
    twin for a record in hand.

    Null-safe on purpose: `http` is null for every row the probe has not
    reached, and in Polars `False | null` is null. A caller negating this to
    keep the *unsettled* rows would otherwise drop every never-probed file.
    """
    return (fetched.fill_null(False) | (http == 404)).fill_null(False)


def retired_keys() -> pl.LazyFrame:
    """Content keys that should not be probed again, to anti-join against.

    Retired is not the same as probed. A row counts as settled when either:

      * it was fetched -- there are bytes and verdicts, nothing more to learn;
      * or the fetch returned 404 -- the path does not resolve at the pinned
        ref, and no number of retries will change that.

    Everything else was transient (429, 503, a dropped connection) and stays in
    the queue; settling it would lose the file for good to one network blip.
    """
    return _keys_where(settled)


def settles(record: dict[str, Any]) -> bool:
    """Whether this record retires its key, so the next pass will not redraw it.

    The scalar twin of `settled`, and it must stay equal to it. The pass loop
    in `cli` stops when a pass settles nothing, so a version of this that
    disagreed with the anti-join would either stop a run with work left or
    spin on files it thought it had finished.
    """
    return bool(record.get("fetched")) or record.get("http") == 404


def probed_keys() -> pl.LazyFrame:
    """Content keys that actually carry verdicts.

    Not the same set as `retired_keys`: a 404 is retired but never probed. This
    is what "probed" means for the sampling invariant and for coverage --
    metadata exists for this content.
    """
    return _keys_where(lambda fetched, http: fetched)


def candidates(limit: int | None = None, order: str = "uniform",
               oversample: int = 4, seed: int = 0,
               after: tuple[int, int] | None = None) -> list[dict[str, Any]]:
    """Pool files without metadata yet, sampled uniformly by default.

    `after` resumes an `--order repo` draw at the point the previous pass of
    this run stopped. Without it every pass anti-joins against `retired_keys`,
    which parses every heads record written since the last `compact`, so the
    passes of a long run would keep re-parsing what earlier passes wrote. The
    order is a total one -- `(hash(full_name), hash(content_key))` under a
    fixed seed -- so "what the last pass did not reach" is a comparison
    against its final row. The rows carry those hashes back as `_repo_rank`
    and `_key_rank` for the caller to hand to the next pass.

    The cursor only narrows the draw; it does not replace the already-probed
    filter, which runs on every pass off the index's own columns (a predicate,
    not a join). The cursor knows what this run drew, not what previous runs
    probed. A file a previous run probed that `compact` has not yet folded in
    can still be redrawn by a later pass; running `compact` before a re-run
    avoids that.

    The cursor is a within-run device only. A new process has none and pays
    the anti-join once: the log is the durable record of what was probed.

    `order` picks the sampling frame, and they answer different questions:

      * "uniform" -- every unprobed file equally likely, so the draw is a
        random sample *of the pool*. Verdicts aggregated over it describe real
        CSVs on GitHub, which is what the correctness/validation sets are for.
      * "spread" -- equal counts per size octave, so every bucket fills at the
        same rate however few files it holds. That is what a log-uniform corpus
        needs, since bucket 31 holds 18 files of 65.5M and a uniform draw of
        10,000 would take 0.003 of them.
      * "repo" -- a random subset of repositories, taken whole. Not a sample of
        the pool but a cluster sample of it, and the difference is not
        cosmetic: files in one repo share a publisher and an export tool, so
        they share a delimiter and an encoding, and aggregating verdicts over
        this draw describes the repos it happened to pick. Use it for coverage,
        where the order carries no inference and only throughput matters, and
        "uniform" for anything whose numbers get reported.

        What it buys is fetch throughput: `GitHub.blobs` is charged per
        repository resolution, so a clustered draw fills a GraphQL query from
        far fewer repos than a uniform one.

    Uniform is the default because what needs probe metadata is the
    validation set, which wants a sample of the pool; the corpora are drawn
    from the index directly and described after download. A later `--order
    spread` pass draws only what earlier passes did not, and the bucket
    weights of the mixture are recoverable from the index.

    The order within a draw is a seeded hash of `content_key`: deterministic,
    unrelated to scan order (which is discovery-tier order), and it advances
    through the pool as rows are settled. The already-probed filter is applied
    before taking the head of that order. Wide blocking operations -- global
    `unique`, window functions, sorts over the whole pool with its strings
    attached -- are avoided because they materialise the whole index.
    """
    pool = index.files().filter(~pl.col("lfs_suspect"))
    # The settled rows leave on every pass, cursor or no cursor. For anything
    # `compact` has folded in this is a column filter, not a join.
    names = pool.collect_schema().names()
    if "fetched" in names:
        http = (pl.col("http") if "http" in names
                else pl.lit(None, dtype=pl.Int64))
        pool = pool.filter(~settled(pl.col("fetched"), http))
    outstanding = (
        pool
        # `blob_sha` is what the batched GraphQL read addresses a blob by, and
        # `ref` pins a range read to the commit the scan saw.
        .select("content_key", "full_name", "path", "size", "bucket", "is_lfs",
                "blob_sha", "ref")
    )
    if after is None:
        outstanding = outstanding.join(retired_keys(), on="content_key",
                                       how="anti")
    else:
        # On top of the index filter above, never instead of it: this excludes
        # what *this run* has already drawn, not what earlier runs settled.
        repo_rank, key_rank = after
        rh = pl.col("full_name").hash(seed=seed)
        kh = pl.col("content_key").hash(seed=seed)
        outstanding = outstanding.filter(
            (rh > repo_rank) | ((rh == repo_rank) & (kh > key_rank)))
    if order == "repo":
        # Ordered by repo hash first and content key hash second, so the rows
        # arrive grouped and `_repo_chunks` sees whole repositories.
        #
        # No residue thinning here (unlike the branch below): the cursor
        # advances past every repo below the last rank drawn, so repos thinned
        # away would be lost for good.
        #
        # Instead the draw takes two passes, neither sorting anything wide. The
        # first ranks *repositories* only -- one u64 per row -- and reads off
        # the repo rank of the `limit`th row. The second keeps rows at or below
        # that rank (the draw plus at most the tail of the boundary repo) and
        # sorts those. This is exact: the `limit` smallest (repo, key) pairs
        # all have a repo rank no greater than the `limit`th row's.
        #
        # Cutting at a fixed fraction of the hash range instead would be wrong:
        # the order is consumed front-first, so once the front is probed a
        # fixed cut returns nothing, which `cli` reads as an exhausted pool.
        repo_rank = pl.col("full_name").hash(seed=seed)
        key_rank = pl.col("content_key").hash(seed=seed)
        frame = outstanding
        if limit:
            edge = (outstanding.select(repo_rank.alias("_repo_rank"))
                    .sort("_repo_rank").head(limit).tail(1)
                    .collect(engine="streaming"))
            if edge.height:
                frame = frame.filter(repo_rank <= edge.row(0)[0])
        # The ranks ride back on every row: the last row's pair is the cursor
        # the next pass resumes from, and Polars' hash is not reproducible in
        # Python.
        frame = (
            index.compact_strings(frame)
            .with_columns(repo_rank.alias("_repo_rank"),
                          key_rank.alias("_key_rank"))
            .sort("_repo_rank", "_key_rank")
        )
        if limit:
            frame = frame.head(limit)
    elif limit:
        # Sorting the whole pool to take a few thousand rows off the front is a
        # blocking operation over every unprobed file. Instead, thin the pool by
        # a hash residue first so only a few multiples of what is needed
        # survive, then sort that.
        #
        # The residue is a deterministic function of the key, so the sample is
        # still reproducible and still unrelated to scan order. It shifts as
        # the pool shrinks, which is what lets successive runs advance rather
        # than redraw: rows already probed are gone from the anti-join above,
        # the counts fall, and the next modulus admits a different slice.
        per_bucket = max(1, limit // 8) * oversample
        counts = (
            outstanding.group_by("bucket").agg(pl.len().alias("n"))
            .collect(engine="streaming")
        )
        if order == "spread":
            # Thinned against the *largest* bucket, because the draw takes the
            # same count from each and the biggest one is what has to survive
            # the residue in that quantity.
            biggest = int(counts["n"].max()) if counts.height else 0
            modulus = max(1, biggest // max(per_bucket * 4, 1))
        else:
            # Thinned against the whole pool, because a uniform draw takes
            # rows in proportion to what is there and any per-bucket figure
            # would bound the wrong quantity.
            total = int(counts["n"].sum()) if counts.height else 0
            modulus = max(1, total // max(limit * oversample, 1))
        if modulus > 1:
            outstanding = index.compact_strings(outstanding.filter(
                pl.col("content_key").hash(seed=seed) % modulus == 0
            ))
        # Both draws are a hash order over the surviving keys; they differ only
        # in what is ranked. The sort runs over the key and bucket alone, since
        # the survivors can still number in the millions; the full rows are
        # joined back for the `limit` winners only.
        keyed = (outstanding.select("content_key", "bucket")
                 .sort(pl.col("content_key").hash(seed=seed)))
        if order == "spread":
            # A round-robin over buckets: the first row of every bucket, then
            # the second of every bucket, and so on until `limit` -- expressed
            # as "rank within bucket, then sort by (rank, bucket)" so only
            # `limit` rows ever leave Polars.
            winners = (
                keyed
                .with_columns(pl.int_range(pl.len()).over("bucket").alias("_rank"))
                .filter(pl.col("_rank") < per_bucket)
                .sort("_rank", "bucket")
                .head(limit)
            )
        else:
            # Rank over the whole frame, not within a bucket: a per-bucket cut
            # would give equal counts per octave, i.e. a spread draw. A hash
            # order is already a uniform permutation, so the first `limit` of
            # it is the sample.
            winners = (
                keyed
                .with_columns(pl.int_range(pl.len()).alias("_rank"))
                .head(limit)
            )
        frame = (
            index.compact_strings(outstanding.join(
                winners.select("content_key", "_rank")
                .collect(engine="streaming").lazy(),
                on="content_key", how="inner",
            ))
            .sort("_rank", "bucket")
            .drop("_rank")
        )
    else:
        frame = outstanding.sort(pl.col("content_key").hash(seed=seed))
    # Streaming, explicitly: the anti-join and the residue filter then run
    # over the index a morsel at a time instead of after loading it whole.
    rows = frame.collect(engine="streaming").to_dicts()

    # The index is deduped by content already; this only guards against the
    # draw holding two rows for one key.
    seen: set[str] = set()
    fresh: list[dict[str, Any]] = []
    for row in rows:
        if row["content_key"] in seen:
            continue
        seen.add(row["content_key"])
        fresh.append(row)

    if order == "spread" and not limit:
        by_bucket: dict[int, list[dict]] = {}
        for row in fresh:
            by_bucket.setdefault(row["bucket"], []).append(row)
        fresh = list(interleave(dict(sorted(by_bucket.items()))))
    return fresh[:limit] if limit else fresh


def _sniff_pool(procs: int, which: tuple[str, ...]) -> ProcessPoolExecutor | None:
    """A pool for the sniffers, or None to sniff in the calling thread.

    Never the `fork` start method: the parent holds fetch threads and an open
    requests session, and forking gives the child a copy of any lock those
    threads happened to hold, with no thread left to release it. `forkserver`
    forks from a clean process; `spawn` is the fallback where forkserver is
    not available. Both re-import `csvscrape.sniffers` in the child, which
    never imports Polars.
    """
    if procs < 1:
        return None
    failures: list[str] = []
    for method in ("forkserver", "spawn"):
        try:
            ctx = multiprocessing.get_context(method)
        except ValueError as exc:
            failures.append(f"{method}: {exc}")
            continue
        pool = ProcessPoolExecutor(procs, mp_context=ctx,
                                   initializer=sniffers.worker_init,
                                   initargs=(which,))
        try:
            # Start a worker now and wait for it. Whether a start method works
            # is not something get_context can answer -- forkserver wants an
            # AF_UNIX socket and some sandboxes refuse it -- and the failure
            # would otherwise surface on the first real file.
            pool.submit(sniffers.agreement, {}).result()
            return pool
        except Exception as exc:
            failures.append(f"{method}: {type(exc).__name__}: {exc}")
            pool.shutdown(wait=False)
    raise RuntimeError(
        "could not start a sniffer process pool (" + "; ".join(failures) +
        "). Pass --sniff-procs 0 to sniff in the fetch threads instead."
    )


def _sniff(pool: ProcessPoolExecutor | None, head: bytes, size: int,
           which: tuple[str, ...]) -> dict[str, Any]:
    """Sniff one file, in a worker process when there is a pool.

    One file per submit, deliberately. The IPC round trip is negligible next
    to sniffing, and per-file sniff time varies widely with content, so
    chunking would cost load balancing (a chunk of slow files strands a
    worker while the others idle) without saving anything.
    """
    if pool is None:
        return sniffers.run(head, size=size, which=which)
    return pool.submit(sniffers.run, head, size, which).result()


def _via_graphql(row: dict[str, Any]) -> bool:
    """Whether this file's head comes from a batched GraphQL read.

    Routed out to a range request instead:

      * LFS. The git blob is the pointer, which passes every check `blobs`
        makes while being entirely the wrong content. Those take the media
        path.
      * Above `PROBE_GRAPHQL_MAX_BYTES` (a tunable policy) or
        `GRAPHQL_BLOB_MAX` (where GraphQL stops serving the blob at all).
    """
    return (not row["is_lfs"]
            and row["size"] <= config.PROBE_GRAPHQL_MAX_BYTES
            and row["size"] <= config.GRAPHQL_BLOB_MAX)


def _graphql_cost(row: dict[str, Any]) -> int:
    """Bytes this row will add to a GraphQL response, or zero if it takes raw."""
    return min(row["size"], config.GRAPHQL_TEXT_CAP) if _via_graphql(row) else 0


def _repo_chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows by repository, splitting any repo too large for one batch.

    The queue's unit of work, so that a repository stays whole no matter how
    many workers are pulling from it. `GitHub.blobs` charges by repository
    resolution, so a batch's cost is the number of chunks in it rather than the
    number of rows -- which only holds if a chunk is never split across batches
    and two chunks of one repo are never counted as two repos. Splitting an
    oversized repo here rather than in the worker keeps both true: every chunk
    is already inside the per-batch limits when it reaches the queue.

    Repos come out in first-seen order, so a draw `candidates` clustered stays
    clustered and a `--order uniform` draw is unaffected beyond the grouping.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["full_name"], []).append(row)

    chunks: list[list[dict[str, Any]]] = []
    for members in groups.values():
        chunk: list[dict[str, Any]] = []
        cost = 0
        for row in members:
            chunk.append(row)
            cost += _graphql_cost(row)
            if len(chunk) >= config.PROBE_BATCH or cost >= config.GRAPHQL_BATCH_BYTES:
                chunks.append(chunk)
                chunk, cost = [], 0
        if chunk:
            chunks.append(chunk)
    return chunks


def probe_one(gh: GitHub, row: dict[str, Any], store: BlobStore | None,
              head_bytes: int,
              which: tuple[str, ...] = sniffers.DEFAULT,
              pool: ProcessPoolExecutor | None = None,
              body: bytes | None = None) -> dict[str, Any]:
    """Fetch one file's head and describe it, in the calling thread.

    The whole probe for one row. `run` performs the fetch and `describe` in
    separate stages; this composes them for a caller wanting one file.
    """
    size, is_lfs = row["size"], bool(row["is_lfs"])
    # `body` is a blob the caller already has from a batched GraphQL read.
    if body is not None:
        head, status, via = body, 200, "graphql"
    else:
        head, status = gh.head_bytes(row["full_name"], row["path"],
                                     min(size, head_bytes), is_lfs,
                                     ref=row["ref"])
        via = "raw"
    return describe(row, head, status, via, store, which, pool)


def describe(row: dict[str, Any], head: bytes | None, status: int | None,
             via: str, store: BlobStore | None = None,
             which: tuple[str, ...] = sniffers.DEFAULT,
             pool: ProcessPoolExecutor | None = None) -> dict[str, Any]:
    """One heads record for a file whose head has already been fetched.

    Everything here is local -- sniffing, the two content flags, the optional
    cache write -- so it runs in the sniff stage rather than on a fetch thread.
    """
    size = row["size"]
    base = {"content_key": row["content_key"], "bucket": row["bucket"],
            "size": size, "ts": int(time.time())}
    if head is None:
        # No file in hand. `http` decides whether it is worth another try: 404
        # means the path does not resolve at the pinned ref and never will.
        return base | {"fetched": False, "http": status, "via": via}

    # What the bytes are is *recorded*, never used to skip the sniffers, so the
    # binary/HTML policy can live in a selector and change without refetching.
    lowered = head[:256].lstrip().lower()
    record = base | {
        "fetched": True,
        "http": status,
        # Which path served the bytes; the two are not interchangeable
        # (graphql refuses non-UTF-8). Also set on failures, which is how
        # `index._unfetched` tells "tried" from "never drawn".
        "via": via,
        "head_len": len(head),
        "is_lfs_pointer": head.startswith(validate.LFS_MAGIC),
        "looks_like_html": any(lowered.startswith(m) for m in validate.HTML_MARKERS),
    }

    verdicts = _sniff(pool, head, size, which)
    record = record | {"agreement": sniffers.agreement(verdicts)} | verdicts
    if store is not None:
        record["head_blob"] = store.put(head)
    return record


def _next_batch(work: "queue.Queue",
                carry: list[dict[str, Any]] | None
                ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    """One batch of repo chunks, plus the chunk that did not fit.

    Every limit is checked *before* a chunk goes in; checking after would let
    a batch near the byte cap admit a whole further chunk and reach twice it.

    The chunk that does not fit is handed back rather than pushed onto the
    queue, so it stays with this worker and keeps its place in the draw order
    (which `--order repo` relies on to keep each batch to few repos).

    A chunk is always accepted into an empty batch. `_repo_chunks` has already
    split anything oversized, so the alternative is a batch of nothing and a
    worker that never drains its queue.
    """
    batch: list[dict[str, Any]] = []
    planned = 0
    repos = 0
    while repos < config.GRAPHQL_BATCH_REPOS:
        if carry is None:
            try:
                carry = work.get_nowait()
            except queue.Empty:
                break
        # What these rows will cost in the response; rows routed to raw cost
        # nothing.
        cost = sum(_graphql_cost(row) for row in carry)
        if batch and (planned + cost > config.GRAPHQL_BATCH_BYTES
                      or len(batch) + len(carry) > config.PROBE_BATCH):
            break
        batch.extend(carry)
        planned += cost
        repos += 1
        carry = None
    return batch, carry


def pool_size() -> int:
    """Files the probe will eventually describe, probed or not.

    The denominator of the progress line.
    """
    return int(index.files().filter(~pl.col("lfs_suspect"))
               .select(pl.len()).collect(engine="streaming").item())


def outstanding_size() -> int:
    """Files still to probe. The numerator of an honest ETA.

    Needs the anti-join against the heads log that the resume cursor avoids
    paying once per pass; once per invocation is affordable, and `compact`
    makes it cheaper still.

    Seals first, for the reason `run` does: an unsealed shard is invisible to
    every reader of the log.
    """
    log.seal_orphans(HEADS_LOG)
    return int(index.files().filter(~pl.col("lfs_suspect"))
               .select("content_key")
               .join(retired_keys(), on="content_key", how="anti")
               .select(pl.len()).collect(engine="streaming").item())


def _split_routes(rows: list[dict[str, Any]]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The draw, divided between the two fetch paths it will actually take.

    Split up front because the two paths are metered by different budgets
    and must not share a worker: a batch worker that also fetched its raw rows
    would leave its GraphQL slot idle meanwhile.
    """
    batched: list[dict[str, Any]] = []
    ranged: list[dict[str, Any]] = []
    for row in rows:
        (batched if _via_graphql(row) else ranged).append(row)
    return batched, ranged


def run(gh: GitHub, limit: int | None = None,
        workers: int = config.GRAPHQL_MAX_INFLIGHT,
        cache: bool = False, cache_dir: str | None = None,
        head_bytes: int = config.HEAD_BYTES,
        which: tuple[str, ...] = sniffers.DEFAULT, seed: int = 0,
        sniff_procs: int = config.SNIFF_PROCESSES,
        order: str = "uniform",
        raw_workers: int = config.RAW_MAX_INFLIGHT,
        after: tuple[int, int] | None = None,
        pool_total: int | None = None,
        carried: dict[str, int] | None = None,
        on_event: Any = None) -> dict[str, Any]:
    """One pass: draw candidates, fetch their heads, describe them.

    Three stages, each with its own threads, because each is bounded by
    something different and none of them may wait on another:

      * `_batch_worker` reads blobs over GraphQL, bounded by the point
        budget: one point per query whatever it carries.
      * `_range_worker` reads a head by HTTP Range, bounded by the per-IP
        request budget on raw.githubusercontent.com.
      * `_sniff_worker` runs the sniffers, which are CPU-bound and live in a
        process pool.

    The queue between fetch and sniff is bounded (`config.SNIFF_QUEUE`), and
    that bound is the whole of the backpressure.
    """
    emit = on_event or (lambda *a, **k: None)
    # Before the draw: a killed run leaves its shard `.active`, which every
    # reader of the log ignores, so the draw would redraw everything in it.
    log.seal_orphans(HEADS_LOG)
    todo = candidates(limit, order=order, seed=seed, after=after)
    batched, ranged = _split_routes(todo)

    gql_work: queue.Queue = queue.Queue()
    for chunk in _repo_chunks(batched):
        gql_work.put(chunk)
    raw_work: queue.Queue = queue.Queue()
    for row in ranged:
        raw_work.put(row)
    sniff_work: queue.Queue = queue.Queue(maxsize=config.SNIFF_QUEUE)

    stats: dict[str, Any] = {"probed": 0, "errors": 0, "complete": 0,
                             "batch_errors": 0, "queued": len(todo),
                             "batched": len(batched), "ranged": len(ranged),
                             "declined": 0, "retired": 0,
                             "graphql_batches": 0, "graphql_batch_files": 0}
    if todo and "_repo_rank" in todo[-1]:
        # Where the next pass of this run resumes. Taken off the draw rather
        # than off what was fetched: a row this pass failed on is redrawn by
        # the next *run*, which starts from the log.
        stats["cursor"] = (todo[-1]["_repo_rank"], todo[-1]["_key_rank"])
    lock = threading.Lock()
    queued = len(todo)
    # A worker process dying is a bug in a sniffer, not a fact about a file, so
    # nothing here retires a key or records a verdict for it. It is carried out
    # to the caller instead, because a thread that raises only kills itself.
    crash: list[BaseException] = []
    stopping = threading.Event()
    # Stop/accept handshake for `util.run_workers`.
    accepting = threading.Event()
    accepting.set()
    # Stage-drained flags, set by the last worker of each stage to leave. A
    # downstream stage may only stop on an empty queue once the stage above it
    # has finished, or it races a producer: a batch worker hands its declined
    # blobs to `raw_work`, and a range worker that had already found that queue
    # empty would exit and strand them.
    batching = threading.Event()
    ranging = threading.Event()
    live = {"batch": workers, "range": raw_workers}
    # When the progress line was last repainted (see config.PROGRESS_INTERVAL),
    # and the name it carried, so the idle repaint in the join loop does not
    # blank the file it was showing.
    painted: list[Any] = [0.0, ""]

    def _retire(stage: str, flag: threading.Event) -> None:
        with lock:
            live[stage] -= 1
            last = live[stage] <= 0
        if last:
            flag.set()

    def _progress() -> tuple[int, int, str]:
        """What the display shows: files done, files in total, and the note.

        All three come from one place so they count the same thing: run-wide
        figures (`carried` holds earlier passes' totals), not per-pass ones.

        Counts `retired` rather than `probed`, so that done + outstanding is
        the pool exactly: a transient failure is a file probed and not settled,
        and it comes round again. Falls back to this pass's own counts when the
        caller gave no pool figures.
        """
        prior = carried or {}
        note = (f"{prior.get('settled', 0) + stats['retired']:,} retired · "
                f"{prior.get('errors', 0) + stats['errors']:,} errors")
        if pool_total is None:
            return stats["probed"], queued, note
        return prior.get("retired", 0) + stats["retired"], pool_total, note

    def _handoff(job: tuple) -> bool:
        """Queue one fetched head for the sniffers. False means stop.

        Times out rather than blocking outright so a Ctrl-C during a stall in
        the sniff stage is still seen.

        Each timeout is counted: a full queue means the fetchers are waiting
        on the sniffers, and `_sniff_worker` counts the opposite. Both are in
        units of `_STALL`, so `sniff_blocked_s` and `fetch_blocked_s` read as
        thread-seconds spent waiting, and the larger names the bottleneck.
        """
        while not stopping.is_set():
            try:
                sniff_work.put(job, timeout=_STALL)
                return True
            except queue.Full:
                with lock:
                    stats["sniff_blocked_s"] = round(
                        stats.get("sniff_blocked_s", 0.0) + _STALL, 1)
        return False

    store = BlobStore(config.cache_root(cache_dir)) if cache else None
    pool = _sniff_pool(sniff_procs, which)
    try:
        with log.Writer(HEADS_LOG, rotate=config.HEADS_ROTATE) as heads_log:
            def _batch_worker() -> None:
                # A batch is assembled from whole repo chunks, never single rows:
                # `GitHub.blobs` is charged per repository resolution, so a batch
                # dealt out row by row would span as many repos as it has rows.
                try:
                    carry: list[dict[str, Any]] | None = None
                    while not stopping.is_set():
                        batch, carry = _next_batch(gql_work, carry)
                        if not batch:
                            return
                        with lock:
                            # Batches sent. `graphql_requests` counts attempts,
                            # so the ratio shows batches costing retries.
                            stats["graphql_batches"] += 1
                            stats["graphql_batch_files"] += len(batch)
                        try:
                            bodies = gh.blobs(batch)
                        except Exception as exc:
                            # A failed batch is not a fact about any file in
                            # it; every row falls back to a range request. The
                            # first error is kept so the fallback is not silent.
                            bodies = [None] * len(batch)
                            with lock:
                                stats["batch_errors"] += 1
                                if "batch_error" not in stats:
                                    stats["batch_error"] = repr(exc)[:200]
                        for row, body in zip(batch, bodies):
                            if stopping.is_set():
                                return
                            if body is None:
                                # GitHub declined this one blob (binary, or not
                                # valid UTF-8). Handed to the range path so this
                                # worker keeps its GraphQL slot busy.
                                with lock:
                                    stats["declined"] += 1
                                raw_work.put(row)
                            elif not _handoff((row, body, 200, "graphql")):
                                return
                finally:
                    _retire("batch", batching)

            def _range_worker() -> None:
                try:
                    while not stopping.is_set():
                        try:
                            row = raw_work.get(timeout=0.2)
                        except queue.Empty:
                            # Only once no batch worker can still hand one
                            # over (declined blobs). See `batching`.
                            if batching.is_set():
                                return
                            continue
                        try:
                            head, status = gh.head_bytes(
                                row["full_name"], row["path"],
                                min(row["size"], head_bytes),
                                bool(row["is_lfs"]), ref=row["ref"])
                        except Exception as exc:
                            head, status = None, None
                            with lock:
                                stats.setdefault("range_error", repr(exc)[:200])
                        if not _handoff((row, head, status, "raw")):
                            return
                finally:
                    _retire("range", ranging)

            def _sniff_worker() -> None:
                while True:
                    try:
                        job = sniff_work.get(timeout=_STALL)
                    except queue.Empty:
                        # `ranging` is set only after `batching`, so this one
                        # flag says both fetch stages are finished.
                        if ranging.is_set():
                            return
                        # An empty queue is the sniffers waiting on the
                        # network, the mirror of the count in `_handoff`.
                        with lock:
                            stats["fetch_blocked_s"] = round(
                                stats.get("fetch_blocked_s", 0.0) + _STALL, 1)
                        continue
                    if not _write(*job):
                        return

            def _write(row: dict[str, Any], head: bytes | None,
                       status: int | None, via: str) -> bool:
                """Describe one fetched head and log it. False means stop."""
                try:
                    rec = describe(row, head, status, via, store, which, pool)
                except BrokenProcessPool as exc:
                    # Every file in flight fails with this, not just the one
                    # that killed the worker, so it says nothing about this
                    # key. Treating it as an ordinary failure would carry on
                    # fetching files only to throw them away; stop instead.
                    with lock:
                        crash.append(exc)
                    stopping.set()
                    return False
                except Exception as exc:
                    # Shaped like `describe`'s own failure record, with `via`
                    # set: this file was tried.
                    rec = {"content_key": row["content_key"],
                           "bucket": row["bucket"], "size": row["size"],
                           "ts": int(time.time()), "fetched": False,
                           "http": None, "via": via,
                           "message": repr(exc)[:200]}
                with lock:
                    if not accepting.is_set():
                        # Interrupted, and this fetch missed the grace
                        # period. The writer is closing or closed, so the
                        # record has nowhere to go; an unwritten key is an
                        # unprobed one and `probe` draws it again.
                        return False
                    heads_log.write(rec)
                    stats["probed"] += 1
                    stats["errors"] += not rec.get("fetched")
                    # Keys this pass took out of the pool. A transient
                    # failure raises `probed` without settling anything, so
                    # only this tells progress from redrawing the same rows.
                    stats["retired"] += settles(rec)
                    stats["complete"] += bool(
                        sniffers.verdict(rec, "custom").get("complete"))
                    # Repainted on a clock rather than per file; see
                    # config.PROGRESS_INTERVAL.
                    now = time.monotonic()
                    due = now - painted[0] >= config.PROGRESS_INTERVAL
                    if due:
                        painted[0] = now
                        painted[1] = row["path"][-28:]
                if due:
                    done, of, note = _progress()
                    emit("repo", done=done, of=of, note=note, name=painted[1])
                return True

            threads = (
                [threading.Thread(target=_batch_worker, daemon=True)
                 for _ in range(workers)]
                + [threading.Thread(target=_range_worker, daemon=True)
                   for _ in range(raw_workers)]
                + [threading.Thread(target=_sniff_worker, daemon=True)
                   for _ in range(config.sniff_threads(sniff_procs))]
            )

            def _repaint_idle() -> None:
                # Repaint even when nothing was written, so a stalled pipeline
                # shows its rate falling rather than a frozen line.
                with lock:
                    idle = (time.monotonic() - painted[0]
                            >= config.PROGRESS_INTERVAL)
                    if idle:
                        painted[0] = time.monotonic()
                if idle:
                    done, of, note = _progress()
                    emit("repo", done=done, of=of, note=note, name=painted[1])

            def _announce_interrupt() -> None:
                # The sniffer processes ignore SIGINT (see
                # `sniffers.worker_init`), so in-flight sniffs still finish
                # within the grace period.
                done, of, note = _progress()
                emit("repo", done=done, of=of, note=note,
                     name="interrupted, finishing in-flight files")

            run_workers(threads, stats, stopping=stopping, accepting=accepting,
                        lock=lock, on_interrupt=_announce_interrupt,
                        on_tick=_repaint_idle)
            if crash:
                # Raised inside the writer's `with`, so everything probed
                # before the crash is sealed into the log and keeps its
                # verdicts. Keys that were in flight were never written, and
                # an unwritten key is an unprobed one, so `probe` redraws them
                # next run without any of them being marked as anything.
                raise RuntimeError(
                    "a sniffer subprocess died (segfault, or the OOM killer). "
                    "That is a bug in a sniffer rather than a property of any "
                    "file, so nothing has been recorded against the files that "
                    "were in flight and re-running `probe` will draw them "
                    "again. Re-run with --sniff-procs 0 to sniff in-process "
                    "and get the real traceback."
                ) from crash[0]
    finally:
        # What each budget spent, and where the raw rate controller settled.
        for limiter in ("raw", "core_bucket", "graphql_bucket"):
            if hasattr(getattr(gh, limiter, None), "stats"):
                stats |= getattr(gh, limiter).stats
        if pool is not None:
            pool.shutdown(wait=True)
        # Peak RSS of this process: this pass is the scraper's memory
        # high-water mark (the draw, plus a GRAPHQL_BATCH_BYTES response per
        # batch worker). RUSAGE_CHILDREN is no use for the sniff workers: they
        # fork before exec and are charged the parent's resident set. It is a
        # lifetime high-water mark, which is why `cli` takes the max across
        # passes.
        import resource

        stats["peak_rss_mb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if store is not None:
            # Close before measuring: the shard is still buffered in userspace,
            # so stat() on the open file reports 0.
            store.close()
            stats["cache_dir"] = str(store.root)
            stats["cache_bytes"] = store.bytes_written
    return stats


def resniff(source: str, target: str = HEADS_LOG,
            cache_dir: str | None = None,
            which: tuple[str, ...] = ("custom",),
            sniff_procs: int = config.SNIFF_PROCESSES,
            limit: int | None = None, on_event: Any = None,
            skip_unchanged: bool = False) -> dict[str, Any]:
    """Sniff the heads of an earlier run again, from the cache instead of the
    network, and write the results to the `target` log.

    `source` is the heads log of the earlier run, moved aside first, e.g.
    `data/heads` renamed to `data/heads-old` and passed as "heads-old". It
    cannot simply be deleted: the cache is one big pile of bytes, and only the
    `head_blob` entry of each record says where in it a head lies. The new
    records carry it over, so once this has run through, the source log is
    redundant.

    Only the sniffers in `which` are run. What the others said is copied from
    the old record, so that a change to our own sniffer does not cost a rerun
    of the much slower CleverCSV and DuckDB.

    `skip_unchanged` also keeps the old CleverCSV and DuckDB verdicts for
    heads on which they would get the same input as before, see
    `sniffers.run_changed`.

    A record whose head is not in the cache is left out. Its file then counts
    as not probed and the next ordinary `probe` fetches it from the network.
    Records of failed fetches are copied, since a 404 still settles a file.

    Can be interrupted (Ctrl-C) and run again: records are written in the
    order of the source log, so a rerun finds the last record of the target in
    the source and carries on after it. That only works while nothing else
    writes to `target`, so finish this before the next ordinary `probe`.

    The returned counts include `seconds` and `sniffed_per_s`, the average
    speed of this run.
    """
    emit = on_event or (lambda *a, **k: None)
    if source == target:
        raise ValueError("resniff reads one log and writes another: move the "
                         f"old {target!r} log aside and pass its new name")
    store = BlobStore(config.cache_root(cache_dir))
    log.seal_orphans(target)
    # Where an earlier, interrupted run stopped: the last record it wrote.
    # Counting records is not enough to find it in the source, because heads
    # that were not in the cache have no record in the target.
    written, last = 0, None
    for record in log.read(target):
        written, last = written + 1, (record.get("content_key"), record.get("ts"))
    stats = {"skipped": 0, "sniffed": 0, "copied": 0, "not_cached": 0}
    started = time.monotonic()
    pool = _sniff_pool(sniff_procs, which)
    # Sniffed in parallel but written in source order, see above. The window
    # bounds how many heads are in memory at once.
    window: collections.deque = collections.deque()
    width = max(1, sniff_procs) * 4

    def write_oldest(heads_log: log.Writer) -> None:
        # Taken off the window only once it is written, so that an interrupt
        # in between cannot lose the record.
        old, verdicts = window[0]
        if verdicts is not None:
            # A future when there is a pool, the verdicts themselves without.
            fresh = verdicts.result() if pool is not None else verdicts
            old = old | fresh
            old["agreement"] = sniffers.agreement(old)
            stats["sniffed"] += 1
        else:
            stats["copied"] += 1
        heads_log.write(old)
        window.popleft()

    try:
        with log.Writer(target, rotate=config.HEADS_ROTATE) as heads_log:
            try:
                for n, old in enumerate(log.read(source)):
                    if last is not None:
                        # Still in the part an earlier run has dealt with. It
                        # ends at the first record from position `written - 1`
                        # on that is the last one written.
                        if n >= written - 1 and (
                                old.get("content_key"), old.get("ts")) == last:
                            last = None
                            # the speed is that of the sniffing alone
                            started = time.monotonic()
                        stats["skipped"] += 1
                        continue
                    if limit is not None and n - stats["skipped"] >= limit:
                        break
                    # The writer gives every record a new id.
                    old.pop("rec_id", None)
                    verdicts = None
                    if old.get("fetched"):
                        try:
                            head = store.get(old["head_blob"])
                        except (KeyError, OSError, zstandard.ZstdError):
                            stats["not_cached"] += 1
                            continue
                        # Old verdicts that can be kept: the sniffer ran and did
                        # not fail.
                        have = tuple(
                            name for name in which
                            if skip_unchanged and old.get(name)
                            and not old[name].get("error"))
                        job = (sniffers.run_changed, head, old["size"], which, have)
                        verdicts = (job[0](*job[1:]) if pool is None
                                    else pool.submit(*job))
                    window.append((old, verdicts))
                    if len(window) >= width:
                        write_oldest(heads_log)
                    if n % 100 == 0:
                        emit("resniff", done=n, sniffed=stats["sniffed"],
                             not_cached=stats["not_cached"])
            except KeyboardInterrupt:
                # Stop reading, but still write what is being sniffed.
                stats["interrupted"] = True
            while window:
                write_oldest(heads_log)
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    stats["seconds"] = round(time.monotonic() - started, 1)
    stats["sniffed_per_s"] = round(stats["sniffed"] / max(stats["seconds"], 0.1), 1)
    return stats


# The pass that fills each heads log, so an empty one can say what to run.
# The Kaggle name is spelled out rather than imported: `kaggle` imports this
# module, and a test asserts the two stay equal.
FILLED_BY = {HEADS_LOG: "probe", "kaggle/heads": "kaggle-bundle"}


def report(log_name: str = HEADS_LOG) -> str:
    """Distribution across everything probed, per sniffer.

    `log_name` selects which heads log is counted: the Kaggle pass writes the
    same record shape through the same `describe`, so both pools are
    summarised by this one function and stay comparable.

    One streaming pass over the log; holding the records would not fit in
    memory at scale.
    """
    if not list(log.log_dir(log_name).glob("part-*.jsonl*")):
        return ("nothing probed yet: run `python -m csvscrape "
                f"{FILLED_BY.get(log_name, 'probe')}`")
    import collections

    n = 0
    complete = 0
    unfetched: collections.Counter = collections.Counter()
    # Indicators, not exclusions: these rows are in the pool and carry verdicts.
    # The `text-only` selector is what acts on them.
    flagged: collections.Counter = collections.Counter()
    delim: collections.Counter = collections.Counter()
    agree: collections.Counter = collections.Counter()
    axes = ("quote_style", "quotes_in_unquoted_fields",
            "double_quote_escape", "escape_in_unquoted_fields",
            "terminator", "comment", "quoting")
    custom_axes: dict[str, collections.Counter] = {
        field: collections.Counter() for field in axes}
    enc: collections.Counter = collections.Counter()
    for r in log.read(log_name):
        if not r.get("fetched"):
            unfetched[r.get("http")] += 1
            continue
        n += 1
        custom = sniffers.verdict(r, "custom")
        complete += bool(custom.get("complete"))
        if custom.get("magic_kind"):
            flagged["magic/" + custom["magic_kind"]] += 1
        if r.get("looks_like_html"):
            flagged["html"] += 1
        if r.get("is_lfs_pointer"):
            flagged["lfs_pointer"] += 1
        for name, v in sniffers.delimiters(r).items():
            delim[(name, v)] += 1
        agree[r.get("agreement", 0)] += 1
        for field in axes:
            custom_axes[field][custom.get(field)] += 1
        enc[sniffers.verdict(r, "charset_normalizer").get("encoding")] += 1
    if not n:
        return "no successful probes"

    out = [f"{n:,} files probed · {complete:,} read in full "
           f"({100 * complete / n:.0f}%)"]
    if unfetched:
        # A record with no status is a fetch that never got one: a transport
        # failure on the GitHub side, a member that would not come out of the
        # archive on the Kaggle one. "http None" reads like a status code.
        out.append("  not fetched: " + "  ".join(
            (f"http {st}:{c:,}" if st else f"no status:{c:,}")
            for st, c in unfetched.most_common(6)))
    if flagged:
        out.append("  flagged (kept, filter with `--select text-only`): " +
                   "  ".join(f"{k}:{c:,}" for k, c in flagged.most_common(6)))
    out.append("")

    for name in ("custom", "clevercsv", "duckdb"):
        top = sorted(((v, c) for (nm, v), c in delim.items() if nm == name),
                     key=lambda kv: -kv[1])[:6]
        if top:
            out.append(f"  delimiter/{name:<10}" +
                       "  ".join(f"{v!r}:{c:,}" for v, c in top))

    out.append("\n  sniffer agreement on delimiter (votes for the winner):")
    for k in sorted(agree, reverse=True):
        out.append(f"    {k} of 3   {agree[k]:>6,}  ({100*agree[k]/n:.0f}%)")

    out.append("\n  custom -- axes the other sniffers do not model:")
    for field in axes:
        c = custom_axes[field]
        out.append(f"    {field:<26}" +
                   "  ".join(f"{k!r}:{v:,}" for k, v in c.most_common(4)))

    out.append("\n  charset_normalizer encoding: " +
               "  ".join(f"{k!r}:{v:,}" for k, v in enc.most_common(6)))
    return "\n".join(out)


# Shape fields promoted to top-level columns, because selectors filter on them.
FLAT_FIELDS = (
    "delimiter", "quote", "quote_style", "quoting", "terminator",
    "comment", "escape", "double_quote_escape",
    "escape_in_unquoted_fields", "quotes_in_unquoted_fields",
    "embedded_newline", "embedded_delimiter", "preamble_records",
    "ragged_rows", "mixed_terminators", "errors", "delimiter_found",
    "has_nul", "common_field_count", "complete", "encoding", "encoding_class",
    "magic_kind", "control_frac", "wide_encoding",
)

# Recorded by probe_one rather than by a sniffer: facts about the fetch and the
# first bytes. Promoted alongside the Shape fields so the binary/HTML policy can
# live in a selector instead of in the ingest path.
PROBE_FIELDS: dict[str, Any] = {
    "fetched": pl.Boolean,
    "http": pl.Int64,
    "via": pl.String,
    "is_lfs_pointer": pl.Boolean,
    "looks_like_html": pl.Boolean,
}

_DTYPES: dict[type, pl.DataType] = {
    str: pl.String, bool: pl.Boolean, int: pl.Int64, float: pl.Float64,
}


def _shape_dtype(name: str) -> pl.DataType:
    """The column type for one `Shape` field, read off its annotation.

    Shared by the empty frame and by the scan schema, so a field added to
    `Shape` cannot be typed one way in one and another way in the other.
    """
    from .features import Shape

    base = Shape.__annotations__.get(name, "str").replace(" | None", "").strip()
    if base == "list[int]":
        return pl.List(pl.Int64)
    return _DTYPES.get({"str": str, "bool": bool, "int": int,
                        "float": float}.get(base, str), pl.String)


def _empty_metadata() -> pl.LazyFrame:
    """A typed empty frame with the full column set.

    Typing matters even when there are no rows: an untyped empty frame gives
    `content_key` the Null dtype, and the left join in `sample.candidates` then
    fails outright rather than yielding nulls. Missing columns fail the same
    way, in the selector rather than the join. Types come from the `Shape`
    annotations so the two cannot drift apart.
    """
    schema: dict[str, pl.DataType] = {
        "content_key": pl.String, "agreement": pl.Int64,
    }
    for name in FLAT_FIELDS:
        schema[name] = _shape_dtype(name)
    schema |= PROBE_FIELDS
    schema["cn_encoding"] = pl.String
    schema["chardet_encoding"] = pl.String
    return pl.LazyFrame(schema=schema)


def metadata_schema() -> dict[str, "pl.DataType"]:
    """Every column `compact` folds probe verdicts into, with its dtype.

    The index only carries these columns once there is a heads log to fold
    in, and a selector naming a missing column fails outright, so
    `sample.candidates` supplies the missing ones as typed nulls off this.
    """
    return dict(_empty_metadata().collect_schema())


def metadata_rows(log_name: str = HEADS_LOG) -> pl.LazyFrame | None:
    """The projected probe records, one row per *record*, not per content.

    `log_name` selects which heads log is read; the Kaggle pass writes the
    same record shape to its own log, so one set of selectors works over both
    pools.

    Split out of `metadata` so `compact` can dedup a bucket at a time: dedup
    is a sort, and sorting the whole log at once does not fit in memory.

    Carries `ts`/`rec_id` so `dedupe` can order the records; `dedupe` drops
    them again. Returns None when there is nothing to read at all, so the
    caller can skip the join entirely.
    """
    parts = log.sealed(log_name)
    if not parts:
        return None
    # Records for files that were never fetched are kept, with null verdicts:
    # `fetched`, `http` and `via` are the fetch's outcome, and the index needs
    # them (a 404 settles a file). Nothing here filters on *what* a file is;
    # that stays in the selectors.
    frame = _scan_heads(parts)
    present = frame.collect_schema().names()
    if "content_key" not in present:
        return None

    # A log with no successful fetch has no `custom` or `agreement` column,
    # so both are supplied as typed nulls rather than failing the fold.
    cols = [pl.col("content_key")]
    cols.append(pl.col("agreement") if "agreement" in present
                else pl.lit(None, dtype=pl.Int64).alias("agreement"))
    cols += ([pl.col("custom").struct.field(name).alias(name)
              for name in FLAT_FIELDS] if "custom" in present
             else [pl.lit(None, dtype=_shape_dtype(name)).alias(name)
                   for name in FLAT_FIELDS])
    for name, dtype in PROBE_FIELDS.items():
        cols.append(pl.col(name).cast(dtype) if name in present
                    else pl.lit(None, dtype=dtype).alias(name))
    for sniffer, alias in (("charset_normalizer", "cn_encoding"),
                           ("chardet", "chardet_encoding")):
        if sniffer in present:
            cols.append(pl.col(sniffer).struct.field("encoding").alias(alias))
        else:
            cols.append(pl.lit(None, dtype=pl.String).alias(alias))
    # `ts`/`rec_id` come along only to order the dedup, then leave again. Both
    # are present on anything the writer produced; a hand-built log without
    # them falls back to scan order inside `newest_per`.
    ordering = [c for c in ("ts", "rec_id") if c in present]
    return frame.select(cols + [pl.col(c) for c in ordering])


def dedupe(rows: pl.LazyFrame) -> pl.LazyFrame:
    """The record to keep per content_key: fetched beats unfetched, then newest.

    Takes the frame rather than reading the log so it can be applied to one
    bucket at a time. It is exact on any subset that is closed under
    content_key -- every record for a key lands in the same bucket, because
    the bucket is a hash of the key -- so bucketing changes what it costs and
    not what it answers.

    A content key is a content address, so verdicts taken off those bytes stay
    true however the path that served them fares later: a file described once
    and 404 later has moved, not become undescribed. So any success beats any
    failure, and recency decides within each class.
    """
    names = rows.collect_schema().names()
    ordering = [c for c in ("ts", "rec_id") if c in names]
    if "fetched" in names:
        deduped = (
            rows.sort(pl.col("fetched").fill_null(False), *ordering)
            .unique(subset=["content_key"], keep="last", maintain_order=True)
        )
    else:
        deduped = log.newest_per(rows, ["content_key"])
    return deduped.drop(ordering) if ordering else deduped


def metadata(log_name: str = HEADS_LOG) -> pl.LazyFrame:
    """Probe results as a flat frame, keyed by content_key.

    `custom` is flattened into top-level columns because selectors filter on it;
    the other sniffers stay nested and verbatim, reachable by anyone who wants
    them but never normalised into a lossy common shape.

    An unprobed pool -- or one whose heads log predates the current record
    shape -- yields the typed empty frame rather than an error, so metadata
    selectors return nothing instead of failing.
    """
    rows = metadata_rows(log_name)
    return _empty_metadata() if rows is None else dedupe(rows)
