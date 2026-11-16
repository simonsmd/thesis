"""Drawing a corpus out of the indexed pool, in one of two shapes.

Two datasets are wanted and they want opposite things, so the draw is
parameterised by which one is being built rather than duplicated:

  * **log-uniform** over size, for throughput benchmarking: an equal count
    per log2 octave. Buckets are not equally supplied though -- bucket 13 holds
    thousands of files while bucket 30 may hold two -- so a flat N/B quota
    would silently under-deliver. Allocation is therefore water-filling: every
    bucket is offered an equal share, whatever a starved bucket cannot take is
    returned to the pot, and the surplus is redistributed among buckets that
    still have capacity until N is met or the pool runs dry. Within an octave
    every file is equally likely. The pool is already close to log-uniform
    inside an octave, so an extra 1/size weight there pushed files to the
    lower end (36/28/21/15% per quarter octave instead of 29/27/23/20%).

  * **uniform** over the pool, for correctness and cross-parser comparison.
    Size carries no information there, so stratifying by it would be a
    deliberate distortion: a uniform draw is the one whose results generalise
    to "a CSV file on GitHub", and it is also cheap, because the pool's mass
    sits in the small buckets a log-uniform draw has to climb out of.

The two share everything else -- the diversity caps, the redraw loop, the
selector gates -- by sharing a single notion of a *stratum*. Log-uniform
strata are size octaves; a uniform draw has exactly one stratum holding the
whole pool, so the same water-filling, the same per-stratum quota and the same
deficit arithmetic describe both without a second code path.
"""
from __future__ import annotations

import random
from typing import Any

import polars as pl

from . import index

LOG_UNIFORM = "log-uniform"
UNIFORM = "uniform"
MODES = (LOG_UNIFORM, UNIFORM)

# The stratum id of a uniform draw. Not a bucket number and never confusable
# with one: real buckets are floor(log2(size)) and cannot be negative.
WHOLE_POOL = -1


def stratum(mode: str) -> pl.Expr:
    """The column a draw allocates its quota across."""
    if mode == LOG_UNIFORM:
        return pl.col("bucket")
    if mode == UNIFORM:
        return pl.lit(WHOLE_POOL, dtype=pl.Int32)
    raise SystemExit(f"unknown sampling mode {mode!r}; choose from {', '.join(MODES)}")


def allocate(available: dict[int, int], n: int) -> dict[int, int]:
    """Water-fill n across strata, respecting each stratum's supply."""
    alloc = {b: 0 for b in available}
    remaining = n
    while remaining > 0:
        hungry = [b for b in available if alloc[b] < available[b]]
        if not hungry:
            break
        share = remaining // len(hungry)
        if share == 0:
            # Fewer slots left than strata: hand them out one apiece, smallest
            # bucket first so the scarce large buckets are not starved by a tie.
            for bucket in sorted(hungry):
                if remaining == 0:
                    break
                alloc[bucket] += 1
                remaining -= 1
            break
        for bucket in hungry:
            take = min(share, available[bucket] - alloc[bucket])
            alloc[bucket] += take
            remaining -= take
    return {b: k for b, k in alloc.items() if k}


def candidates(exclude: set[str] | None = None,
               select: Any = None,
               pool: pl.LazyFrame | None = None,
               require_probed: bool | None = None,
               utf8: bool = False,
               source: str = "github") -> pl.LazyFrame:
    """One row per unique content, narrowed by a selector.

    `source` picks which pool -- the GitHub index or the Kaggle one. They carry
    the same columns on purpose, so every selector and both draw shapes work
    over either; what they are not is interchangeable within one corpus, which
    is why this takes a name and never a union.

    `select` is a selector name or a resolved `Selector`; `utf8` folds the
    whole-file UTF-8 requirement into it.

    Whether unprobed files are drawable is the selector's own answer, read off
    `needs_probe`. A selector that filters on probe metadata cannot describe a
    row whose metadata columns are all null, so it draws from the probed
    fraction only; one that does not -- `main` and the corpora built on it --
    draws from the whole pool and lets its verifier settle each file against
    the bytes that land.

    `require_probed` overrides that per call, for anyone deliberately
    inspecting one side of it.

    Returns a LazyFrame, and deliberately does not collect: callers need only
    per-stratum counts or a per-stratum subsample, and collecting the whole
    pool costs gigabytes.
    """
    from . import selectors

    chosen = select if isinstance(select, selectors.Selector) else selectors.resolve(
        select, utf8=utf8)
    if pool is None:
        # No sort, no unique, no join: `compact` collapsed the index to one row
        # per distinct content and folded the probe verdicts in as columns.
        frame = _with_metadata_columns(
            index.pool_for(source).filter(~pl.col("lfs_suspect")))
        probed = (require_probed if require_probed is not None
                  else chosen.needs_probe)
        if probed:
            # An index compacted before anything was probed has no `fetched`
            # column at all. That means nothing is probed, so nothing is
            # drawable under a selector that needs it -- which is the
            # invariant holding, not an error.
            frame = frame.filter(
                pl.col("fetched").fill_null(False)
                if "fetched" in frame.collect_schema().names()
                else pl.lit(False)
            )
        pool = chosen.filter(frame)
    if exclude:
        pool = pool.filter(~pl.col("content_key").is_in(list(exclude)))
    return pool


def _with_metadata_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Supply the probe columns an unprobed index does not have, as typed nulls.

    `compact` writes them only when there is a heads log to fold in, so an
    index built before the first probe pass has none of them, and a selector
    naming one would fail with `ColumnNotFoundError`. Null is what an unprobed
    row carries in a populated index too, so filling them in changes no answer.
    """
    from . import probe

    have = set(frame.collect_schema().names())
    missing = [pl.lit(None, dtype=dtype).alias(name)
               for name, dtype in probe.metadata_schema().items()
               if name not in have]
    return frame.with_columns(missing) if missing else frame


def coverage(source: str = "github") -> pl.DataFrame:
    """Probed share of the pool, per bucket.

    Only the probe-dependent selectors are bounded by this: `main` draws
    whether or not a file has been probed.
    """
    frame = index.pool_for(source).filter(~pl.col("lfs_suspect"))
    probed = (
        pl.col("fetched").fill_null(False)
        if "fetched" in frame.collect_schema().names()
        else pl.lit(False)
    )
    return (
        frame.group_by("bucket")
        .agg(pl.len().alias("unique_files"), probed.sum().alias("probed"))
        .sort("bucket")
        .collect(engine="streaming")
    )


def supply(pool: pl.LazyFrame, mode: str = LOG_UNIFORM,
           sizes: bool = False) -> dict[int, tuple[int, float]]:
    """(count, mean size) per stratum, in one streaming pass.

    The mean is only computed when `sizes` is set: only a uniform draw needs
    it (see `projected_bytes`), and projecting the size column is not free.
    """
    aggs = [pl.len().alias("n")]
    if sizes:
        aggs.append(pl.col("size").mean().alias("mean_size"))
    rows = (
        pool.group_by(stratum(mode).alias("_stratum"))
        .agg(aggs)
        .collect(engine="streaming")
        .iter_rows()
    )
    if sizes:
        return {s: (n, float(m or 0.0)) for s, n, m in rows}
    return {s: (n, 0.0) for s, n in rows}


def projected_bytes(quota: dict[int, int], mode: str = LOG_UNIFORM,
                    means: dict[int, tuple[int, float]] | None = None) -> int:
    """Rough total size of a corpus drawn to this quota.

    Worth printing before a download rather than discovering on the disk, and
    the two modes need different arithmetic because they weight sizes
    differently:

      * log-uniform draws an equal count per octave, and the middle of the
        octave is used as the expected size. Equal counts per octave means
        the top bucket dominates: at
        `-n 1000` over buckets 6-31, the ~38 files landing in bucket 31 are
        ~76 GB on their own against a few megabytes from the bottom twenty.
      * uniform draws with the pool's own size distribution, so the expected
        size is the pool's mean.
    """
    if mode == UNIFORM:
        means = means or {}
        return sum(k * int(means.get(s, (0, 0.0))[1]) for s, k in quota.items())
    return sum(k * int(2 ** b * 1.5) for b, k in quota.items())


def draw(
    quota: dict[int, int],
    seed: int = 0,
    per_repo: int = 2,
    per_owner: int = 5,
    exclude: set[str] | None = None,
    select: Any = None,
    pool: pl.LazyFrame | None = None,
    mode: str = LOG_UNIFORM,
    utf8: bool = False,
    repo_balance: float = 0.0,
) -> tuple[pl.DataFrame, dict[int, int]]:
    """Pick files to satisfy an explicit per-stratum quota.

    `repo_balance` sets how much a repo's file count matters. Each file is
    weighted by 1 / (files its repo has in the stratum) ** repo_balance: 0
    gives every file the same chance, so a repo with 5000 files is 5000 times
    as likely to be picked as a repo with one; 1 gives every repo in the
    stratum the same chance. The per-repo and per-owner caps only limit how
    many files one repo contributes, not how likely it is to be picked at all.

    Returns (rows, shortfall_by_stratum).
    """
    available = candidates(exclude, select, pool=pool, utf8=utf8)
    # One narrow pass for the stratum sizes, so each one below can be thinned
    # before it is materialised rather than after.
    supplies = {s: n for s, (n, _) in supply(available, mode).items()}
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    shortfall: dict[int, int] = {}
    repo_used: dict[int, int] = {}
    owner_used: dict[str, int] = {}

    # Largest bucket first. The per-repo and per-owner caps are a single global
    # budget, and the small buckets have abundant supply, so they should not
    # spend it before the scarce top buckets are reached.
    for bucket in sorted(quota, reverse=True):
        want = quota[bucket]
        # Only this stratum is ever materialised, and only a few multiples of
        # `headroom` rows of it. The headroom keeps the diversity caps
        # workable.
        #
        # The thinning is a hash residue of the content key, applied while the
        # stratum streams past, so the stratum is never held in memory (a
        # uniform draw's single stratum is the whole pool). The residue is
        # uniform and seeded, so the draw stays a reproducible uniform
        # pre-sample.
        headroom = max(want * 100, 2_000)
        sub_seed = _substream(seed, bucket)
        if repo_balance:
            order = _balanced_order(available, bucket, mode, repo_balance,
                                    sub_seed, headroom)
        else:
            modulus = max(1, supplies.get(bucket, 0) // (headroom * 2))
            rows = (
                index.compact_strings(
                    _stratum_rows(available, bucket, mode)
                    .filter(pl.col("content_key").hash(seed=sub_seed) % modulus == 0)
                )
                .collect(engine="streaming")
                .filter(pl.int_range(pl.len()).shuffle(seed=sub_seed) < headroom)
                .to_dicts()
            )
            # Every file in the stratum is equally likely in both modes; the
            # modes differ only in what a stratum is.
            rng.shuffle(rows)
            order = rows

        taken = 0
        deferred: list[dict[str, Any]] = []
        for row in order:
            if taken >= want:
                break
            repo, owner = row["repo_id"], row["owner"]
            if repo_used.get(repo, 0) >= per_repo or owner_used.get(owner, 0) >= per_owner:
                deferred.append(row)
                continue
            repo_used[repo] = repo_used.get(repo, 0) + 1
            owner_used[owner] = owner_used.get(owner, 0) + 1
            picked.append(row | {"stratum": bucket, "stratum_quota": want})
            taken += 1
        # Diversity caps are a preference, not a constraint: a stratum that can
        # only be filled by re-using a prolific repo is better filled than left
        # short, so the capped-out rows are reconsidered before giving up.
        for row in deferred:
            if taken >= want:
                break
            picked.append(row | {"stratum": bucket, "stratum_quota": want})
            taken += 1
        if taken < want:
            shortfall[bucket] = want - taken

    # Typed from the pool rather than inferred from the rows: inference looks
    # at the first rows only, and a metadata column null in those (unprobed
    # rows) would fail on the first later row that carries a value.
    schema = dict(available.collect_schema())
    schema["stratum"] = pl.Int64
    schema["stratum_quota"] = pl.Int64
    return pl.DataFrame(picked, schema=schema), shortfall


def _substream(seed: int, value: int) -> int:
    """A non-negative seed for one stratum's thinning and its shuffle.

    Each stratum gets its own substream so the thinning in one cannot correlate
    with the thinning in the next. Polars' `hash` takes an *unsigned* seed, so
    the offset keeps `WHOLE_POOL` (-1) non-negative and the mask covers a
    negative `--seed`.
    """
    return (seed + value - WHOLE_POOL) & 0xFFFF_FFFF


def _stratum_rows(pool: pl.LazyFrame, value: int, mode: str) -> pl.LazyFrame:
    """The rows of one stratum.

    A uniform draw's single stratum *is* the pool, and saying so costs nothing
    -- where `bucket == n` is a predicate the Parquet reader evaluates on its
    row-group statistics, `lit(-1) == -1` is one it would have to materialise
    the frame to discover is vacuous.
    """
    if mode == UNIFORM:
        return pool
    return pool.filter(pl.col("bucket") == value)


def _balanced_order(pool: pl.LazyFrame, value: int, mode: str, balance: float,
                    seed: int, limit: int) -> list[dict]:
    """The first `limit` rows of a weighted shuffle of one stratum, with each
    file's weight divided by its repo's file count ** `balance`.

    The unbalanced path first thins the stratum to a uniform random subset of
    files and weights that. Here the subset would be mostly files from the
    largest repos, and the small repos the weighting is meant to favour would
    be missing from it. So the whole stratum is ranked instead, by weighted
    sampling without replacement (Efraimidis-Spirakis keys in log space),
    computed in Polars from a seeded hash, and only the `limit` lowest keys
    are kept.

    The ranking reads only the two columns it needs; the full rows are read
    afterwards for the winners alone. Sorting full rows of the whole GitHub
    pool took ~20 GB.
    """
    rows = _stratum_rows(pool, value, mode)
    narrow = rows.select("content_key", "repo_id")
    repo_files = narrow.group_by("repo_id").agg(pl.len().alias("_repo_files"))
    # A uniform number in (0, 1) per file: the top 53 bits of the hash, so the
    # float is exact and never 0 or 1.
    u = ((pl.col("content_key").hash(seed=seed) // 2 ** 11).cast(pl.Float64) + 0.5) / 2.0 ** 53
    # log(-log(u) / weight), smallest first.
    key = (-u.log()).log() + balance * pl.col("_repo_files").cast(pl.Float64).log()
    winners = (
        index.compact_strings(
            narrow.join(repo_files, on="repo_id")
            .select("content_key", key.alias("_key"))
            .bottom_k(limit, by=["_key", "content_key"])
        )
        .collect(engine="streaming")
    )
    return (
        index.compact_strings(
            rows.filter(pl.col("content_key").is_in(winners["content_key"].implode()))
        )
        .collect(engine="streaming")
        .join(winners, on="content_key")
        .sort("_key", "content_key")
        .drop("_key")
        .to_dicts()
    )

