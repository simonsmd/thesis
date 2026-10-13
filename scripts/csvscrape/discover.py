"""Phase 1: code search, used purely to discover repositories.

Search never contributes a file to the corpus. Its results feed the tree
scanner, so that a file's inclusion probability depends only on whether its
*repo* was discovered -- one bias, at the repo level -- rather than stacking
repo-level bias on top of the relevance ranking within each size class.

Size qualifiers are therefore just a device for partitioning around the
hard 1000-result cap, not a way of selecting files. Two deliberate skews, both
free:

  * shards are ordered largest-size-first, because a repo holding a 300 KB CSV
    is far likelier to hold a 30 MB one, and the big buckets are the ones that
    are hard to fill;
  * .gitattributes shards find LFS repos through code search, reaching repos
    whose CSVs are all far past the 350 KiB index ceiling.

Known coverage holes. Each one costs a repo only when *every* delimited file in
it falls in the hole, since one reachable file pulls the whole tree in:

  * files over 350 KiB are not in the search index at all, so a repo whose CSVs
    are all that large and not LFS-tracked cannot be reached from here;
  * `extension:csv` matches on the final extension, so `data.csv.gz` is a `gz`
    file to search. `config.is_kept` accepts the compression suffixes and
    `trees` harvests them, so a repo publishing only compressed CSVs is
    harvestable but not discoverable;
  * 0-byte files are not in the search index, so a repo holding only empty
    CSVs is unreachable. Empty files found by `trees` are still indexed;
  * the size tiers are half-open and chain -- `tiny` ends at 1,023 where `low`
    starts at 1,024 -- but nothing sits above `hi`, so a file of exactly
    358,400 bytes is uncovered. A query string is a shard's identity in the
    log, so widening it would re-run every csv shard to gain a handful of
    files.

GitHub's index is also narrower than it looks: it excludes vendored and
generated code, non-UTF-8 files, and any file with more than one line over
4,096 bytes. That last one selects against exactly the wide CSVs this corpus
wants, which is a further argument for reaching repos through .gitattributes
and Kaggle rather than through content search alone.
"""
from __future__ import annotations

import itertools
import time
from typing import Any, Iterator

from . import config, log
from .util import interleave
from .client import GitHub

# One query per shard. Ordered by descending expected large-file yield.
#
# Every shard is far over the 1000-result cap, so a tier's step only decides
# how many shards it gets, i.e. what share of the tier the schedule samples.
# `micro`'s step of 3 gives it roughly the same sampling rate as `tiny`.
# 0 bytes is out because `size:0..0` returns nothing.
TIERS: list[tuple[str, int, int, int]] = [
    # (tier, lo, hi, step) -- byte ranges within the searchable window
    ("hi", 262_144, config.CODE_SEARCH_MAX_BYTES, 256),
    ("mid", 65_536, 262_144, 512),
    ("low", 1_024, 65_536, 256),
    ("tiny", 64, 1_024, 8),
    ("micro", 1, 64, 3),
]

# `path:.gitattributes` matches nothing here; `filename:` is the qualifier that
# works for this.
#
# Sharded on the size of the .gitattributes file itself, because one query
# returns at most 1000 hits. These are the repos holding files above the
# search ceiling -- the scarce top buckets `trees.SCAN_WEIGHTS` prioritises.
#
# The edges are cut from a histogram of hits per .gitattributes size, aiming
# at ~900 hits per shard rather than the cap: `total_count` is only an
# estimate, and a shard sitting on the cap loses the difference silently. A
# single byte length is the finest a size shard can get, so a shard that ever
# passes 1000 hits on one length cannot be split further. Re-cut the edges
# from a fresh histogram if the population moves.
LFS_BASE = '"filter=lfs" csv filename:.gitattributes'
LFS_EDGES = (0, 44, 48, 56, 64, 76, 85, 96, 112, 124, 128, 136, 160, 168,
             176, 208, 224, 256, 288, 320, 384, 512, 768, 1_024, 1_800,
             2_560, 4_096, config.CODE_SEARCH_MAX_BYTES + 1)

# The same for TSV- and PSV-tracked repos, which are far smaller populations
# (roughly a thousand and ten hits). psv is the scarcest extension in the
# index, and this is the only route to a psv file above the search ceiling.
LFS_TSV_BASE = '"filter=lfs" tsv filename:.gitattributes'
LFS_TSV_EDGES = (0, 256, 1_024, config.CODE_SEARCH_MAX_BYTES + 1)

LFS_PSV_BASE = '"filter=lfs" psv filename:.gitattributes'
LFS_PSV_EDGES = (0, config.CODE_SEARCH_MAX_BYTES + 1)


def _lfs_shards() -> Iterator[tuple[str, str, str, int]]:
    for ext, base, edges in (("csv", LFS_BASE, LFS_EDGES),
                             ("tsv", LFS_TSV_BASE, LFS_TSV_EDGES),
                             ("psv", LFS_PSV_BASE, LFS_PSV_EDGES)):
        for lo, hi in zip(edges, edges[1:]):
            yield ("lfs", ext, f"{base} size:{lo}..{hi - 1}",
                   config.CODE_SEARCH_MAX_BYTES)


# Hits per size tier, as `extension:<ext> size:<lo>..<hi>` total_count. This
# table is the sizing input for `_scaled_tiers`: the tsv and psv shard counts
# are derived from it rather than hand-cut, so re-measuring these numbers
# resizes the schedule. GitHub quantises total_count at this magnitude, so they
# are estimates good to a percent or two, which is all a shard budget needs.
SEARCH_TOTALS = {
    "csv": {"hi": 2_064_384, "mid": 13_205_504, "low": 57_409_536,
            "tiny": 28_770_304, "micro": 5_472_256},
    "tsv": {"hi":    60_544, "mid":    516_096, "low":  2_715_648,
            "tiny":  2_052_096, "micro":   180_736},
    "psv": {"hi":       104, "mid":        884, "low":    870_400,
            "tiny":     46_848, "micro":       426},
}


def _tier_shards(tier: str, lo: int, hi: int, step: int,
                 ext: str = "csv") -> Iterator[tuple[str, str, str, int]]:
    """Tile [lo, hi) downwards in `step`-wide shards, largest range first.

    Walking down from the top rather than up from `lo` puts the ragged shard --
    the narrow one left over when `step` does not divide the tier width -- at
    the *bottom*, where a shard covering fewer bytes costs least.

    Chaining `end = start - 1` tiles exactly, with no overlap or gap whatever
    the step. Byte `hi` itself is left to the next tier up (or uncovered, for
    `hi`; see the module docstring). The query strings are a shard's identity
    in the log, so changing how they are generated re-runs every shard.
    """
    end = hi - 1
    while end >= lo:
        start = max(lo, end - step + 1)
        yield (tier, ext, f"extension:{ext} size:{start}..{end}", (start + end) // 2)
        end = start - 1


def _scaled_tiers(ext: str) -> Iterator[tuple[str, int, int, int]]:
    """csv's tier layout, re-stepped for a rarer extension.

    A shard is worth at most 1000 repos whatever it covers, so shard count is
    purely a budget allocation. `TIERS` is deliberately not a uniform sample --
    it samples the scarce large-file tiers far more densely than the small
    ones -- and tsv and psv are not shaped like csv, so one global ratio would
    get both ends wrong. Instead each tier matches csv's *sampling rate within
    that tier*:

        n(ext, tier) = n(csv, tier) x total(ext, tier) / total(csv, tier)

    Two clamps. A tier never gets zero shards, because a tier missing from the
    schedule is a hole in the size spectrum rather than a saving; and it never
    gets more shards than its population can fill.
    """
    for tier, lo, hi, step in TIERS:
        width = hi - lo
        csv_n = -(-width // step)
        share = SEARCH_TOTALS[ext][tier] / SEARCH_TOTALS["csv"][tier]
        fillable = -(-SEARCH_TOTALS[ext][tier] // config.CODE_SEARCH_RESULT_CAP)
        n = min(max(1, round(csv_n * share)), max(1, fillable))
        yield tier, lo, hi, -(-width // n)


def _ext_shards(ext: str) -> Iterator[tuple[str, str, str, int]]:
    """Shards for a non-csv extension, largest tier first."""
    for tier, lo, hi, step in _scaled_tiers(ext):
        yield from _tier_shards(tier, lo, hi, step, ext)


# `.tab` is deliberately not searched; see config.INDEX_EXTENSIONS.


# A shard records two independent facts. `tier` is the *size stratum* (lfs,
# hi, mid, low, tiny, micro) and is a claim about the repo: one reachable
# through a 300 KB file publishes data, one reachable only through a 90-byte
# file is likelier holding a test fixture. `index.calibrate` groups by it to
# answer "which kind of repo fills the buckets I am short of". `ext` is the
# extension the shard searched, and says nothing about size.

# Shards drawn from each tier per round of the interleave. Weighted towards the
# scarce large files, but every tier appears in every round so that *any*
# prefix of the schedule spans the size spectrum. `tiny` is weighted up to keep
# discovery ahead of `trees.SCAN_WEIGHTS`, which draws heavily from it.
#
# `tiny` supply is finite: its shards can yield at most 1000 repos each, so if
# the scan exhausts it, halving its step in `TIERS` doubles the ceiling.
TIER_WEIGHTS = {"hi": 4, "mid": 3, "low": 2, "tiny": 4, "micro": 2}


def shards() -> Iterator[tuple[str, str, str, int]]:
    """Yield (tier, ext, query, size_hint), interleaved across tiers.

    Discovery only ever returns repo *names*; tree enumeration then harvests
    every CSV in a repo regardless of size. So a shard's size range does not
    select file sizes -- it selects a population of repos. A repo holding a
    350 KB CSV is a data-publishing repo; one holding a 90 byte CSV is more
    likely a test fixture or a config table, and those populations carry
    different dialects.

    Weighted round-robin across the csv tiers keeps the large-file bias while
    making every prefix of the schedule diverse; concatenating them would mean
    a partial run drew only from `hi`.

    The lfs, tsv and psv schedules are emitted in full first: they are small
    and irreplaceable, and a run stopped early should not have only a fraction
    of them. Within each, the largest size tier still comes first.
    """
    yield from _lfs_shards()
    yield from _ext_shards("tsv")
    yield from _ext_shards("psv")
    yield from interleave(
        {tier: _tier_shards(tier, lo, hi, step) for tier, lo, hi, step in TIERS},
        TIER_WEIGHTS)


def completed() -> set[str]:
    """Shards that stopped for an intrinsic reason and will never yield more."""
    return {r["query"] for r in log.read("shards") if r.get("state") == "done"}


def progress() -> dict[str, int]:
    """Pages already fetched for shards that stopped early on a page budget.

    A shard cut short by `--pages` is not finished -- it just ran out of
    allowance -- so it is recorded as `partial` and resumes at the next page
    rather than being skipped like a completed one.
    """
    retired = completed()
    furthest: dict[str, int] = {}
    for rec in log.read("shards"):
        # `error` counts as progress too. A shard that failed on page 5 already
        # harvested pages 1-4, and a retry from page 1 would re-read them.
        if rec.get("state") not in ("partial", "error") or rec["query"] in retired:
            continue
        furthest[rec["query"]] = max(furthest.get(rec["query"], 0), rec.get("pages", 0))
    return furthest


def known_repos() -> set[int]:
    return set(log.scan_field("discovery", "repo_id"))


def run(
    gh: GitHub,
    max_shards: int | None = None,
    max_pages: int = 10,
    min_new_per_page: int = 0,
    on_event: Any = None,
) -> dict[str, int]:
    """Walk shards until the budget is spent, appending discovered repos.

    `min_new_per_page` stops paging a shard once a page yields fewer than that
    many previously unseen repos. It defaults to 0 (off) because it is sharper
    than it looks: `seen` is shared across the run, so the count measures
    overlap with every shard scanned *before* this one rather than this
    shard's own depth, and the resulting `low_yield` retirement is permanent.
    Bounding cost with `--pages` is the safe lever: it records `partial`,
    which resumes.
    """
    emit = on_event or (lambda *a, **k: None)
    done_queries = completed()
    resume_from = progress()
    seen = known_repos()
    emit("start", retired=len(done_queries), partial=len(resume_from), known=len(seen))
    stats = {"shards": 0, "partial": 0, "failed": 0, "requests": 0,
             "hits": 0, "new_repos": 0}
    interrupted = False

    with log.Writer("discovery") as repos_log, log.Writer("shards", rotate=500) as shard_log:
        # A shard is only pending if it is not retired *and* this run's page
        # budget can actually advance it. Without the second condition a shard
        # already at page N would consume a --shards slot, issue no requests
        # and rewrite its own record -- so `--shards 3` would do nothing at all
        # rather than moving on to the next three actionable shards.
        pending = (
            s for s in shards()
            if s[2] not in done_queries and resume_from.get(s[2], 0) < max_pages
        )
        for index, (tier, ext, query, size_hint) in enumerate(
            itertools.islice(pending, max_shards), start=1
        ):
            emit("shard_start", index=index, of=max_shards, tier=tier, ext=ext,
                 query=query, resume=resume_from.get(query, 0))
            total = None
            start = resume_from.get(query, 0) + 1
            pages = start - 1
            failed = False
            # Why the shard stopped. Only an intrinsic reason means "done";
            # running out of page budget must stay resumable.
            reason = "page_budget"
            # Ctrl-C is a resumable reason like any other: recording `pages`
            # for the in-flight shard lets a restart resume at the right page
            # instead of re-reading pages whose repos are already known.
            try:
                for page in range(start, max_pages + 1):
                    result = gh.search_code(query, page=page)
                    stats["requests"] += 1
                    if "error" in result:
                        failed = True
                        stats["failed"] += 1
                        emit("shard_error", index=index, query=query,
                             message=result.get("message", ""))
                        # Same ordering invariant as a clean finish: the repos this
                        # shard already found must be durable before `pages` claims
                        # they were harvested.
                        repos_log.flush()
                        shard_log.write(
                            {"query": query, "tier": tier, "ext": ext, "state": "error",
                             "pages": pages, "http": result.get("error"),
                             "message": result.get("message", ""), "ts": int(time.time())}
                        )
                        break
                    total = result.get("total_count", 0)
                    items = result.get("items", [])
                    # A backend timeout returns a short (possibly empty) page with
                    # this flag set, which would otherwise look like a natural end
                    # and retire the shard. It is transient, so resumable.
                    if result.get("incomplete_results"):
                        reason = "incomplete"
                        break
                    if not items:
                        reason = "exhausted"
                        break

                    new_here = 0
                    for item in items:
                        repo = item["repository"]
                        stats["hits"] += 1
                        if repo["id"] in seen:
                            continue
                        seen.add(repo["id"])
                        new_here += 1
                        stats["new_repos"] += 1
                        repos_log.write(
                            {
                                "repo_id": repo["id"],
                                "full_name": repo["full_name"],
                                "path": item["path"],
                                "blob_sha": item["sha"],
                                "tier": tier,
                                "ext": ext,
                                "shard": query,
                                "size_hint": size_hint,
                                "ts": int(time.time()),
                            }
                        )
                    pages = page
                    emit("page", index=index, page=page, hits=len(items), new=new_here,
                         total=stats["new_repos"])
                    # The cap is hard: page 11 returns 422 regardless of total_count.
                    if len(items) < 100:
                        reason = "exhausted"
                        break
                    if page * 100 >= config.CODE_SEARCH_RESULT_CAP:
                        reason = "result_cap"
                        break
                    # Off by default: see the note above on why a thin page is
                    # weak evidence and an expensive thing to act on.
                    if min_new_per_page and new_here < min_new_per_page:
                        reason = "low_yield"
                        break

            except KeyboardInterrupt:
                interrupted = True
                reason = "interrupted"

            # Only a shard that actually completed is recorded as done. Marking
            # a failed shard done would silently drop its slice of the size
            # spectrum from the corpus forever, since completed() skips it.
            if failed:
                # An interrupt landing on an already-failed shard still ends
                # the run: the error branch has written this shard's record, so
                # there is nothing further to record for it.
                if interrupted:
                    break
                continue
            # Ordering invariant: the repos must be on disk before the shard is
            # recorded as done. The two logs have independent buffers, so
            # without this an unclean exit can persist the `done` marker while
            # this shard's last repos are still buffered, and completed() would
            # skip the shard forever. (A buffer flush, not an fsync.)
            repos_log.flush()
            # `partial` keeps the shard in the queue so a later run continues
            # from where this one stopped; `done` retires it. Only an answer
            # about the shard itself (`exhausted`, `result_cap`, `low_yield`) is
            # a reason to stop offering it.
            RESUMABLE = ("page_budget", "incomplete", "interrupted")
            state = "partial" if reason in RESUMABLE else "done"
            shard_log.write(
                {"query": query, "tier": tier, "ext": ext, "state": state,
                 "reason": reason, "pages": pages, "total_count": total,
                 "ts": int(time.time())}
            )
            stats["shards" if state == "done" else "partial"] += 1
            emit("shard_end", index=index, tier=tier, ext=ext, query=query, state=state,
                 reason=reason, pages=pages, total_count=total,
                 total=stats["new_repos"])
            if interrupted:
                break
    if interrupted:
        stats["interrupted"] = True
    # How many of `requests` the server refused. Guarded because a caller may
    # hand in a double that only answers `search_code`.
    if hasattr(getattr(gh, "search_pacer", None), "stats"):
        stats |= gh.search_pacer.stats
    return stats
