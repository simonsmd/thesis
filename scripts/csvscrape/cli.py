"""Command line entry point: python -m csvscrape <command>.

Each subcommand registers its handler with `set_defaults(run=...)`, so dispatch
is one call rather than a chain of comparisons against `args.command`.
Handlers that touch the network get a live progress reporter from
`_with_reporter`, which owns creating and closing it in one place.
"""
from __future__ import annotations

import argparse
import signal
from typing import Any, Callable

from . import config


# The draw shapes, spelled out rather than read off `sample.MODES`: building
# the parser must not import Polars, which every non-Polars subcommand and
# `--help` would otherwise pay for. `sample` is the authority on what they
# mean, and a test asserts the two lists stay equal.
SAMPLING_MODES = ("log-uniform", "uniform")

# The pools a corpus can be drawn from, and the per-dataset file cap Kaggle
# needs. Both are spelled out here for the same reason as SAMPLING_MODES --
# `kaggle` and `index` are the authorities, and tests assert these stay equal.
SOURCES = ("github", "kaggle")
KAGGLE_PER_DATASET = 250
KAGGLE_RECORD_MEMBERS = 5_000


def _with_reporter(quiet: bool, body: Callable[[Any], dict]) -> dict:
    """Run `body` with a reporter, closing it however the body exits."""
    from .report import Reporter

    reporter = None if quiet else Reporter()
    try:
        return body(reporter)
    finally:
        if reporter:
            reporter.close()


def _github(reporter: Any):
    from .client import GitHub

    return GitHub(on_event=reporter or (lambda *a, **k: None))


def _print_stats(stats: dict) -> int:
    print(f"data root: {config.DATA_ROOT}")
    for key, value in stats.items():
        print(f"  {key:12s} {value}")
    return 0


# -- handlers --------------------------------------------------------------


def _cmd_discover(args) -> int:
    from . import discover

    stats = _with_reporter(args.quiet, lambda rep: discover.run(
        _github(rep), max_shards=args.shards, max_pages=args.pages,
        min_new_per_page=args.min_new, on_event=rep))
    _print_stats(stats)
    if stats.get("interrupted"):
        # Same contract as the other passes: 130 is the shell's convention for
        # "killed by SIGINT", so a driving script can tell this from a pass
        # that finished.
        print("  interrupted -- the shard resumes at its next page; re-run to continue")
        return 130
    return 0


def _cmd_trees(args) -> int:
    from . import trees

    stats = _with_reporter(args.quiet, lambda rep: trees.run(
        _github(rep), limit=args.limit, workers=args.workers, tier=args.tier,
        ext=args.ext,
        order=args.order, seed=args.seed, on_event=rep))
    _print_stats(stats)
    if stats.get("interrupted"):
        print("  interrupted -- scanned repos are recorded; re-run to continue")
        return 130
    return 0


# Numbers a pass reports that are *not* counters, and how several passes of
# them combine. Everything absent from here is a count and adds up; summing a
# gauge would report a much worse run than happened.
_GAUGES = {
    "peak_rss_mb": max,
    "raw_rate_min": min,
    "raw_rate_max": max,
    "raw_retry_after": max,
    # The rate the controller settled at, and the last pass is the one that
    # settled it. Same for anything the server only ever states once.
    "raw_rate_final": lambda _old, new: new,
    "raw_rate_wall": max,
    # The metered Kaggle pool states its budget only in a refusal, so the
    # longest one it asked for is the reading -- same shape as raw's.
    "kaggle_retry_after": max,
}


def _fold(key: str, total: dict, value):
    """One pass's `value` merged into the running `total` for `key`."""
    if key not in total:
        return value
    old = total[key]
    if key in _GAUGES:
        # A gauge only combines with another reading of itself; a pass that
        # never set one reports None, which is not a number to take a max of.
        if old is None or value is None:
            return value if old is None else old
        return _GAUGES[key](old, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return old + value
    return value


def _cmd_probe(args) -> int:
    from . import probe

    def body(rep) -> dict:
        # Without --limit, passes of PROBE_DRAW until there is nothing left to
        # draw. Every pass seals its own heads shards on exit, which is how the
        # next draw knows what the previous one did. A limited run is one pass.
        #
        # The run stops on an empty draw (a short draw is not exhaustion:
        # `--order repo` draws whole repositories), or on a pass that settled
        # nothing. A transient failure does not retire its key, so without the
        # second condition files that keep failing would be redrawn forever.
        #
        # `--order repo` resumes on a cursor instead, with two consequences:
        #
        #   * the seed is fixed rather than advanced per pass. The other orders
        #     re-thin the pool by a hash residue, so reusing the seed would
        #     redraw the same rows; the cursor is a position in one total
        #     order, which moving the seed would scramble.
        #   * a pass that settled nothing still advances the cursor, so only a
        #     cursor that does not move counts as a stall. Files skipped that
        #     way come back on the next *run*, which starts from the log.
        total: dict = {}
        passes = 0
        cursor = None
        resumable = args.order == "repo"
        # Counted once per invocation, not per pass, so the progress line can
        # show a position in the whole pool.
        pool_total = probe.pool_size()
        # Carried across passes for the progress line. `retired` is where the
        # *pool* stands, so it opens at what is already settled; `settled` and
        # `errors` are what this invocation has done, so they open at zero.
        carried = {"retired": pool_total - probe.outstanding_size(),
                   "settled": 0, "errors": 0}
        # One client for the whole run, not one per pass: `RawPool` learns the
        # raw rate limit as it goes, and the request counters and keep-alive
        # connections live on it too.
        gh = _github(rep)
        while True:
            stats = probe.run(
                gh, limit=args.limit or config.PROBE_DRAW,
                workers=args.workers, raw_workers=args.raw_workers,
                cache=args.cache, cache_dir=args.cache_dir,
                head_bytes=args.head_bytes if args.head_bytes is not None
                           else config.HEAD_BYTES,
                which=tuple(x.strip() for x in args.sniffers.split(",")
                            if x.strip()),
                seed=args.seed if resumable else args.seed + passes,
                sniff_procs=args.sniff_procs, after=cursor,
                pool_total=pool_total, carried=dict(carried),
                order=args.order, on_event=rep)
            passes += 1
            # Whether the cursor moved, not merely whether there is one.
            advanced = resumable and stats.get("cursor") not in (None, cursor)
            cursor = stats.get("cursor") if resumable else None
            carried["retired"] += stats.get("retired", 0)
            carried["settled"] += stats.get("retired", 0)
            carried["errors"] += stats.get("errors", 0)
            for key, value in stats.items():
                total[key] = _fold(key, total, value)
            if (args.limit or stats.get("interrupted") or stats.get("abandoned")
                    or not stats.get("queued", 0)
                    or (not advanced and not stats.get("retired", 0))):
                break
        total["passes"] = passes
        return total

    stats = _with_reporter(args.quiet, body)
    for key, value in stats.items():
        print(f"  {key:12s} {value}")
    if stats.get("interrupted"):
        print("  interrupted -- probed files are recorded; re-run to continue")
        return 130
    return 0


def _cmd_download(args) -> int:
    from . import download

    # The default dataset name is derived from the flags that define a corpus
    # (source included), so two different draws never read each other's log
    # as their own progress.
    dataset = args.dataset or (
        ("" if args.source == "github" else f"{args.source}-")
        + f"{args.select}-{args.sampling}" + ("-utf8" if args.utf8 else ""))
    def _client(rep):
        if args.source == "kaggle":
            from .kaggle import Kaggle

            return Kaggle(on_event=rep or (lambda *a, **k: None))
        return _github(rep)

    stats = _with_reporter(args.quiet, lambda rep: download.run(
        _client(rep), n=args.count, seed=args.seed, workers=args.workers,
        per_repo=args.per_repo, per_owner=args.per_owner,
        select=args.select, utf8=args.utf8, mode=args.sampling, dataset=dataset,
        max_bucket=args.max_bucket, sniff=not args.no_sniff,
        sniffers_which=tuple(x.strip() for x in args.sniffers.split(',') if x.strip()),
        sniff_procs=args.sniff_procs, source=args.source, on_event=rep))
    print(download.summary(stats))
    if stats.get("interrupted"):
        print("  interrupted -- downloaded files are kept; re-run to continue")
        return 130
    return 0


def _cmd_compact(args) -> int:
    from . import index

    print(f"wrote {index.build()}")
    return 0


def _cmd_status(args) -> int:
    from . import index

    print(index.render())
    return 0


def _cmd_calibrate(args) -> int:
    from . import index

    print(index.calibrate(args.target))
    return 0


def _cmd_selectors(args) -> int:
    from . import selectors

    print(selectors.describe_all())
    return 0


def _cmd_datasets(args) -> int:
    """What has been drawn so far, per dataset."""
    import json

    from . import download, index

    names = download.datasets()
    if not names:
        print("no datasets yet: run `csvscrape download`")
        return 0
    for name in names:
        path = download.manifest_path(name)
        if not path.exists():
            download._write_manifest(name)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        total = sum(r.get("size", 0) for r in rows)
        buckets = {r["bucket"] for r in rows}
        span = f"buckets {min(buckets)}-{max(buckets)}" if buckets else "empty"
        print(f"  {name:<24} {len(rows):>7,} files · {index.human(total):>10} · {span}")
        print(f"  {'':<24} {path}")
    return 0


def _cmd_dialects(args) -> int:
    from . import probe

    print(probe.report())
    return 0


def _cmd_kaggle_dialects(args) -> int:
    from . import kaggle, probe

    print(probe.report(kaggle.HEADS_LOG))
    return 0


def _cmd_kaggle_compact(args) -> int:
    from . import index

    print(f"wrote {index.build_kaggle()}")
    return 0


def _cmd_kaggle_status(args) -> int:
    from . import index

    print(index.render_kaggle())
    return 0


def _cmd_kaggle_discover(args) -> int:
    from . import kaggle

    def body(rep):
        kg = kaggle.Kaggle(on_event=rep or (lambda *a, **k: None))
        return kaggle.discover(kg, max_shards=args.shards, max_pages=args.pages,
                               recheck=args.recheck, on_event=rep)

    return _print_stats(_with_reporter(args.quiet, body))


def _cmd_kaggle_bundle(args) -> int:
    from . import kaggle

    def body(rep):
        kg = kaggle.Kaggle(on_event=rep or (lambda *a, **k: None))
        return kaggle.bundle(
            kg, limit=args.limit,
            routes=tuple(x.strip() for x in args.routes.split(",") if x.strip()),
            head_workers=args.head_workers, order=args.order,
            head_bytes=args.head_bytes if args.head_bytes is not None
                       else config.HEAD_BYTES,
            cache=args.cache, cache_dir=args.cache_dir,
            which=tuple(x.strip() for x in args.sniffers.split(",") if x.strip()),
            sniff_procs=args.sniff_procs, cap=args.per_dataset,
            record=args.record_members, on_event=rep)

    stats = _with_reporter(args.quiet, body)
    _print_stats(stats)
    if stats.get("interrupted"):
        # A dataset interrupted part-way has no bundle record, so the next run
        # redoes it whole.
        print("  interrupted -- bundled datasets are recorded; re-run to continue")
        return 130
    return 0


# -- parser ----------------------------------------------------------------


def _positive(value: str) -> int:
    """argparse type for a thread count that must be at least one: each
    stage of a pass waits for the stage before it to drain, so a stage with no
    threads never finishes."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csvscrape")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress live progress on stderr")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("discover", help="code search -> repository names")
    p.set_defaults(run=_cmd_discover)
    p.add_argument("--shards", type=int, default=None, help="stop after N shards")
    p.add_argument("--pages", type=int, default=10, help="max pages per shard (cap is 10)")
    p.add_argument("--min-new", type=int, default=0,
                   help="retire a shard permanently once a page yields fewer "
                        "than N new repos (0 = never). Counts novelty against "
                        "every shard already scanned, so on a broad sweep it "
                        "measures overlap rather than exhaustion; use --pages "
                        "to bound cost instead, which stays resumable")

    p = sub.add_parser("trees", help="enumerate files in discovered repos")
    p.set_defaults(run=_cmd_trees)
    p.add_argument("--limit", type=int, default=None, help="stop after N repos")
    p.add_argument("--workers", type=int, default=3,
                   help="concurrent repo scans (should saturate api budget)")
    p.add_argument("--tier", choices=["lfs", "hi", "mid", "low", "tiny", "micro"],
                   help="scan only this discovery size tier")
    p.add_argument("--ext", choices=["csv", "tsv", "psv"],
                   help="scan only repos discovered through this extension")
    p.add_argument("--order", choices=["interleave", "priority", "random"],
                   default="interleave",
                   help="queue order: interleave = every tier represented, "
                        "priority = biggest files first, random = uniform")
    p.add_argument("--seed", type=int, default=0, help="seed for --order random")

    p = sub.add_parser("compact", help="rebuild the Parquet index from the logs")
    p.set_defaults(run=_cmd_compact)

    p = sub.add_parser("status", help="per-bucket fill level")
    p.set_defaults(run=_cmd_status)

    p = sub.add_parser("download", help="sample N files log-uniformly and fetch them")
    p.set_defaults(run=_cmd_download)
    p.add_argument("-n", "--count", type=int, default=1000,
                   help="number of usable files wanted in the dataset")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--per-repo", type=int, default=2,
                   help="soft cap on files taken from one repo per bucket")
    p.add_argument("--per-owner", type=int, default=5, help="soft cap per owner")
    p.add_argument("--select", default="main",
                   help="named selector deciding which slice of the pool to draw "
                        "from (default: main, which excludes only what is "
                        "plainly not text); see `csvscrape selectors`")
    p.add_argument("--utf8", action="store_true",
                   help="keep only files that decode as UTF-8 end to end. "
                        "Composes with any selector")
    p.add_argument("--sampling", choices=list(SAMPLING_MODES),
                   default="log-uniform",
                   help="shape of the draw. log-uniform or uniform")
    p.add_argument("--dataset", default=None,
                   help="name of the dataset (default: <select>-<sampling>)")
    p.add_argument("--sniffers",
                   default="custom,clevercsv,duckdb,chardet",
                   help="sniffers to run over the downloaded files. Their "
                        "verdicts are normalised into the parser's own Config "
                        "vocabulary and written to datasets/<name>.json")
    p.add_argument("--sniff-procs", type=int, default=0,
                   help="processes running the CPU-bound sniffers over the "
                        f"corpus (default {config.SNIFF_PROCESSES}). 0 uses "
                        "that default; 1 keeps a crashing sniffer in-process "
                        "where it leaves a usable traceback")
    p.add_argument("--no-sniff", action="store_true",
                   help="fetch only; don't modify datasets/<name>.json")
    p.add_argument("--source", choices=list(SOURCES), default="github",
                   help="which pool to draw from")
    p.add_argument("--max-bucket", type=int, default=None,
                   help="skip buckets above this power of two: "
                        "--max-bucket 24 caps files at 32 MiB")

    p = sub.add_parser("selectors", help="list available corpus selectors")
    p.set_defaults(run=_cmd_selectors)

    p = sub.add_parser("datasets", help="list datasets")
    p.set_defaults(run=_cmd_datasets)

    p = sub.add_parser("probe", help="range-fetch file heads for metadata")
    p.set_defaults(run=_cmd_probe)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=_positive, default=config.GRAPHQL_MAX_INFLIGHT,
                   help="GraphQL batch readers, which is where nearly every "
                   "file's head comes from. Default is"
                        f"GRAPHQL_MAX_INFLIGHT ({config.GRAPHQL_MAX_INFLIGHT}). "
                        "Raising it does not raise throughput: a batch costs "
                        "one rate-limit point whatever it carries, so the pool "
                        "is paced to 4,750 an hour and only "
                        "GRAPHQL_BATCH_BYTES increases throughput")
    p.add_argument("--raw-workers", type=_positive, default=config.RAW_MAX_INFLIGHT,
                   help="threads on the range-request path, which serves LFS "
                        "objects, blobs GraphQL will not return, and anything "
                        f"over GRAPHQL_BLOB_MAX (default "
                        f"{config.RAW_MAX_INFLIGHT}). Metered per IP at "
                        "~113,000 requests an hour")
    p.add_argument("--sniff-procs", type=int, default=config.SNIFF_PROCESSES,
                   help="processes running the CPU-bound sniffers "
                        f"(default {config.SNIFF_PROCESSES}). 0 sniffs in the "
                        "fetch threads instead: several times slower, but it "
                        "keeps a crashing sniffer in-process where it leaves a "
                        "usable traceback")
    p.add_argument("--order", choices=["uniform", "spread", "repo"],
                   default="uniform",
                   help="sampling frame: uniform = every unprobed file equally "
                        "likely, so statistics describe the pool (default) "
                        "spread = equal counts per size octave"
                        "repo = whole repositories at a time, several times "
                        "faster per file because GraphQL is charged per "
                        "repository resolution (cheaper for full scans)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the sample order")
    p.add_argument("--head-bytes", type=int, default=None,
                   help=f"bytes to read from the start (default {config.HEAD_BYTES:,})")
    p.add_argument("--cache", action="store_true",
                   help="keep the fetched bytes so metadata can be recomputed offline")
    p.add_argument("--sniffers", default="custom,clevercsv,duckdb,charset_normalizer,chardet",
                   help="comma-separated sniffers to run "
                        "(custom,clevercsv,duckdb,charset_normalizer,chardet)")
    p.add_argument("--cache-dir", default=None,
                   help="where cached bytes go (or CSVSCRAPE_CACHE); may be a "
                        "different filesystem from the data root")

    p = sub.add_parser("dialects",
                       help="dialect breakdown of probed github files")
    p.set_defaults(run=_cmd_dialects)

    p = sub.add_parser("calibrate", help="project a small scan to full-run cost")
    p.set_defaults(run=_cmd_calibrate)
    p.add_argument("--target", type=int, default=250,
                   help="unique files wanted per bucket")

    p = sub.add_parser("kaggle-discover", help="enumerate kaggle datasets")
    p.set_defaults(run=_cmd_kaggle_discover)
    p.add_argument("--shards", type=int, default=None,
                   help="stop after N shards (a split counts as one)")
    p.add_argument("--pages", type=int, default=500,
                   help="max pages per shard (the API's own cap is 500). "
                        "Below 500 the capped-shard test is skipped, so "
                        "subdivision stops: use the default for a full sweep")
    p.add_argument("--recheck", action="store_true",
                   help="make every finished shard prove it is finished. Each "
                        "resumes at the page after its last and stops only on "
                        "an empty one")

    p = sub.add_parser("kaggle-bundle",
                       help="read each dataset's archive: file list + head probe")
    p.set_defaults(run=_cmd_kaggle_bundle)
    p.add_argument("--limit", type=int, default=None, help="stop after N datasets")
    p.add_argument("--routes", default="rest,rpc",
                   help="comma-separated routes that sign bundle URLs, one "
                        "worker each: rest (/datasets/download) and rpc (the "
                        "SDK's DatasetApiService). Each is metered on its own "
                        "budget of ~1,000 an hour, so dropping one halves the "
                        "pass's throughput")
    p.add_argument("--head-workers", type=_positive, default=16,
                   help="threads reading members out of the archives and "
                        "sniffing them. These spend no metered budget")
    p.add_argument("--order", choices=["interleave", "priority", "random"],
                   default="interleave",
                   help="queue order: interleave = every size decade "
                        "represented, so an interrupted run spans the spectrum")
    p.add_argument("--per-dataset", type=int, default=KAGGLE_PER_DATASET,
                   help="files head-probed per dataset. The pool is extremely "
                        "top-heavy, so this is what bounds the pass and keeps "
                        "pool statistics from describing a handful of "
                        "publishers. Raising it "
                        "later reaches the bundled datasets too, up to "
                        "--record-members: each one still owing a head is "
                        "signed once more and read at its recorded offsets, "
                        "and nothing already described is fetched again")
    p.add_argument("--record-members", type=int, default=KAGGLE_RECORD_MEMBERS,
                   help="members whose coordinates reach the log and the "
                        "index, probed or not. Recording more than is probed "
                        "is what makes --per-dataset revisitable: the rows are "
                        "already indexed, so raising it adds head fetches "
                        "instead of a second pass over every archive")
    p.add_argument("--head-bytes", type=int, default=None,
                   help=f"bytes to read from the start (default {config.HEAD_BYTES:,})")
    p.add_argument("--cache", action="store_true",
                   help="keep the fetched heads so metadata can be recomputed "
                        "offline")
    p.add_argument("--cache-dir", default=None,
                   help="where cached bytes go (or CSVSCRAPE_CACHE)")
    p.add_argument("--sniffers", default="custom,clevercsv,duckdb,charset_normalizer,chardet",
                   help="comma-separated sniffers to run over each head")
    p.add_argument("--sniff-procs", type=int, default=config.SNIFF_PROCESSES,
                   help=f"processes running the CPU-bound sniffers "
                        f"(default {config.SNIFF_PROCESSES}); 0 sniffs in the "
                        f"fetch threads")

    p = sub.add_parser("kaggle-dialects",
                       help="dialect breakdown of head-probed kaggle files")
    p.set_defaults(run=_cmd_kaggle_dialects)

    p = sub.add_parser("kaggle-compact", help="rebuild the kaggle parquet index")
    p.set_defaults(run=_cmd_kaggle_compact)

    p = sub.add_parser("kaggle-status", help="per-bucket fill level for kaggle")
    p.set_defaults(run=_cmd_kaggle_status)

    return parser


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # SIGTERM is handled as a Ctrl-C, so `timeout`, `kill`, systemd and a
    # closed terminal reach the same wind-down every pass already has for
    # Ctrl-C. Its default action exits without unwinding: the open log shard is
    # not sealed, the records since its last zstd frame are lost, and no
    # summary is printed.
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        return args.run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
