"""Tunables and paths for the corpus scraper."""
from __future__ import annotations

import math
import os
from pathlib import Path

# Root of the data lake. Everything the scraper produces lives here.
DATA_ROOT = Path(os.environ.get("CSVSCRAPE_DATA", "data")).resolve()

# Two-level filtering. KEEP_EXTENSIONS decides what reaches the tree logs and is
# deliberately wider than INDEX_EXTENSIONS, so widening the index later is a
# `compact` rather than a rescan. On real trees .csv is ~2% of all blobs and
# .txt roughly doubles the share, which is still a small log.
KEEP_EXTENSIONS = (
    ".csv", ".tsv", ".tab", ".psv",
    ".dat", ".data", ".txt",
)

# Compressed data files are kept too: `foo.csv.gz` is worth knowing about even
# though nothing decompresses it yet.
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz", ".zst", ".zip")

# What the Parquet index exposes as the corpus universe, and so what `probe`
# ever fetches. .tsv and .psv are the same grammar with a different delimiter
# (both are in features.DELIMITERS), and nearly free in pool size. The Kaggle
# index uses the same list.
#
# .txt and .dat stay out: on Kaggle they are 42% of all files, and a sample of
# the .txt ones was mostly object detection labels (space separated numbers,
# often a single line), text, lists and config files; about 6% were comma,
# tab, semicolon or pipe tables.
#
# .tab stays out: the extension is not specific to tabular data (guitar
# tablature, editor state files), and admitting it would put files with no
# delimiter structure into a pool meant for dialect stratification. It is in
# KEEP_EXTENSIONS, so adding it later costs a `compact`, not a rescan.
INDEX_EXTENSIONS = (".csv", ".tsv", ".psv")

# Smallest file the index admits. 0 on purpose: an empty file, a lone newline,
# a bare BOM or a header with no rows are the inputs parsers crash on, which is
# what a correctness corpus wants. GitHub search cannot return empty files, so
# they only arrive via `trees` harvesting a repo found some other way.
INDEX_MIN_BYTES = 0


def is_kept(path: str) -> bool:
    """Whether a blob path earns a row in the tree log."""
    name = path.rsplit("/", 1)[-1].lower()
    if name.endswith(KEEP_EXTENSIONS):
        return True
    for suffix in COMPRESSION_SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)].endswith(KEEP_EXTENSIONS):
            return True
    return False

# GitHub's code search index silently drops anything larger than this, which is
# why search is used for repo discovery only. Bisected: 350 KiB exactly, with
# files of exactly 358,400 bytes still indexed.
CODE_SEARCH_MAX_BYTES = 358_400

# A code search query never yields more than this many results, whatever
# per_page/page say, and every sort order returns the same set -- so narrow
# shards are the only way past it.
CODE_SEARCH_RESULT_CAP = 1000

# Documented rate limits, used to size the limiters. The x-ratelimit-* headers
# on each response are authoritative and take precedence.
#
# `search/code` is metered by the `code_search` pool (10/min), not by `search`
# (30/min); /rate_limit reports both.
CODE_SEARCH_PER_MINUTE = 10
CORE_PER_HOUR = 5000

# Pace just below the documented ceiling. Our bucket refills continuously while
# GitHub counts per window, so a request we account to one window can land in
# the server's previous one; that is worth about one request of margin.
RATE_SAFETY = 0.95

# Burst allowance for the core pool: enough to absorb jitter between the scan
# workers, not enough to spend an hour's budget in the first minute (a bucket
# is per-process, so every restart would get the burst again).
CORE_BURST = 10

# Minimum spacing between two `search/code` requests. ceil(60 / s) is the most
# that can land in any 60s window, so 6.0s would allow exactly ten -- but
# GitHub bills a request when it reaches their edge, so one we accounted to the
# next window can land in the previous one as an eleventh. One slot is
# reserved instead. RATE_SAFETY is not reused: 5% is half a request at 10/min.
CODE_SEARCH_INTERVAL = 60.0 / (CODE_SEARCH_PER_MINUTE - 1)

# Rate control for raw.githubusercontent.com and media.githubusercontent.com.
#
# These hosts send no x-ratelimit-* headers on success, but they are limited:
# per IP, stated only in a 429 (`X-RateLimit-Resource: ip-raw`, plus an
# `x-ratelimit-rate` counter). Measured behaviour: a burst allowance of roughly
# 40,000 requests served at 250+ req/s, then a sustained rate of ~113,000/h
# (~31/s). A refused request is charged like a served one, so retrying into a
# throttle spends the budget being waited for.
#
# The limit is not a published quota and moves with the IP, so `client.RawPool`
# discovers it (AIMD) rather than hardcoding it. RAW_MAX_RATE is ~2x the
# sustained rate: high enough to spend the burst on a fresh start, low enough
# that the sawtooth settles near the real number.
RAW_START_RATE = 10.0   # requests/second at process start
RAW_MIN_RATE = 0.5
RAW_MAX_RATE = 60.0
RAW_RATE_STEP = 1.0     # added per ~second of clean requests; ~50s to the cap
# Multiplier on a congestion event. An AIMD sawtooth has mean utilisation
# (1 + backoff) / 2. Halving is TCP's choice for a shared network; here the
# limit is a private bucket with a large burst, so an overshoot is cheap and
# 0.75 (87.5% utilisation) is the better trade.
RAW_RATE_BACKOFF = 0.75
RAW_ATTEMPTS = 5        # a throttle parks every worker, so retrying is cheap

# How much slower to climb once a 429 has been seen. Before one, nothing is
# known and the burst is there to be spent; after one, every step above the
# wall costs a rejected request. A single-speed AIMD keeps sawtoothing into the
# limit, and the depth of each tooth is set here: after a throttle, ~30 seconds
# of clean requests per RAW_RATE_STEP (TCP's slow start vs congestion
# avoidance split).
RAW_PROBE_PATIENCE = 30

# Where caution starts, as a fraction of the last observed wall. Below it the
# rate climbs at full speed (that rate was served a moment ago); at or above it
# every step pays the patience factor. CUBIC's shape: recover fast to the last
# known ceiling, approach anything beyond it slowly.
RAW_PROBE_MARGIN = 0.9

# Floor on the hold after a congestion event, in seconds. `1 / rate` is a few
# milliseconds at high rates, which is a re-offence rather than a pause, and
# rejected requests are charged to the budget. Recovery after a throttle is
# fast (seconds), so one second is ample.
RAW_MIN_HOLD = 1.0

# Window over which a pool measures what it achieves, as against what it
# permits. The two diverge whenever something else (threads, the sniff stage)
# is the constraint, and backing off from the permitted rate then changes
# nothing on the wire -- so `RawPool.penalise` cuts from the achieved rate.
RAW_RATE_WINDOW = 10.0

# The same measurement for the pools metered by the hour, which run at ~1 req/s.
# A window can hold at most `capacity + rate * window` requests, so a short one
# reads far above the limit after a burst and near zero during a quiet spell;
# 120s keeps the reading within a few percent of the limit while staying no
# older than one heartbeat.
API_RATE_WINDOW = 120.0

# Round-trip time to raw, in seconds. Only used to turn a rate into a
# concurrency via Little's law: sustaining R req/s at latency L needs R x L in
# flight. Measure it on files the CDN has not cached -- a warm Fastly cache
# reports about half the real figure.
RAW_LATENCY = 0.26

# In-flight cap, derived. Above R x L a thread cannot raise throughput, only
# wait somewhere else.
RAW_MAX_INFLIGHT = math.ceil(RAW_MAX_RATE * RAW_LATENCY)

# How long a congestion event on api.github.com stays open after its hold
# expires, so the rejections handed to requests already on the wire count
# against the event that caused them rather than each announcing a new one.
# Sized to a whole GraphQL batch rather than a round trip: too generous only
# merges a genuinely new event into the old one.
API_LATENCY = 15.0

# Sniffer worker processes. The sniffers are CPU-bound and mostly pure Python,
# and threads make them slower, not faster: CleverCSV's type detection makes
# thousands of `regex` calls per file, each dropping and retaking the GIL. In
# processes the work scales close to linearly. Batched GraphQL delivers heads
# in bursts of thousands, so the sniffers are what the pipeline waits on and
# the pool is sized to the core count.
#
# `--sniff-procs 0` sniffs in the fetch threads, which is the way to get a real
# traceback out of a crashing sniffer.
SNIFF_PROCESSES = os.cpu_count() or 4


def sniff_threads(procs: int) -> int:
    """Threads feeding the sniff process pool, given its size.

    Each feeder submits one file and waits on its future, so the pool runs only
    as deep as there are feeders. Two per process keeps every worker fed across
    the round trip.

    Never zero: `--sniff-procs 0` means "sniff in the calling thread", and the
    calling thread is the feeder.
    """
    return max(2, (procs or 1) * 2)


# Fetched heads allowed to sit between the fetch stages and the sniff stage.
# The only backpressure in the pass: a batched read delivers a thousand blobs in
# seconds, so without a bound the fetchers would read the draw into memory.
# Worst case is this many heads at GRAPHQL_TEXT_CAP (~250 MiB); large enough to
# hold a whole GraphQL batch plus the next one arriving.
SNIFF_QUEUE = 512

# Minimum seconds between repaints of the progress line. The pass writes
# hundreds of records a second, and repainting on every one (inside the writer
# lock) made the terminal write the slowest step in the writer path.
PROGRESS_INTERVAL = 0.2

# Records per heads shard. Every `candidates` draw opens every shard, so tens
# of thousands of small shards would dominate it; the crash window is bounded
# by the zstd frame flush (log.FLUSH_RECORDS) regardless of shard size.
HEADS_ROTATE = 100_000

# GraphQL point budget. Batched blob reads alias many blobs across several
# repositories into one query, and GitHub bills each query exactly one point
# however much it returns (confirmed via `rateLimit { cost }`). That makes
# this budget, times GRAPHQL_BATCH_BYTES, the throughput of the probe.
GRAPHQL_PER_HOUR = 5000

# GraphQL batches in flight at once. The point budget paces the pool to
# ~1.3 requests/s, and at ~6s per batch Little's law puts ~8 in flight at that
# rate; 16 leaves room for variance. A sweep to 32 in flight drew no secondary
# limit refusals, so the ceiling is memory: each batch is up to
# GRAPHQL_BATCH_BYTES on the wire and again as parsed JSON. Raising this does
# not raise throughput -- only GRAPHQL_BATCH_BYTES does.
GRAPHQL_MAX_INFLIGHT = 16

# Burst allowance for the GraphQL pool. A burst larger than the in-flight cap
# can only queue at the semaphore, and a large one released when a hold clears
# fires straight back into the limit that caused it.
GRAPHQL_BURST = 2

# Files larger than this skip GraphQL and take a range request instead. Both
# paths return min(size, GRAPHQL_TEXT_CAP) bytes, so total head bytes are the
# same either way; the question is only which budget pays. GraphQL throughput
# does not depend on file size, only on bytes returned, while raw is metered
# per request and is the scarcer budget. So everything GraphQL will serve goes
# through it, and the range path is left with LFS objects, blobs above
# GRAPHQL_BLOB_MAX and whatever GraphQL declines.
#
# Separate from GRAPHQL_BLOB_MAX because the two mean different things: this
# is a policy about which budget to spend, that one a fact about the server.
# Lower this if the raw pool is ever the one with headroom.
PROBE_GRAPHQL_MAX_BYTES = 10_000_000

# Maximum blobs (aliases) per GraphQL batch. Besides bounding memory, this is a
# hard serving limit: with the cap lifted, every batch that failed (504) carried
# ~3,000 aliases or more, while batches of a few hundred aliases were served at
# 64 MiB. Lowering GRAPHQL_BATCH_BYTES is not a substitute for it.
PROBE_BATCH = 2000
# Files `probe` draws per pass when run without --limit. Drawing the whole
# unprobed pool at once sorts it and holds every row as a dict -- tens of GB on
# a large index -- so an unlimited run is a sequence of passes of this size.
# Each pass re-reads the index and the heads written since the last `compact`,
# so this is large to amortise that; it is not a memory knob.
PROBE_DRAW = 200_000

# Response-byte budget per GraphQL batch, and the most important number in the
# probe: a batch costs one point whatever it carries, so throughput is
# 4,750 points/h x this. Swept against the live API:
#
#     budget    served   mean time
#     16 MiB      6/6        6.6s
#     32 MiB      6/6        6.1s
#     48 MiB      5/6       10.1s
#     64 MiB      4/6        8.0s
#
# 32 MiB is the largest that was always served. A failed batch is retried by
# `_request` before being abandoned, so it costs far more than the point a
# larger batch would save.
GRAPHQL_BATCH_BYTES = 32 << 20

# Distinct repositories per GraphQL batch, the cap that usually binds. Inside
# a query a `repository()` resolution costs 20-75x an `object()` lookup in an
# already-resolved repo, so batches spanning many repos are refused at sizes
# and blob counts that a few repos serve easily:
#
#     repos  blobs  bytes     result
#       125    125  1.70 MB   ok
#       150    150  2.20 MB   502
#       296    400  0.29 MB   partial: 74 served, 326 refused
#        20   2400  0.61 MB   ok
#        50   6000  1.39 MB   504
#
# 20 leaves room under the ~125 repo-equivalent ceiling for the blob term.
# `trees._blob_bodies` is unaffected: all its aliases address one repository.
GRAPHQL_BATCH_REPOS = 20

# Where GitHub truncates `Blob.text`: exactly 512,000 bytes, a clean byte
# prefix identical to what a range request returns, with `byteSize` still the
# file's true length. Larger files come back as a 512,000-byte head rather than
# failing, which is what lets one path serve most of the pool.
GRAPHQL_TEXT_CAP = 512_000

# Above this, GraphQL stops serving the blob at all, and says so badly: `text`
# is an empty string, `isTruncated` true, `isBinary` null. The fidelity check in
# `client._blob_body` catches it, but a file known to fail wastes a batch slot.
# Bisected between 10,155,001 (served) and 11,416,275 (empty); set below the
# known-good point, since erring low costs one range request.
GRAPHQL_BLOB_MAX = 10_000_000

# How much of each file the probe reads. Equal to GRAPHQL_TEXT_CAP so the whole
# corpus has one sampling depth: most heads arrive through GraphQL, which
# truncates there, and a deeper head on the range path (where the large and LFS
# files are) would correlate sample depth with file size.
#
# 512,000 bytes is thousands of rows, and dialect is settled in the first few
# (`sniffers._clevercsv` reads only 8 KiB). What is given up is `complete` for
# files above the cap, and with it exact `rows_sampled`, `max_line_len` and
# `ends_with_newline`; covering those properly means downloading the file.
#
# Files already probed are never re-probed (`probe.retired_keys`), so changing
# this later leaves the corpus split between two sampling depths.
HEAD_BYTES = GRAPHQL_TEXT_CAP

# Optional cache of the fetched head bytes. Off by default because a large
# run is measured in terabytes; when on it can live on a different filesystem
# from the logs (a NAS, say) via CSVSCRAPE_CACHE.
CACHE_ROOT = Path(os.environ.get("CSVSCRAPE_CACHE", "")) if os.environ.get("CSVSCRAPE_CACHE") else None


def cache_root(explicit: str | None = None) -> Path:
    """Where cached head bytes go. Defaults under the data root."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    if CACHE_ROOT:
        return CACHE_ROOT.expanduser().resolve()
    return DATA_ROOT / "cache"


# Git LFS pointers are ~130 bytes: a version line, an "oid sha256:" line and a
# "size" line. Any indexed blob in this range is a pointer candidate and gets
# its magic prefix checked.
LFS_POINTER_MIN = 100
LFS_POINTER_MAX = 250
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

# Hard ceiling on what pointer resolution will fetch. Many LFS repos track
# `*.csv` wholesale, so pattern matching alone would make every csv in them a
# candidate, including files committed before LFS was enabled. A real pointer
# is always ~130 bytes, so a .gitattributes rule may widen the size window but
# never past this.
LFS_CANDIDATE_MAX_BYTES = 1024

# Blobs per GraphQL request when resolving LFS pointers. Candidates are at most
# LFS_CANDIDATE_MAX_BYTES, so bytes never bind; response time grows linearly
# with alias count, so a bigger batch saves points but not wall time. 1,000
# batches, run a few at a time, keep a pointer-heavy repo to seconds.
LFS_BATCH = 1_000

# Repos with enormous trees are recorded but their blob list is capped, so one
# monorepo cannot dominate a log shard.
MAX_BLOBS_PER_REPO = 50_000

# Log shards roll over this often. Smaller means less lost on an unclean exit
# and more files.
ROTATE_RECORDS = 2_000


def token() -> str:
    """Prefer the environment; fall back to the git-ignored token file."""
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    for candidate in (Path("github_token"), Path(__file__).parent.parent / "github_token"):
        if candidate.exists():
            return candidate.read_text().strip()
    raise SystemExit("no token: set GITHUB_TOKEN or create ./github_token")

# Memory one sniffer's DuckDB connection may use. A guard rail, not a working
# set: `sniff_csv` over a head needs a few MB, so anything reaching this is
# pathological and is better refused (the refusal becomes that file's verdict).
# DuckDB's default is 80% of host RAM *per connection*, with one connection per
# sniffer process, which is enough to get the host OOM-killed.
SNIFF_DUCKDB_MEMORY = "256MB"
