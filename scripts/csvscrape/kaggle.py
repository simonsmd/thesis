"""Kaggle as a second, independent corpus source.

Deliberately parallel to the GitHub pipeline rather than merged into it: two
logs, two indexes, two status views. The Parquet schema is column-compatible,
so every selector and both draw shapes work over either pool -- but a dataset
draws from one or the other, never a union.

The API shape, established by probing rather than by reading the docs:

  * `/datasets/list` returns a fixed 20 items per page and caps at page 500,
    so any single query reaches at most 10,000 datasets. `minSize`/`maxSize`
    filter on the *dataset's* total bytes and take arbitrary integers, so the
    cap is defeated by subdividing the size axis until no shard is capped --
    see `shards`.
  * `minSize=0` silently disables the size filter. Every shard floors at 1.
  * `/datasets/download/{ref}` 302s to a signed GCS URL for the dataset's
    `bundle/archive.zip`, valid for three days, and GCS honours HTTP Range.
    That one metered call is therefore worth an unlimited number of unmetered
    reads into the archive, which is what the whole design below turns on.
  * the per-file endpoint `/datasets/download/{ref}/{name}` buys nothing: it
    returns `text/csv` for some files and a single-member ZIP for others, and
    it is charged to the same budget as the bundle. Everything reads out of
    the bundle instead, where the framing is uniform. (A member name holding a
    `/` must be quoted whole if it is ever used: unquoted, no route matches
    and the edge answers an HTML 404 that looks like a throttle.)
  * the SDK's RPC route, `POST api.kaggle.com/v1/datasets.DatasetApiService/
    DownloadDataset`, 302s to the same signed bundle URL on a budget of its
    own -- see `KAGGLE_PER_HOUR`. Otherwise it answers exactly as the REST
    route does, pinned or not.
  * a 404 from `/datasets/download` is frequently a *throttle*, not a missing
    dataset: a burst draws HTML 404s from Google's edge that clear after a
    couple of minutes. A JSON 404 is real; an HTML one is the penalty box.
    `Kaggle._metered` tells them apart, because treating the latter as
    terminal would permanently retire healthy datasets.

Two structural differences from GitHub, both consequences of the same fact --
there is no per-file content hash in the API:

  * identity comes from the bundle's central directory instead: CRC32 and
    uncompressed size, which `index.build_kaggle` uses as the content key.
  * a dataset is *versioned*, not committed. The signed URL pins a version and
    `datasetVersionNumber` re-requests one. A pinned request is honoured
    whenever Kaggle still holds that version, but it does not keep them all.
    So a dataset republished between indexing and download is fetched as
    indexed if its old version survived, and is a loss otherwise. `download`
    records those as `version_drift` and redraws.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import queue
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import requests

from . import config, log, zipindex
from .util import interleave, run_workers
from .client import Bucket

API = "https://www.kaggle.com/api/v1"
RPC = "https://api.kaggle.com/v1/datasets.DatasetApiService"

# The download budget, measured: a token bucket of ~1,000 requests refilling
# at ~16.7 a minute (1,000 an hour). In front of it sits a faster limit of
# ~50/minute with a 50-request burst, which is all a short trial sees -- pacing
# to that drains the 1,000 allowance within the hour. It is a bucket, not an
# hourly window: once drained the rate sits on the refill.
#
# A refused call is charged against it, so the working rate must stay under
# the refill: every call past it buys a 429 with a token that would have
# signed a dataset.
#
# The budget is per route: the REST download and the SDK's RPC download each
# have their own. The per-file download shares the REST route's budget;
# `/datasets/list` has its own (below).
#
# Nothing in a response states any of this -- a served call carries no budget
# header and a 429 carries only `retry-after` -- so it is paced blind.
KAGGLE_PER_HOUR = 1000

# `/datasets/list`, which discovery spends. Measured on the ~50/minute fast
# tier only; a discovery sweep is small enough that a slower tier behind it
# has not been looked for.
KAGGLE_LIST_PER_MINUTE = 50

# Burst allowance for every Kaggle bucket -- deliberately not the server's
# 1,000. A local bucket starts full on every restart while the server's may
# not, so a large burst would be spent on refusals, which cost budget.
KAGGLE_BURST = 8

# The routes that sign a bundle URL, each metered on its own budget: the REST
# download and the SDK's RPC. `bundle` runs one sign worker per route.
ROUTES = ("rest", "rpc")

# The only answers that retire a dataset: the API itself saying it is not
# there for us. Both routes answer a private or deleted dataset with a JSON
# 403, and a version Kaggle has not kept with a JSON 404. Anything else -- a
# throttle that outlasted every retry (`-1`), a 5xx, a dropped connection --
# says nothing about the dataset, and recording it would retire a healthy one
# for good, because `bundled` never asks about a recorded dataset again.
#
# The bundle pass signs pinned to discovery's version, so a 404 there is that
# version no longer kept, and it retires the dataset: nothing this pipeline
# holds knows a newer version number, and reading one unpinned would record
# offsets under a version they do not belong to.
GONE = (403, 404)

# Messages older logs recorded as `no_bundle` when a read failed rather than
# when the object was not an archive. Kept only so `settled` can pass over
# those records; nothing writes them now.
_UNREAD = frozenset({
    "could not read the archive tail",
    "no content-range on the tail read",
    "could not read the central directory",
})


class Unreadable(Exception):
    """An archive read failed this time -- as against `zipindex.BadZip`,
    which is the bytes arriving and not being an archive."""

PAGE_SIZE = 20
MAX_PAGE = 500

# One sort order. Size subdivision is what gets past the 10,000 cap: two
# disjoint size shards cannot return the same dataset, and subdivision
# terminates with the pool provably enumerated.
SORT = "hottest"

# The size axis, subdivided adaptively. Roots only: everything below them is
# derived by `split_at` as shards turn out to be capped, so the schedule adapts
# to where the datasets actually are instead of to a guess baked in here.
#
# Floored at 1, never 0: `minSize=0` disables the filter entirely. The lower
# bound costs nothing real -- a 0-byte dataset holds no files.
ROOT_SHARDS = (
    (1, 1_000),
    (1_000, 1_000_000),
    (1_000_000, 1_000_000_000),
    (1_000_000_000, 10 ** 13),
)

# Files head-probed per dataset. The pool is extraordinarily top-heavy: most
# datasets hold one CSV and a few hold tens of thousands, so uncapped, the
# pass's cost and any pool-level statistic would be dominated by a handful of
# publishers. Only a small fraction of datasets reach the cap.
#
# This is not what keeps a drawn corpus diverse: `sample.draw`'s per-repo and
# per-owner caps already do that.
MAX_FILES_PER_DATASET = 250

# Members whose coordinates are written to the log, whether or not they are
# probed. Recording more than is probed is what makes the probe cap a decision
# that can be revisited: the rows are already in the index, unprobed, so
# raising `--per-dataset` later adds head fetches rather than another pass
# over every archive's central directory.
#
# A log-size decision rather than a statistical one. The recorded set is in
# hash order and the probed set is its prefix, so raising the cap re-probes
# nothing.
RECORD_MEMBERS = 5_000

# How long a signed GCS URL stays good (three days). Shorter than a full bundle
# pass, which is why `bundle` probes inline rather than in a second pass that
# would have to re-sign nearly every dataset.
SIGNED_URL_TTL = 259_200

# Bytes read from a member's compressed stream to produce a head. Deflate
# rarely expands, so this is normally more than enough for `config.HEAD_BYTES`
# of output; when it is not, `_head_of` widens once rather than guessing high
# for every file.
HEAD_COMPRESSED = 512 * 1024

DATASETS_LOG = "kaggle/datasets"
BUNDLES_LOG = "kaggle/bundles"
SHARDS_LOG = "kaggle/shards"
HEADS_LOG = "kaggle/heads"

# How long a queue wait lasts before a worker checks whether to stop.
_STALL = 0.2


def token() -> str:
    """Kaggle API token. Newer tokens are `KGAT...` bearer strings."""
    import os

    tok = os.environ.get("KAGGLE_TOKEN")
    if tok:
        return tok.strip()
    for candidate in (Path("kaggle_token"),
                      Path(__file__).parent.parent / "kaggle_token"):
        if candidate.exists():
            raw = candidate.read_text().strip()
            if raw.startswith("{"):
                import json

                return json.loads(raw)["key"]
            return raw
    raise SystemExit("no kaggle token: set KAGGLE_TOKEN or create ./kaggle_token")


class Kaggle:
    """The metered API, plus unmetered reads against the storage it points at.

    Only `get` and `signed_url` spend the rate budget, each against the bucket
    of the budget it is charged to: `list` for discovery, `rest` and `rpc` for
    the two routes that sign a bundle. `range_get` talks to GCS, which has
    shown no throttling at the rates this pipeline uses, and is deliberately
    not paced.
    """

    def __init__(self, tok: str | None = None, on_event: Any = None):
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {tok or token()}",
             "User-Agent": "csv-corpus-scraper"}
        )
        self.on_event = on_event or (lambda *a, **k: None)
        # One bucket per server-side budget, each stated in the unit its limit
        # is, reporting its rate windows through `on_event`.
        hourly = KAGGLE_PER_HOUR * config.RATE_SAFETY
        self.buckets = {
            "list": Bucket(60.0, KAGGLE_LIST_PER_MINUTE * config.RATE_SAFETY,
                           capacity=KAGGLE_BURST, on_event=self.on_event,
                           name="kaggle_list"),
            **{route: Bucket(3600.0, hourly, capacity=KAGGLE_BURST,
                             on_event=self.on_event, name=f"kaggle_{route}")
               for route in ROUTES},
        }
        # What the server said about the budget, for the run summary. Only a
        # 429 says anything, and only `retry-after`.
        self.limits: dict[str, Any] = {"kaggle_retry_after": None,
                                       "kaggle_limit_headers": None}
        self.ranges = 0
        self.range_bytes = 0
        self._signed: dict[tuple[str, int | None],
                           tuple[str | None, int | None, float]] = {}
        self._lock = threading.Lock()

    # -- metered ----------------------------------------------------------

    def get(self, path: str, _redirect: bool = False, _budget: str = "list",
            **params: Any) -> Any:
        """One metered GET against the REST API, retried through throttles."""
        return self._metered(
            _budget, _redirect,
            lambda: self.session.get(API + path, params=params, timeout=60,
                                     allow_redirects=not _redirect))

    def _metered(self, budget: str, redirect: bool, send: Any) -> Any:
        """One metered call, paced by `budget`'s bucket and retried through
        its throttles.

        A 404 carrying HTML is the edge's throttle response rather than a
        statement about the resource, so it is retried like a 429. A 404
        carrying JSON -- or nothing -- is the API's own answer and is final.
        Getting this backwards would permanently retire healthy datasets.
        """
        bucket = self.buckets[budget]
        for attempt in range(6):
            # Our own pacing goes unannounced: at these rates every call waits
            # on it, so a line per wait is noise. Only the server's refusals
            # are reported, by `_throttled`.
            bucket.take()
            try:
                resp = send()
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            headers = self._read_limits(resp)
            if resp.status_code == 429:
                # Floored at the pacing interval: a retry inside the throttle
                # cannot succeed and is still charged to the budget.
                wait = max(self._retry_after(headers, 30.0 * (attempt + 1)),
                           bucket.min_interval)
                bucket.refused()
                self._throttled(bucket, wait, "rate limit")
                bucket.hold_for(wait)
                continue
            if resp.status_code == 404 and _is_html(resp):
                # The edge's penalty box, which takes minutes to clear. Back off
                # hard rather than hammering it.
                wait = 30.0 * (attempt + 1)
                bucket.refused()
                self._throttled(bucket, wait, "edge block")
                bucket.hold_for(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if redirect:
                return resp
            if resp.status_code != 200:
                return {"error": resp.status_code, "message": resp.text[:200]}
            try:
                return resp.json()
            except ValueError:
                return {"error": 0, "message": "non-json response"}
        return {"error": -1, "message": "retries exhausted"}

    # -- what the pool spent, and what the server said about it ----------

    @property
    def stats(self) -> dict[str, Any]:
        """What the metered pool spent, for the run summary.

        Each bucket reports under its own name, so the two signing routes are
        read separately: a route refusing while the other is clean has a
        misjudged budget.
        """
        spent: dict[str, Any] = {}
        for bucket in self.buckets.values():
            spent |= bucket.stats
        return spent | self.limits

    def _throttled(self, bucket: Bucket, wait: float, reason: str) -> None:
        """Announce a congestion event once, however many workers it refuses.

        See `client.Congestion`. Per bucket, because an event is per budget.
        """
        if bucket.event.first(wait):
            self.on_event("wait", seconds=wait, reason=f"{bucket.name} {reason}",
                          throttled=True)

    @staticmethod
    def _retry_after(headers: dict[str, Any], default: float) -> float:
        """The server's own number, off the already-lowercased headers.

        Not off `resp.headers`, which is case-insensitive only when it comes
        from `requests`.
        """
        try:
            return float(headers.get("retry-after", default))
        except (TypeError, ValueError):
            return default

    def _read_limits(self, resp: requests.Response) -> dict[str, Any]:
        """Record whatever the response says about the budget, and return its
        headers lowercased for everything else here to read.

        Read from every response, not only refusals, so a budget header that
        starts appearing is noticed.
        """
        seen = {str(k).lower(): str(v) for k, v in resp.headers.items()}
        if resp.status_code == 429:
            retry = seen.get("retry-after")
            if retry:
                try:
                    self.limits["kaggle_retry_after"] = float(retry)
                except (TypeError, ValueError):
                    pass
        named = {k: v for k, v in seen.items()
                 if "ratelimit" in k or "rate-limit" in k}
        if named:
            self.limits["kaggle_limit_headers"] = ", ".join(
                f"{k}={v}" for k, v in sorted(named.items()))
        return seen

    def signed_url_cached(self, ref: str, version: int | None = None
                          ) -> tuple[str | None, int | None]:
        """`signed_url`, memoised for the life of the URL.

        A corpus draw routinely takes several files from one dataset, and each
        would otherwise spend a metered call to re-sign the same archive. The
        URL is re-signed at half its lifetime.
        """
        with self._lock:
            hit = self._signed.get((ref, version))
            if hit and time.time() - hit[2] < SIGNED_URL_TTL / 2:
                return hit[0], hit[1]
        url, status = self.signed_url(ref, version)
        with self._lock:
            self._signed[(ref, version)] = (url, status, time.time())
        return url, status

    def signed_url(self, ref: str, version: int | None = None,
                   route: str = "rest") -> tuple[str | None, int | None]:
        """A signed GCS URL for a dataset's bundle, and the HTTP status.

        `version` pins `datasetVersionNumber`, and a 404 on a pinned version is
        how version drift is detected on the download path: the pin is served
        as long as Kaggle kept that version, so the pin failing *is* the drift.

        `route` picks the budget the call is charged to, and nothing else: the
        two routes answer identically, pinned or not.
        """
        if route == "rpc":
            owner, slug = ref.split("/", 1)
            body = {"ownerSlug": owner, "datasetSlug": slug}
            if version:
                body["datasetVersionNumber"] = version
            resp = self._metered(
                "rpc", True,
                lambda: self.session.post(f"{RPC}/DownloadDataset", json=body,
                                          timeout=60, allow_redirects=False))
        else:
            params = {"datasetVersionNumber": version} if version else {}
            resp = self.get(f"/datasets/download/{ref}", _redirect=True,
                            _budget="rest", **params)
        if isinstance(resp, dict):
            return None, resp.get("error")
        if resp.status_code in (301, 302, 303, 307, 308):
            return resp.headers.get("location"), resp.status_code
        return None, resp.status_code

    # -- unmetered --------------------------------------------------------

    def range_get(self, url: str, start: int | None, end: int | None = None,
                  attempts: int = 4) -> tuple[bytes, int | None] | None:
        """Bytes from a signed URL, with the object's total length.

        `start=None` asks for a *suffix* range -- `bytes=-N`, the last N bytes
        -- which is how the archive tail is read without knowing how long the
        archive is. The response's `content-range` carries the total.

        `end` is inclusive, as in the header. A signed URL that has expired
        comes back 403, which the caller distinguishes from a transport
        failure by re-signing rather than retrying.
        """
        span = (f"bytes=-{end}" if start is None
                else f"bytes={start}-{'' if end is None else end}")
        for attempt in range(attempts):
            try:
                resp = self.session.get(url, headers={"Range": span},
                                        timeout=300)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (200, 206):
                with self._lock:
                    self.ranges += 1
                    self.range_bytes += len(resp.content)
                header = resp.headers.get("content-range", "")
                total = None
                if "/" in header:
                    try:
                        total = int(header.rsplit("/", 1)[1])
                    except ValueError:
                        total = None
                return resp.content, total
            if resp.status_code in (403, 404, 416):
                return None
            time.sleep(2 ** attempt)
        return None

    def read_range(self, url: str, start: int, end: int | None = None,
                   attempts: int = 4) -> bytes | None:
        """`range_get` for callers that only want the bytes."""
        got = self.range_get(url, start, end, attempts)
        return None if got is None else got[0]


def _is_html(resp: requests.Response) -> bool:
    kind = resp.headers.get("content-type", "")
    return "html" in kind.lower() or resp.content[:9].lower() == b"<!doctype"


# -- phase 1: dataset discovery -------------------------------------------


def split_at(lo: int, hi: int) -> int:
    """Where a capped shard divides.

    Geometric, not arithmetic: dataset sizes are distributed over orders of
    magnitude, so the arithmetic midpoint of 1,000..1,000,000,000 puts
    essentially the whole population on one side and splits it again and again
    before making progress. The geometric midpoint halves the *log* range,
    which is where the density is roughly flat.
    """
    mid = math.isqrt(lo * hi) if lo > 0 else hi // 2
    return max(lo + 1, min(mid, hi - 1))


def _shard_key(lo: int, hi: int) -> str:
    return f"{lo}..{hi}"


def shard_states() -> dict[str, dict[str, Any]]:
    """Newest record per shard."""
    states: dict[str, dict[str, Any]] = {}
    for rec in log.read(SHARDS_LOG):
        states[rec["shard"]] = rec
    return states


def shards(states: dict[str, dict[str, Any]] | None = None
           ) -> Iterator[tuple[int, int, int]]:
    """(lo, hi, resume_page) for every shard still owing work.

    Walks the subdivision tree from `ROOT_SHARDS`, consulting what the log says
    about each node. A node recorded `capped` is an interior node: it is never
    enumerated -- its children cover exactly the same datasets -- so only its
    two halves are descended into. A node recorded `done` (or `atomic`) is
    finished. Anything else is work, resuming at the page it got to.

    The tree is recomputed rather than stored because `split_at` is pure: the
    same log always yields the same frontier, so a resumed run needs no state
    beyond the shard records themselves.
    """
    states = shard_states() if states is None else states
    stack = list(reversed(ROOT_SHARDS))
    while stack:
        lo, hi = stack.pop()
        rec = states.get(_shard_key(lo, hi))
        state = rec.get("state") if rec else None
        if state == "done" or state == "atomic":
            continue
        if state == "capped":
            mid = split_at(lo, hi)
            stack.append((mid, hi))
            stack.append((lo, mid))
            continue
        yield lo, hi, (rec.get("pages", 0) if rec else 0)


def reopen_exhausted() -> int:
    """Re-open every shard that claims to be finished, keeping its progress.

    For a sweep whose shards may have stopped early: rather than guessing
    which, every finished shard resumes at `pages + 1` and must find an empty
    page. That costs one request per genuinely exhausted shard, and exactly
    the missed pages for the rest.

    Appends a `partial` record rather than rewriting, so the old record stays
    readable. `capped` (interior) nodes are left alone: their children carry
    the work.
    """
    states = shard_states()
    reopened = 0
    with log.Writer(SHARDS_LOG, rotate=200) as sh_log:
        for key in sorted(states):
            rec = states[key]
            if rec.get("state") not in ("done", "atomic"):
                continue
            sh_log.write({**rec, "state": "partial", "reason": "recheck",
                          "ts": int(time.time())})
            reopened += 1
    return reopened


def known_refs() -> set[str]:
    return {r["ref"] for r in log.read(DATASETS_LOG)}


def discover(kg: Kaggle, max_shards: int | None = None,
             max_pages: int = MAX_PAGE, recheck: bool = False,
             on_event: Any = None) -> dict[str, Any]:
    """Enumerate CSV datasets, subdividing the size axis until nothing is capped.

    A shard is tested for capping *before* it is enumerated, by asking for its
    last page: one call answers "are there more than 10,000 datasets here", and
    if there are, the shard is split without any of its pages being read.
    """
    emit = on_event or (lambda *a, **k: None)
    reopened = reopen_exhausted() if recheck else 0
    states = shard_states()
    seen = known_refs()
    emit("start", known=len(seen),
         retired=sum(1 for r in states.values() if r.get("state") == "done"),
         partial=sum(1 for r in states.values()
                     if r.get("state") in ("partial", "error")))
    # `new` is what this invocation added; `known` is what the log held before
    # it; `total` is the size of the pool.
    stats = {"shards": 0, "split": 0, "partial": 0, "failed": 0,
             "requests": 0, "new": 0, "known": len(seen), "total": len(seen),
             "reopened": reopened}
    index = 0
    # Shards this invocation has already taken a turn at. The frontier is
    # recomputed on every iteration (so a split enqueues its children
    # immediately), but a shard that ends `error` or `partial` stays on it;
    # without this the loop would pick it again forever. The next *run*
    # retries them.
    attempted: set[str] = set()

    with log.Writer(DATASETS_LOG) as ds_log, \
            log.Writer(SHARDS_LOG, rotate=200) as sh_log:
        try:
            while True:
                pending = list(shards(states))
                if not pending:
                    break
                if max_shards is not None and stats["shards"] + stats["split"] >= max_shards:
                    break
                fresh = [s for s in pending if _shard_key(s[0], s[1]) not in attempted]
                if not fresh:
                    break
                lo, hi, resume = fresh[0]
                key = _shard_key(lo, hi)
                attempted.add(key)
                index += 1
                emit("shard_start", index=index, of=max_shards, tier="size",
                     query=key, resume=resume)

                # Is this shard capped? One call, at the last reachable page.
                # Only worth asking when the budget reaches the API's own cap:
                # below that a full last page says the budget ran out, not that
                # the shard is too big to enumerate, so the call buys nothing.
                if resume == 0 and max_pages >= MAX_PAGE:
                    probe_page = kg.get("/datasets/list", page=max_pages,
                                        sortBy=SORT, fileType="csv",
                                        minSize=lo, maxSize=hi)
                    stats["requests"] += 1
                    if isinstance(probe_page, dict) and "error" in probe_page:
                        record = {"shard": key, "lo": lo, "hi": hi,
                                  "state": "error", "pages": 0,
                                  "message": str(probe_page.get("message"))[:200],
                                  "ts": int(time.time())}
                        sh_log.write(record)
                        states[key] = record
                        stats["failed"] += 1
                        emit("shard_error", index=index,
                             message=str(probe_page.get("message", ""))[:60])
                        continue
                    if probe_page:
                        if hi - lo <= 1:
                            # Cannot subdivide further. Enumerate what is
                            # reachable and record that the rest is not; the
                            # alternative is splitting forever on a shard that
                            # is already one byte wide.
                            emit("shard_end", index=index, tier="size",
                                 query=key, state="atomic", reason="atomic",
                                 pages=0, total=stats["new"])
                        else:
                            mid = split_at(lo, hi)
                            record = {"shard": key, "lo": lo, "hi": hi,
                                      "state": "capped", "split_at": mid,
                                      "pages": 0, "ts": int(time.time())}
                            sh_log.write(record)
                            states[key] = record
                            stats["split"] += 1
                            emit("shard_end", index=index, tier="size",
                                 query=key, state="split",
                                 reason=f"capped, split at {mid:,}",
                                 pages=0, total=stats["new"])
                            continue

                pages = resume
                reason = "page_budget"
                failed = False
                # Pages the API cut short of PAGE_SIZE; see below.
                short = 0
                for page in range(resume + 1, min(max_pages, MAX_PAGE) + 1):
                    body = kg.get("/datasets/list", page=page, sortBy=SORT,
                                  fileType="csv", minSize=lo, maxSize=hi)
                    stats["requests"] += 1
                    if isinstance(body, dict) and "error" in body:
                        failed = True
                        stats["failed"] += 1
                        emit("shard_error", index=index,
                             message=str(body.get("message", ""))[:60])
                        ds_log.flush()
                        record = {"shard": key, "lo": lo, "hi": hi,
                                  "state": "error", "pages": pages,
                                  "message": str(body.get("message", ""))[:200],
                                  "ts": int(time.time())}
                        sh_log.write(record)
                        states[key] = record
                        break
                    if not body:
                        reason = "exhausted"
                        break

                    new_here = 0
                    for item in body:
                        ref = item.get("ref")
                        if not ref or ref in seen:
                            continue
                        seen.add(ref)
                        new_here += 1
                        stats["new"] += 1
                        ds_log.write({
                            "ref": ref,
                            "owner": item.get("ownerRef") or ref.split("/")[0],
                            "title": item.get("title"),
                            # The version pin. Free here, and a metered call per
                            # dataset later.
                            "dataset_id": item.get("id"),
                            "version": item.get("currentVersionNumber"),
                            "total_bytes": item.get("totalBytes"),
                            "license": item.get("licenseName"),
                            "votes": item.get("voteCount"),
                            "downloads": item.get("downloadCount"),
                            "usability": item.get("usabilityRating"),
                            "last_updated": item.get("lastUpdated"),
                            "shard": key, "size_hint": hi,
                            "ts": int(time.time()),
                        })
                    pages = page
                    emit("page", index=index, page=page, hits=len(body),
                         new=new_here, total=stats["new"])
                    if len(body) < PAGE_SIZE:
                        # NOT exhaustion. `fileType=csv` is applied after the
                        # page is cut, so a page can come back short with more
                        # pages behind it (common in bands dominated by image
                        # and audio datasets). Only an empty page ends a shard.
                        short += 1

                if failed:
                    continue
                ds_log.flush()
                state = ("atomic" if hi - lo <= 1 and reason != "exhausted"
                         else "partial" if reason == "page_budget" else "done")
                record = {"shard": key, "lo": lo, "hi": hi, "state": state,
                          "reason": reason, "pages": pages, "short_pages": short,
                          "ts": int(time.time())}
                sh_log.write(record)
                states[key] = record
                stats["shards" if state in ("done", "atomic") else "partial"] += 1
                emit("shard_end", index=index, tier="size", query=key,
                     state=state, reason=reason, pages=pages,
                     total=stats["new"])
        except KeyboardInterrupt:
            stats["interrupted"] = True
    stats["total"] = len(seen)
    # `getattr`: a test double may answer `get` and nothing else.
    stats |= getattr(kg, "stats", {})
    return stats


# -- phase 2: bundles -- enumeration and head-probing behind one signed URL --


def settled(record: dict[str, Any]) -> bool:
    """Whether a bundle record is final, so its dataset is never asked again.

    `ok` is, and `no_bundle` on an object that was read and is not an archive.
    `gone` only on a status in `GONE`: the API saying the dataset is not there
    for us. Nothing else says anything about the dataset.

    An unpinned `ok` (older records) read whatever version was current, so if
    its archive length differs from discovery's `total_bytes` it read a later
    version than the one it names, and is read again, pinned. `drifted`, a
    revisit finding the recorded archive changed, is read again too.

    Older logs also hold `gone` records for throttles that outlasted every
    retry (`http: -1`) and `no_bundle` records for failed reads (`_UNREAD`);
    those are passed over so the next run asks again.
    """
    if record.get("state") == "gone":
        return record.get("http") in GONE
    if record.get("state") == "no_bundle":
        return record.get("message") not in _UNREAD
    if record.get("state") == "drifted":
        return False
    if record.get("state") == "ok" and not record.get("pinned"):
        return record.get("zip_bytes") == record.get("total_bytes")
    return True


_VERDICT = ("ref", "state", "http", "message", "pinned", "zip_bytes",
            "total_bytes", "ts", "rec_id")


def _newest_verdicts() -> dict[str, dict[str, Any]]:
    """The newest bundle record per dataset, as the scalars `settled` reads.

    The newest, not any: an `ok` can stop being final (see `settled`), and a
    record written after it is what says so. `extended` records say nothing
    about the archive and are passed over. Scanned rather than parsed, so the
    member arrays of a 5,000-file record are never materialised.
    """
    newest: dict[str, dict[str, Any]] = {}
    for values in log.scan_fields(BUNDLES_LOG, _VERDICT):
        rec = dict(zip(_VERDICT, values))
        if rec["state"] == "extended":
            continue
        held = newest.get(rec["ref"])
        stamp = (rec["ts"] or 0, rec["rec_id"] or "")
        if held is None or (held["ts"] or 0, held["rec_id"] or "") <= stamp:
            newest[rec["ref"]] = rec
    return newest


def bundled() -> set[str]:
    """Datasets whose newest bundle record is a final answer."""
    return {ref for ref, rec in _newest_verdicts().items() if settled(rec)}


def outcomes() -> dict[str, str]:
    """Where each dataset the pass has read stands, for `kaggle-status`.

    `indexed`, or read again by the next pass -- `reread` (read unpinned,
    from an archive not the size discovery listed), `drifted`, `retry` -- or
    lost for good: `private` (403), `not_found` (404; pinned, the version is no
    longer kept) and `no_bundle`.
    """
    out = {}
    for ref, rec in _newest_verdicts().items():
        state = rec["state"]
        if not settled(rec):
            out[ref] = {"ok": "reread", "drifted": "drifted"}.get(state, "retry")
        elif state == "ok":
            out[ref] = "indexed"
        elif state == "gone":
            out[ref] = "private" if rec["http"] == 403 else "not_found"
        else:
            out[ref] = "no_bundle"
    return out


def pending_datasets(order: str = "interleave") -> list[dict[str, Any]]:
    done = bundled()
    best: dict[str, dict[str, Any]] = {}
    for rec in log.read(DATASETS_LOG):
        if rec["ref"] in done:
            continue
        best.setdefault(rec["ref"], rec)
    rows = list(best.values())
    if order == "priority":
        return sorted(rows, key=lambda r: r.get("total_bytes") or 0, reverse=True)
    if order == "random":
        import random

        rows.sort(key=lambda r: r["ref"])
        random.Random(0).shuffle(rows)
        return rows
    # Interleave by size decade, so an interrupted run spans the spectrum
    # rather than holding only whichever end it started at.
    from collections import defaultdict

    groups: dict[int, list[dict]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: -(r.get("total_bytes") or 0)):
        groups[(row.get("size_hint") or 0).bit_length()].append(row)
    return list(interleave(groups))


def revisits(cap: int = MAX_FILES_PER_DATASET) -> list[dict[str, Any]]:
    """Bundled datasets whose probe prefix, at `cap`, holds undescribed content.

    `pending_datasets` never returns a bundled dataset, so this is what
    retries a failed head, writes a head a stopped run dropped, and reaches
    recorded-but-unprobed files when `--per-dataset` is raised. The
    coordinates are in the bundle record, so a revisit is one metered call to
    sign the archive again and then only the head reads.

    Owed is judged by content, as the index judges it: a key some head has
    already described needs nothing, whichever dataset it was read out of.
    A key owed in several datasets goes to the one owing the most, so the
    signing calls are spent where most heads come back per call.

    Each entry carries `extend_to` when the prefix grows past what earlier
    passes queued; the pass records that as an `extended` record, which is
    what `index.build_kaggle` reads `n_probed` from. A dataset whose owed
    keys were all claimed by another still gets one, and no signing call.
    """
    described = {key for key, fetched
                 in log.scan_fields(HEADS_LOG, ("content_key", "fetched"))
                 if fetched and key}
    newest: dict[str, tuple[tuple, dict[str, Any] | None]] = {}
    queued: dict[tuple[str, Any], int] = {}
    for rec in log.read(BUNDLES_LOG):
        ref, state = rec.get("ref"), rec.get("state")
        if state == "extended":
            at = (ref, rec.get("version"))
            queued[at] = max(queued.get(at, 0), rec["n_probed"])
            continue
        stamp = (rec.get("ts") or 0, rec.get("rec_id") or "")
        if ref in newest and newest[ref][0] > stamp:
            continue
        if state != "ok" or not settled(rec):
            # Newer than the archive the coordinates came from -- a `drifted`
            # revisit, or a failure the dataset has since been re-read past --
            # or coordinates `pending_datasets` is about to replace.
            newest[ref] = (stamp, None)
            continue
        target = min(cap, rec["n_recorded"])
        keys = rec["keys"]
        newest[ref] = (stamp, {
            "ref": ref, "version": rec.get("version"),
            "license": rec.get("license"), "zip_bytes": rec.get("zip_bytes"),
            "n_probed": rec.get("n_probed") or 0, "target": target,
            "members": [
                zipindex.Member(name=rec["names"][i], method=rec["methods"][i],
                                crc=rec["crcs"][i], csize=rec["csizes"][i],
                                usize=rec["sizes"][i],
                                header_offset=rec["offsets"][i])
                for i in range(target) if keys[i] not in described
            ],
        })

    items = [item for _, item in newest.values() if item is not None]
    items.sort(key=lambda item: (-len(item["members"]), item["ref"]))
    claimed: set[str] = set()
    out = []
    for item in items:
        members = []
        for member in item["members"]:
            key = content_key(member.crc, member.usize)
            if key not in claimed:
                claimed.add(key)
                members.append(member)
        done = max(item.pop("n_probed"), queued.get((item["ref"], item["version"]), 0))
        target = item.pop("target")
        item["members"] = members
        item["extend_to"] = target if target > done else None
        if members or item["extend_to"]:
            out.append(item)
    return out


def content_key(crc: int, usize: int) -> str:
    """The key a Kaggle file's bytes hash to.

    CRC32 and uncompressed size together, which is the only content identity
    Kaggle exposes ahead of a download. The CRC leads so that the first two
    characters spread across 256 store directories, and the dash keeps it
    unambiguous against a GitHub blob sha, which is 40 bare hex digits.
    """
    return f"{crc:08x}-{usize:x}"


def select_members(members: list[zipindex.Member],
                   cap: int = MAX_FILES_PER_DATASET,
                   record: int = RECORD_MEMBERS
                   ) -> tuple[list[zipindex.Member], int, int]:
    """(members to record, how many of them to probe, how many were kept).

    Two tiers over one ordering. The recorded members go to the log and so to
    the index; the probed ones are a *prefix* of them, which is the property
    that makes the probe cap revisitable -- raising it takes the next files in
    the same order, and everything already described keeps its verdict.

    The order is a hash of the name rather than the directory's own, which is
    neither random nor meaningless: entries come out grouped by directory, so
    the first N of a big dataset would be one subtree -- a year, a sensor, a
    language -- and every file in it shares a producer. Hashing spreads the
    sample across the archive while staying deterministic, so re-reading a
    dataset picks the same files in the same order.
    """
    kept = [m for m in members
            if not m.is_dir
            and m.readable
            and not m.name.startswith("__MACOSX/")
            and "/.." not in m.name
            and config.is_kept(m.name)]
    ranked = sorted(kept, key=lambda m: hashlib.blake2b(
        m.name.encode("utf-8", "replace"), digest_size=8).digest())
    recorded = ranked[:record]
    return recorded, min(cap, len(recorded)), len(kept)


def read_directory(kg: Kaggle, url: str) -> tuple[list[zipindex.Member], int, int]:
    """Every member of a bundle -- usually in a single range read.

    One suffix range fetches the archive's tail and, through `content-range`,
    its length. The central directory usually lies inside that tail (most
    datasets hold a handful of files); only a larger directory needs a second
    read.

    Returns (members, declared_entries, archive_bytes). Raises `BadZip` when
    the object is not an archive at all, which is how a dataset Kaggle serves
    unbundled is recognised, and `Unreadable` when a read failed -- which says
    nothing about the object, and must not be recorded as though it did.
    """
    got = kg.range_get(url, None, zipindex.EOCD_SEARCH)
    if got is None:
        raise Unreadable("could not read the archive tail")
    tail, total = got
    if total is None:
        raise Unreadable("no content-range on the tail read")
    directory = zipindex.find_directory(tail, total)

    tail_start = total - len(tail)
    if directory.offset >= tail_start:
        # Already in hand. Slicing it out of the tail is exact: `find_directory`
        # resolved the offset against `total`, and the tail's own position in
        # the file is what makes it addressable.
        begin = directory.offset - tail_start
        body = tail[begin:begin + directory.size]
    else:
        body = kg.read_range(url, directory.offset,
                             directory.offset + directory.size - 1)
        if body is None:
            raise Unreadable("could not read the central directory")
    return zipindex.parse_directory(body), directory.entries, total


def _head_of(kg: Kaggle, url: str, member: zipindex.Member,
             head_bytes: int) -> bytes | None:
    """`head_bytes` uncompressed bytes from the start of one member.

    One request in the common case: the local header and the leading
    compressed bytes are fetched together, since the header's length is bounded
    by the name plus a small extra field. A member whose local extra exceeds
    the slack, or whose deflate stream needs more input than was fetched to
    yield the requested output, costs a second read rather than a wrong answer.
    """
    name_len = len(member.name.encode("utf-8", "replace"))
    want_compressed = min(member.csize, max(head_bytes, HEAD_COMPRESSED))
    span = 30 + name_len + zipindex.LOCAL_EXTRA_SLACK + want_compressed
    blob = kg.read_range(url, member.header_offset,
                         member.header_offset + span - 1)
    if blob is None:
        return None
    try:
        start = zipindex.data_offset(blob, member.header_offset)
    except zipindex.BadZip:
        return None
    local = start - member.header_offset
    if local > len(blob):
        # The extra field overran the slack. Re-read from the real data offset.
        body = kg.read_range(url, start, start + want_compressed - 1)
        if body is None:
            return None
    else:
        body = blob[local:local + want_compressed]

    head = zipindex.inflate_prefix(body, member.method, head_bytes)
    if (len(head) < head_bytes and len(body) >= want_compressed
            and want_compressed < member.csize):
        # Ran out of input before running out of appetite: an unusually
        # incompressible member. Widen once, which covers every case short of
        # a stream that does not compress at all.
        wider = min(member.csize, want_compressed * 4)
        body = kg.read_range(url, start, start + wider - 1)
        if body is not None:
            head = zipindex.inflate_prefix(body, member.method, head_bytes)
    return head


def read_member(kg: Kaggle, url: str, member: zipindex.Member) -> bytes | None:
    """One member's whole content, inflated and checked against its CRC.

    The bytes cross the wire deflated, so a member costs its *compressed* size
    to fetch. The CRC comes from the same central directory that located the
    member, so a mismatch means the bytes are not the ones indexed.
    """
    name_len = len(member.name.encode("utf-8", "replace"))
    span = 30 + name_len + zipindex.LOCAL_EXTRA_SLACK + member.csize
    blob = kg.read_range(url, member.header_offset,
                         member.header_offset + span - 1)
    if blob is None:
        return None
    try:
        start = zipindex.data_offset(blob, member.header_offset)
    except zipindex.BadZip:
        return None
    local = start - member.header_offset
    body = blob[local:local + member.csize]
    if len(body) < member.csize:
        body = kg.read_range(url, start, start + member.csize - 1)
        if body is None:
            return None
    try:
        data = zipindex.inflate_all(body, member.method, member.usize)
    except zipindex.BadZip:
        return None
    return data if zipindex.crc_of(data) == member.crc else None


def member_of(row: dict[str, Any]) -> zipindex.Member:
    """Rebuild the archive coordinates an index row carries."""
    return zipindex.Member(
        name=row["path"], method=int(row["method"]), crc=int(row["crc"]),
        csize=int(row["csize"]), usize=int(row["size"]),
        header_offset=int(row["header_offset"]),
    )


def member_row(ref: str, row: dict[str, Any], member: zipindex.Member
               ) -> dict[str, Any]:
    """The index/probe row for one member of one dataset's bundle."""
    size = member.usize
    return {
        "source": "kaggle",
        "content_key": content_key(member.crc, member.usize),
        "full_name": ref,
        "owner": ref.split("/")[0],
        "path": member.name,
        "size": size,
        "bucket": max(size.bit_length() - 1, 0),
        "crc": member.crc,
        "csize": member.csize,
        "method": member.method,
        "header_offset": member.header_offset,
        "version": row.get("version"),
        "license": row.get("license"),
        # Kept so `probe.describe`'s row contract is satisfied without it
        # having to know which source it is describing.
        "is_lfs": False,
        "ref": str(row.get("version") or ""),
    }


def bundle(kg: Kaggle, limit: int | None = None,
           routes: tuple[str, ...] = ROUTES,
           head_workers: int = 16, order: str = "interleave",
           head_bytes: int = config.HEAD_BYTES, cache: bool = False,
           cache_dir: str | None = None,
           which: tuple[str, ...] | None = None,
           sniff_procs: int = config.SNIFF_PROCESSES,
           cap: int = MAX_FILES_PER_DATASET,
           record: int = RECORD_MEMBERS,
           on_event: Any = None) -> dict[str, Any]:
    """Enumerate every dataset's files and describe them, behind one call each.

    Two stages. The scarce budget is the metered call that signs a URL, while
    reads from GCS are not paced, so:

      * `_sign_worker` spends the metered budget: one call per dataset, then
        usually a single unmetered read for the whole file list. One per
        route in `routes`, each charged to its own budget, pulling from one
        queue. One worker per budget covers the paced rate many times over; a
        second would only add rejections, which are charged like served calls.
      * `_head_worker` does everything else: the per-member range reads, the
        sniffers, and the log write. Sniffing blocks these threads, which is
        fine because the budget they use is not scarce -- so the separate sniff
        stage `probe.run` needs does not exist here.

    Enumeration and probing are one pass rather than two because a signed URL
    lives three days, less than a full pass: a separate probe pass would have
    to re-sign nearly every dataset.

    The re-signing is paid only where it buys something: a bundled dataset
    whose probe prefix, at `cap`, still holds content no head described is
    signed once more and read at its recorded coordinates. See `revisits`.
    """
    unknown = set(routes) - set(ROUTES)
    if unknown or not routes:
        raise ValueError(f"routes must be drawn from {ROUTES}, got {routes}")
    from . import probe, sniffers
    from .blobstore import BlobStore

    emit = on_event or (lambda *a, **k: None)
    which = which or sniffers.DEFAULT
    log.seal_orphans(BUNDLES_LOG)
    log.seal_orphans(HEADS_LOG)

    # Revisits first: there are few of them, each is one metered call, and
    # they are what a raised `cap` or a failed head is waiting on.
    todo = revisits(cap) + pending_datasets(order)
    if limit:
        todo = todo[:limit]
    sign_work: queue.Queue = queue.Queue()
    for row in todo:
        sign_work.put(row)
    head_work: queue.Queue = queue.Queue(maxsize=config.SNIFF_QUEUE)

    stats: dict[str, Any] = {
        "datasets": 0, "queued": len(todo), "errors": 0, "no_bundle": 0,
        # Asked and not answered -- left unrecorded for the next run.
        "deferred": 0,
        "files": 0, "probing": 0, "files_seen": 0, "capped": 0, "probed": 0,
        "head_failures": 0, "ranges": 0, "range_bytes": 0,
        # Bundled datasets signed again for heads they owe, and those whose
        # recorded archive could no longer be signed to check them against.
        "revisited": 0, "drifted": 0,
    }
    lock = threading.Lock()
    stopping = threading.Event()
    accepting = threading.Event()
    accepting.set()
    signing = threading.Event()
    live = {"sign": len(routes), "head": head_workers}

    store = BlobStore(config.cache_root(cache_dir)) if cache else None
    pool = probe._sniff_pool(sniff_procs, which)
    crash: list[BaseException] = []

    # The sniffer pool is entered first so it is shut down (and joined) last,
    # after the logs are sealed, however the pass ends.
    with (pool or contextlib.nullcontext()), \
            log.Writer(BUNDLES_LOG, rotate=2000) as bundle_log, \
            log.Writer(HEADS_LOG, rotate=config.HEADS_ROTATE) as heads_log:

        def _record(target: Any, record: dict[str, Any]) -> None:
            with lock:
                if accepting.is_set():
                    target.write(record)

        def _queue_heads(url: str, ref: str, row: dict[str, Any],
                         members: list[zipindex.Member]) -> bool:
            """Hand members to the head workers; False once none are left."""
            for member in members:
                if stopping.is_set():
                    return False
                while not stopping.is_set():
                    try:
                        head_work.put((url, ref, row, member), timeout=_STALL)
                        break
                    except queue.Full:
                        # Every head worker has died (a crash, re-raised at the
                        # end), so nothing will drain the queue. Stop rather
                        # than block forever.
                        with lock:
                            gone = live["head"] <= 0
                        if gone:
                            return False
            return True

        def _revisit(route: str, item: dict[str, Any]) -> bool:
            """Heads for a bundled dataset, read at the coordinates recorded."""
            ref = item["ref"]
            base = {"ref": ref, "owner": ref.split("/")[0],
                    "version": item["version"], "route": route,
                    "ts": int(time.time())}
            url = None
            if item["members"]:
                # Pinned, because the offsets belong to one archive, and checked
                # by archive length before any offset is used: the recorded
                # version is discovery's, not read off the archive itself.
                url, status = kg.signed_url(ref, item["version"], route=route)
                if not url and status not in GONE:
                    with lock:
                        stats["deferred"] += 1
                    return True
                got = kg.range_get(url, None, 1) if url else None
                if url and (got is None or got[1] is None):
                    with lock:
                        stats["deferred"] += 1
                    return True
                if not url or got[1] != item["zip_bytes"]:
                    _record(bundle_log, base | {
                        "state": "drifted", "http": status,
                        "zip_bytes": got[1] if got else None})
                    with lock:
                        stats["datasets"] += 1
                        stats["drifted"] += 1
                    return True
            if item["extend_to"]:
                _record(bundle_log, base | {"state": "extended",
                                            "n_probed": item["extend_to"]})
            with lock:
                stats["datasets"] += 1
                stats["revisited"] += bool(url)
                stats["probing"] += len(item["members"])
                emit("repo", done=stats["datasets"], of=len(todo),
                     note=f"revisit · {stats['probed']:,}/"
                          f"{stats['probing']:,} probed",
                     name=ref)
            return not url or _queue_heads(url, ref, item, item["members"])

        def _sign_worker(route: str) -> None:
            try:
                while not stopping.is_set():
                    try:
                        row = sign_work.get_nowait()
                    except queue.Empty:
                        return
                    if "members" in row:
                        if not _revisit(route, row):
                            return
                        continue
                    ref = row["ref"]
                    base = {"ref": ref, "owner": ref.split("/")[0],
                            "dataset_id": row.get("dataset_id"),
                            "version": row.get("version"),
                            "license": row.get("license"),
                            "total_bytes": row.get("total_bytes"),
                            # Which budget signed it.
                            "route": route,
                            # Signed at the version recorded, so the offsets
                            # below belong to the archive the record names.
                            "pinned": True,
                            "ts": int(time.time())}
                    url, status = kg.signed_url(ref, row.get("version"),
                                                route=route)
                    if not url and status not in GONE:
                        # Not an answer about the dataset, so no record: the
                        # next run asks again. See `GONE`.
                        with lock:
                            stats["deferred"] += 1
                        continue
                    if not url:
                        _record(bundle_log, base | {"state": "gone",
                                                    "http": status})
                        with lock:
                            stats["datasets"] += 1
                            stats["errors"] += 1
                        continue
                    try:
                        members, declared, total = read_directory(kg, url)
                    except Unreadable:
                        with lock:
                            stats["deferred"] += 1
                        continue
                    except zipindex.BadZip as exc:
                        _record(bundle_log, base | {"state": "no_bundle",
                                                    "message": str(exc)[:200]})
                        with lock:
                            stats["datasets"] += 1
                            stats["no_bundle"] += 1
                        continue
                    recorded, n_probe, n_kept = select_members(
                        members, cap, record)
                    _record(bundle_log, base | {
                        "state": "ok", "zip_bytes": total,
                        "n_entries": declared, "n_kept": n_kept,
                        # Every member below goes to the index; the first
                        # `n_probed` (a stable prefix of the hash order, see
                        # `select_members`) are the ones this pass described.
                        "n_recorded": len(recorded),
                        "n_probed": n_probe,
                        # The *recording* cap, not the probe cap: this says
                        # the index is missing members, where an unprobed row
                        # is merely a row without verdicts.
                        "capped": n_kept > len(recorded),
                        "signed_at": int(time.time()),
                        "names": [m.name for m in recorded],
                        # Computed here because Polars has no hex formatter.
                        "keys": [content_key(m.crc, m.usize) for m in recorded],
                        "sizes": [m.usize for m in recorded],
                        "crcs": [m.crc for m in recorded],
                        "csizes": [m.csize for m in recorded],
                        "methods": [m.method for m in recorded],
                        "offsets": [m.header_offset for m in recorded],
                    })
                    with lock:
                        stats["datasets"] += 1
                        stats["files"] += len(recorded)
                        stats["probing"] += n_probe
                        stats["files_seen"] += n_kept
                        stats["capped"] += int(n_kept > len(recorded))
                        emit("repo", done=stats["datasets"], of=len(todo),
                             note=f"{stats['files']:,} indexed · "
                                  f"{stats['probed']:,}/{stats['probing']:,} probed",
                             name=ref)
                    if not _queue_heads(url, ref, row, recorded[:n_probe]):
                        return
            except BaseException as exc:                # noqa: BLE001
                crash.append(exc)
            finally:
                with lock:
                    live["sign"] -= 1
                    last = live["sign"] <= 0
                if last:
                    signing.set()

        def _head_worker() -> None:
            try:
                while not stopping.is_set():
                    try:
                        url, ref, row, member = head_work.get(timeout=_STALL)
                    except queue.Empty:
                        if signing.is_set() and head_work.empty():
                            return
                        continue
                    target = member_row(ref, row, member)
                    head = _head_of(kg, url, member, head_bytes)
                    if head is None:
                        record = probe.describe(target, None, None, "kaggle")
                        with lock:
                            stats["head_failures"] += 1
                    else:
                        record = probe.describe(target, head, 206, "kaggle",
                                                store, which, pool)
                        with lock:
                            stats["probed"] += 1
                    # The index needs the archive coordinates, and the heads
                    # log is the only per-file record, so they ride along with
                    # the verdicts rather than forcing a join at compact time.
                    record |= {
                        "full_name": ref, "path": member.name,
                        "crc": member.crc, "csize": member.csize,
                        "method": member.method,
                        "header_offset": member.header_offset,
                        "version": row.get("version"),
                        "license": row.get("license"),
                        "owner": ref.split("/")[0],
                    }
                    _record(heads_log, record)
            except BaseException as exc:                # noqa: BLE001
                crash.append(exc)
            finally:
                with lock:
                    live["head"] -= 1

        threads = [threading.Thread(target=_sign_worker, args=(route,),
                                    daemon=True)
                   for route in routes]
        threads += [threading.Thread(target=_head_worker, daemon=True)
                    for _ in range(head_workers)]
        run_workers(threads, stats, stopping=stopping, accepting=accepting,
                    lock=lock)

    if store is not None:
        # Closed before it is measured: `bytes_written` stats the shard files.
        store.close()
        stats["cache_bytes"] = store.bytes_written
    stats["ranges"] = kg.ranges
    stats["range_bytes"] = kg.range_bytes
    # What the metered pool spent, beside the dataset count.
    stats |= getattr(kg, "stats", {})
    if crash and not stats.get("interrupted"):
        raise crash[0]
    return stats
