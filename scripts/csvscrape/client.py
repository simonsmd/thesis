"""Rate-limited GitHub client.

GitHub meters each budget independently, and each gets its own limiter:

  * `search/code`: 10/min, billed to `code_search` (not `search`). Paced by a
    `Pacer`, because a 10-per-minute window cannot absorb a burst.
  * the REST core pool, ~5000/h: a `Bucket`.
  * GraphQL, 5000 points/h, one point per query: a `Bucket` with an in-flight
    cap.
  * raw.githubusercontent.com and media.githubusercontent.com: a per-IP limit
    that states itself only in a 429, discovered at runtime by `RawPool`.
"""
from __future__ import annotations

import contextlib
import json
import random
import threading
import time
from typing import Any

import requests

from . import config

API = "https://api.github.com"
GRAPHQL = "https://api.github.com/graphql"
# U+FFFD, the marker a UTF-8 decode failure leaves in `Blob.text`.
REPLACEMENT = b"\xef\xbf\xbd"
RAW = "https://raw.githubusercontent.com"
MEDIA = "https://media.githubusercontent.com/media"


class Congestion:
    """Whether a refusal opens a new congestion event or joins one in flight.

    With several workers sharing a limiter, one blown budget is answered with
    a rejection to every request already on the wire. Without this each of
    them would announce its own backoff, and one event would be reported once
    per worker.

    Only the *reporting* is deduplicated. The hold itself is still applied by
    every caller, because `hold_for` takes the maximum and a straggler's
    longer `retry-after` must not be discarded.

    The window runs past the hold by `API_LATENCY` so the rejections handed to
    requests already in flight count against the event that caused them.
    """

    def __init__(self) -> None:
        self._until = 0.0
        self._lock = threading.Lock()

    def first(self, wait: float) -> bool:
        with self._lock:
            now = time.monotonic()
            if now < self._until:
                return False
            self._until = now + wait + config.API_LATENCY
            return True


class Bucket:
    """Token bucket that also honours the limits the server reports back.

    `capacity` is the burst allowance, deliberately decoupled from the rate: a
    bucket sized at the full hourly quota and starting full does no pacing at
    all until it drains, and would do so again on every restart. A handful of
    tokens absorbs jitter without hiding the rate.

    `max_inflight` bounds how many requests may be outstanding at once, which
    is a different quantity from how often they may start.
    """

    def __init__(self, per_seconds: float, rate_per_period: float,
                 capacity: float = 8.0, on_event: Any = None,
                 name: str = "core", max_inflight: int | None = None):
        self.rate = rate_per_period / per_seconds
        self.capacity = capacity
        self._tokens = capacity
        self._updated = time.monotonic()
        self._until = 0.0  # server-imposed hold
        self._held = False  # that hold has not been waited out yet
        self.on_event = on_event or _silent
        self.name = name
        self.event = Congestion()
        self._sem = threading.Semaphore(max_inflight) if max_inflight else None
        self._win_start = time.monotonic()
        self._win_count = 0
        self._taken = 0
        # Requests the server refused for a rate limit, cumulative.
        self._refused = 0
        self._lock = threading.Lock()

    def refused(self) -> None:
        """One request rejected for a rate limit.

        Counted per rejected request, not per congestion event: `Congestion`
        deduplicates the announcement, and this count says how many requests
        the events cost. `RawPool` counts the same way.

        Only the caller can tell a throttle from an ordinary refusal -- a 403 is
        also how the API answers a repo the token may not read -- so the caller
        decides when to call this.
        """
        with self._lock:
            self._refused += 1

    def inflight(self) -> Any:
        """The concurrency limit, as a context manager. A no-op when unbounded.

        Held around the request itself and released before the caller does
        anything with the body, so the slot measures time on the wire.
        """
        return self._sem if self._sem is not None else contextlib.nullcontext()

    def _refill(self, now: float) -> None:
        """Credit elapsed time to the bucket. Caller holds the lock.

        The elapsed term is clamped at zero because `_updated` is parked at the
        *end* of a hold while one is in force (see `_park`), so a response
        landing mid-hold would otherwise compute a negative interval and take
        tokens away.
        """
        self._tokens = min(
            self.capacity,
            self._tokens + max(0.0, now - self._updated) * self.rate,
        )
        self._updated = max(self._updated, now)

    def _park(self, wait: float) -> None:
        """Extend the hold, and stop the clock for its duration. Caller holds
        the lock.

        A hold is the server saying the budget is spent, so the bucket must not
        earn tokens through it. If it did, the hold would end with a full burst
        handed to every queued worker at once, firing straight back into the
        limit -- which GitHub answers with longer and longer holds.
        """
        now = time.monotonic()
        self._refill(now)
        self._until = max(self._until, now + wait)
        self._updated = self._until
        self._held = True

    @property
    def stats(self) -> dict[str, Any]:
        """What this pool spent, for the run summary.

        Keyed by the pool's name so several pools merge without colliding. The
        throttle count travels with the request count because the pair is the
        reading: N requests means something different with 0 refused than with
        N/20.
        """
        return {f"{self.name}_requests": self._taken,
                f"{self.name}_throttled": self._refused}

    def sync(self, remaining: int, reset: float) -> None:
        """Adopt the server's view of the budget.

        Called on every response, off `x-ratelimit-remaining`, so a shared
        token's spending elsewhere shows up here instead of as a 403 later.

        Refills before it clamps: stamping `_updated` without crediting the time
        since the last refill would discard every request's round trip from
        the budget, pacing at interval + latency instead of interval.
        """
        with self._lock:
            self._refill(time.monotonic())
            self._tokens = min(self._tokens, float(max(remaining, 0)))
            if remaining <= 0:
                self._park(max(0.0, reset - time.time()))

    @property
    def min_interval(self) -> float:
        """Spacing at the sustained rate, ignoring the burst allowance."""
        return 1.0 / self.rate

    def take(self, on_wait: Any = None) -> None:
        """Block until a request may start.

        `on_wait(seconds, throttled)` is called once, before the first sleep,
        so a wait is on screen while it is happening.

        `throttled` distinguishes serving a hold the server imposed from
        pacing at our own rate. The caller also uses it to avoid announcing a
        hold twice: holds are announced where they are applied. The flag is
        consumed by whichever caller clears the hold, so with several threads
        one attributes the wait and the rest read as pacing.
        """
        announced = False
        closed = None
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._until:
                    self._refill(now)
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        self._held = False
                        self._taken += 1
                        self._win_count += 1
                        elapsed = time.monotonic() - self._win_start
                        closed = None
                        if elapsed >= config.API_RATE_WINDOW:
                            closed = {"pool": self.name, "wall": None,
                                      "throttled": self._refused,
                                      "requests": self._taken,
                                      "permitted": self.rate,
                                      "achieved": self._win_count / elapsed}
                            self._win_start = time.monotonic()
                            self._win_count = 0
                        break
                    wait = max((1.0 - self._tokens) / self.rate, 0.01)
                else:
                    wait = self._until - now
                held = self._held
            if on_wait and not announced:
                announced = True
                on_wait(wait, held)
            time.sleep(min(wait, 60.0))
        if closed is not None:
            self.on_event("rate", **closed)

    def hold_until(self, epoch: float, floor: float = 0.0) -> float:
        """Park every caller until a server-supplied reset time.

        Returns the wait actually applied. `floor` covers a server that says
        the budget is spent but hands back a reset that has already elapsed (at
        every window boundary, and under clock skew); without it the retry
        would fire immediately into the same blown limit.
        """
        wait = max(0.0, epoch - time.time())
        if wait < floor:
            wait = floor
        with self._lock:
            self._park(wait)
        return wait

    def hold_for(self, seconds: float) -> None:
        with self._lock:
            self._park(seconds)


class Pacer:
    """Fixed minimum spacing between requests, plus server-imposed holds.

    Used for search instead of a `Bucket`. Its whole state is one instant,
    `_next`, so pacing and holds are the same variable and there is no credit
    for a hold to leave behind -- a 10-per-minute window cannot afford a burst.

    Sizing lives in `config.CODE_SEARCH_INTERVAL`. The interval is fixed: the
    `retry-after` refusals seen in practice came from a second consumer of the
    shared token, which `sync` handles, not from a hidden limit below 10/min.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self._next = 0.0
        self._held = False  # `_next` was last moved by the server, not by us
        self.event = Congestion()
        self._refused = 0
        self._lock = threading.Lock()

    @property
    def stats(self) -> dict[str, Any]:
        """Only the throttle count: `search_code`'s caller counts pages."""
        return {"search_throttled": self._refused}

    def refused(self) -> None:
        """One request rejected for a rate limit. See `Bucket.refused`."""
        with self._lock:
            self._refused += 1

    def inflight(self) -> Any:
        """Unbounded: the spacing already keeps search to one request at a
        time. Present so `_request` can treat both limiters alike."""
        return contextlib.nullcontext()

    @property
    def min_interval(self) -> float:
        return self.interval

    def take(self, on_wait: Any = None) -> None:
        """Block until a request may start. See `Bucket.take` for `on_wait`.

        Pacing and holds share `_next`, so `_held` is what recovers which of the
        two a wait is. A hold shorter than the spacing already owed leaves
        `_next` untouched and still sets the flag, so the wait is attributed to
        the server -- the conservative way round.
        """
        announced = False
        while True:
            with self._lock:
                now = time.monotonic()
                if now >= self._next:
                    # From the start of this request, not its completion: the
                    # limit is on how often we call, and charging the call's
                    # own latency against the interval would pace slower than
                    # the budget allows.
                    self._next = now + self.interval
                    self._held = False
                    return
                wait = self._next - now
                held = self._held
            if on_wait and not announced:
                announced = True
                on_wait(wait, held)
            time.sleep(min(wait, 60.0))

    def hold_until(self, epoch: float, floor: float = 0.0) -> float:
        """Park every caller until a server-supplied reset time.

        Returns the wait actually applied. See `Bucket.hold_until` for `floor`.
        """
        wait = max(0.0, epoch - time.time())
        if wait < floor:
            wait = floor
        with self._lock:
            self._next = max(self._next, time.monotonic() + wait)
            self._held = True
        return wait

    def hold_for(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)
            self._held = True

    def sync(self, remaining: int, reset: float) -> None:
        """Adopt the server's view of the budget.

        A restart lands mid-window with whatever the previous process (or
        another machine sharing the token) left. If the requests left cannot
        last until the reset at our spacing, wait the window out: that forfeits
        at most a handful of requests, against a hold plus rejected requests for
        guessing wrong.
        """
        left = max(0.0, reset - time.time())
        if remaining * self.interval < left:
            self.hold_until(reset)


class RawPool:
    """Adaptive rate control for the hosts whose limit cannot be read.

    raw and media state no budget until they refuse, so this pool finds it:
    additive increase while requests succeed, a multiplicative cut on a 429
    (AIMD). That settles just below the limit and follows it when it moves,
    which a per-IP edge limit does.

    The rate is shared: one 429 parks every worker via the shared pacer.
    Backing off per thread instead lets the others keep hitting an edge that is
    already refusing them, and since refusals are charged against the budget
    that is congestion collapse. Concurrency is a ceiling here, not a
    throughput lever: requests leave at the pool's rate whatever the thread
    count.
    """

    def __init__(self, on_event: Any = None, rate: float | None = None,
                 max_inflight: int | None = None, name: str = "raw"):
        self.on_event = on_event or _silent
        self.name = name
        self._rate = rate if rate is not None else config.RAW_START_RATE
        self.pacer = Pacer(1.0 / self._rate)
        self._sem = threading.Semaphore(max_inflight or config.RAW_MAX_INFLIGHT)
        self._lock = threading.Lock()
        self._wins = 0
        # Rate at which the last 429 arrived, or None while none has. This is
        # what separates slow start from congestion avoidance in `reward`.
        self._wall: float | None = None
        # One reduction per congestion event, not per rejected request: a
        # single blown budget answers every request in flight with a 429, and
        # cutting once per 429 would take the rate to the floor for one event.
        # Later 429s inside the window are counted and waited out only.
        self._event_until = 0.0
        # What the pool is actually achieving, which is not what it permits.
        self._achieved = self._rate
        self._win_start = time.monotonic()
        self._win_count = 0
        self.stats: dict[str, Any] = {
            "raw_requests": 0, "raw_throttled": 0,
            # The server's own `retry-after`, the most direct statement of the
            # limit this pool ever gets.
            "raw_retry_after": None,
            "raw_rate_min": self._rate, "raw_rate_max": self._rate,
            "raw_rate_final": self._rate, "raw_rate_wall": None,
            # Off a 429 only: a served response carries no x-ratelimit-*
            # header at all on these hosts.
            "raw_limit_resource": None, "raw_limit_rate": None,
        }

    @property
    def rate(self) -> float:
        return self._rate

    @contextlib.contextmanager
    def slot(self) -> Any:
        """Pace, then hold one of the in-flight slots for the request."""
        self.pacer.take(self._announce)
        with self._sem:
            # Captured under the lock, reported outside it: `on_event` reaches
            # a terminal and a slow write must never hold up every worker.
            closed = None
            with self._lock:
                self.stats["raw_requests"] += 1
                self._win_count += 1
                elapsed = time.monotonic() - self._win_start
                if elapsed >= config.RAW_RATE_WINDOW:
                    self._achieved = self._win_count / elapsed
                    self._win_start = time.monotonic()
                    self._win_count = 0
                    closed = {"pool": self.name,
                              "permitted": self._rate, "achieved": self._achieved,
                              "wall": self._wall,
                              "throttled": self.stats["raw_throttled"],
                              "requests": self.stats["raw_requests"]}
            if closed is not None:
                self.on_event("rate", **closed)
            yield

    def _announce(self, seconds: float, throttled: bool) -> None:
        # A throttle is announced by `penalise`, where it is applied; waiting it
        # out here would report it twice.
        if seconds >= 1.0 and not throttled:
            self.on_event("wait", seconds=seconds, reason="raw pacing",
                          throttled=False)

    def reward(self) -> None:
        """One clean response. Raises the rate, fast or slowly.

        The limit behaves as a budget with a burst allowance rather than a
        fixed rate, so a single-speed AIMD converges and then keeps sawtoothing
        into the wall to check whether it moved. Two regimes instead: before
        the first 429 (`_wall is None`) climb fast, since the burst is there to
        be spent; afterwards, climb `RAW_PROBE_PATIENCE` times slower -- but
        only above `RAW_PROBE_MARGIN` of the last wall. Recovering to a rate
        that was just served is not a probe and should not be paced like one.
        """
        with self._lock:
            self._wins += 1
            # Successes worth roughly a second at the current rate.
            need = self._rate
            if self._wall is not None and self._rate >= self._wall * config.RAW_PROBE_MARGIN:
                need *= config.RAW_PROBE_PATIENCE
            if self._wins < need:
                return
            self._wins = 0
            self._set_rate(self._rate + config.RAW_RATE_STEP)

    def penalise(self, retry_after: Any = None, headers: Any = None) -> None:
        """A 429. Cut the rate and park every worker, not just this one."""
        now = time.monotonic()
        with self._lock:
            self.stats["raw_throttled"] += 1
            if headers:
                # Lowercased rather than trusting the mapping to be
                # case-insensitive: `requests` makes it so, nothing else does.
                seen = {str(k).lower(): v for k, v in headers.items()}
                for header, field in (("x-ratelimit-resource", "raw_limit_resource"),
                                      ("x-ratelimit-rate", "raw_limit_rate")):
                    if seen.get(header):
                        self.stats[field] = seen[header]
            wait = 0.0
            if retry_after:
                try:
                    wait = float(retry_after)
                    self.stats["raw_retry_after"] = wait
                except (TypeError, ValueError):
                    wait = 0.0
            first = now >= self._event_until
            if first:
                self._wins = 0
                # Cut from what was being achieved, not from what was being
                # permitted: when something else is the constraint the permitted
                # rate is above the real one, and cutting it changes nothing on
                # the wire. The achieved rate at the moment the limit bit is
                # also the best available measurement of the limit.
                base = min(self._rate, self._achieved)
                self._wall = base
                self.stats["raw_rate_wall"] = base
                self._set_rate(base * config.RAW_RATE_BACKOFF)
            # Never shorter than the spacing the reduced rate implies, and
            # never shorter than RAW_MIN_HOLD: a retry inside the throttle
            # cannot succeed *and* is charged to the budget being waited for.
            wait = max(wait, 1.0 / self._rate, config.RAW_MIN_HOLD)
            if first:
                # The event lasts until the hold clears and the requests already
                # on the wire have landed; their 429s belong to this event.
                self._event_until = now + wait + config.RAW_LATENCY
        if first:
            self.on_event("wait", seconds=wait, reason="raw throttle",
                          throttled=True)
        self.pacer.hold_for(wait)

    def _set_rate(self, rate: float) -> None:
        """Caller holds the lock."""
        self._rate = min(config.RAW_MAX_RATE, max(config.RAW_MIN_RATE, rate))
        self.pacer.interval = 1.0 / self._rate
        self.stats["raw_rate_min"] = min(self.stats["raw_rate_min"], self._rate)
        self.stats["raw_rate_max"] = max(self.stats["raw_rate_max"], self._rate)
        self.stats["raw_rate_final"] = self._rate


def _blob_body(blob: dict[str, Any]) -> bytes | None:
    """One `Blob` node's bytes, or None when the caller must use raw instead."""
    text = blob.get("text")
    if text is None or blob.get("isBinary"):
        return None
    body = text.encode("utf-8")
    # GitHub cuts the byte stream at GRAPHQL_TEXT_CAP and then decodes, so a cut
    # through a multi-byte character leaves a trailing U+FFFD. Stripping it
    # makes the body a byte-exact prefix of the file.
    if blob.get("isTruncated") and body.endswith(REPLACEMENT):
        body = body[:-len(REPLACEMENT)]
    # Anywhere else U+FFFD means the bytes were not valid UTF-8 and came back
    # re-encoded, which a length check cannot always catch. A file that really
    # contains U+FFFD is refetched over raw for nothing, which is the right way
    # round to be wrong.
    if REPLACEMENT in body:
        return None
    if blob.get("isTruncated"):
        # A truncated body does not land exactly on the cap: the replacement
        # stripped above can be 1-2 bytes off the partial character it stood
        # for. The window allows that and still rejects the empty string a
        # too-large blob returns.
        ok = abs(len(body) - config.GRAPHQL_TEXT_CAP) <= 8
    else:
        ok = len(body) == blob.get("byteSize")
    return body if ok else None


def _silent(event: str, **fields: Any) -> None:
    """Default event sink: report nothing."""


class GitHub:
    def __init__(self, token: str | None = None, user_agent: str = "csv-corpus-scraper",
                 on_event: Any = _silent, **_ignored: Any):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token or config.token()}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": user_agent,
            }
        )
        self.anon = requests.Session()
        self.anon.headers.update({"User-Agent": user_agent})
        # urllib3 pools ten connections per host by default and quietly discards
        # the rest, which would cost keep-alive on exactly the hottest path.
        for session, size in ((self.session, 16), (self.anon, config.RAW_MAX_INFLIGHT)):
            adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=size)
            session.mount("https://", adapter)
        self.search_pacer = Pacer(config.CODE_SEARCH_INTERVAL)
        self.core_bucket = Bucket(3600.0, config.CORE_PER_HOUR * config.RATE_SAFETY,
                                  capacity=config.CORE_BURST,
                                  on_event=on_event, name="core")
        self.on_event = on_event
        self.raw = RawPool(on_event=self.on_event)
        self.graphql_bucket = Bucket(3600.0,
                                     config.GRAPHQL_PER_HOUR * config.RATE_SAFETY,
                                     capacity=config.GRAPHQL_BURST,
                                     max_inflight=config.GRAPHQL_MAX_INFLIGHT,
                                     on_event=on_event, name="graphql")
    # -- plumbing ---------------------------------------------------------

    def _request(self, method: str, url: str, limiter: Bucket | Pacer,
                 timeout: float = 60, **kw: Any) -> requests.Response | None:
        """The response, or None when every attempt failed to produce one."""
        resp = None
        for attempt in range(6):
            # Our own pacing is announced too, so a run spending its budget as
            # intended is distinguishable from one stalled on refusals. Holds
            # are announced where they are applied, below, and sub-second waits
            # are noise.
            limiter.take(
                lambda seconds, throttled: None if throttled or seconds < 1.0
                else self.on_event("wait", seconds=seconds, reason="pacing",
                                   throttled=False)
            )
            try:
                with limiter.inflight():
                    resp = self.session.request(method, url, timeout=timeout, **kw)
            except requests.RequestException:
                time.sleep(2 ** attempt + random.random())
                continue

            remaining = resp.headers.get("x-ratelimit-remaining")
            reset_hdr = resp.headers.get("x-ratelimit-reset")
            if remaining is not None and reset_hdr is not None:
                # The authoritative reading of the budget, fed in on every
                # response so a shared token's spending elsewhere shows up
                # here. (/rate_limit is not a substitute: it lags the headers.)
                limiter.sync(int(remaining), float(reset_hdr))
            if remaining is not None and int(remaining) == 0:
                reset = float(resp.headers.get("x-ratelimit-reset", time.time() + 60))
                # Back off for real even when the reset has already passed, and
                # grow it per attempt so a persistently stale reset cannot spin.
                wait = limiter.hold_until(reset, floor=min(60.0, 2.0 ** attempt))
                # A 200 that spent the last slot in the window is pacing working,
                # not the server refusing anything.
                refused = resp.status_code in (403, 429)
                if wait >= 1.0 and limiter.event.first(wait):
                    self.on_event(
                        "wait", seconds=wait, throttled=refused,
                        reason="primary rate limit" if refused else "window spent")

            # Refusals are counted in the three branches below rather than for
            # every 403: a 403 is also how the API answers a repo the token may
            # not read, and those are not throttles. Each branch ends in
            # `continue`, so nothing is counted twice; any other 403 falls
            # through to `return resp`.
            if resp.status_code in (403, 429):
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    limiter.refused()
                    # Floored at the pacing interval: GitHub meets a spent
                    # search budget with `retry-after: 1`, and obeying that
                    # literally retries into the same blown limit -- which
                    # cannot succeed and still counts against it.
                    wait = max(float(retry_after), limiter.min_interval)
                    if limiter.event.first(wait):
                        self.on_event("wait", seconds=wait, throttled=True,
                                      reason=f"retry-after {self._limit(resp)}")
                    limiter.hold_for(wait)
                    continue
                if remaining is not None and int(remaining) == 0:
                    limiter.refused()
                    continue  # primary limit; the hold above already applies
                # The search pool answers a blown primary limit with a 403 and
                # no x-ratelimit-* headers at all, so the body is the only
                # signal. Covers both "API rate limit exceeded" and the
                # secondary/abuse limit.
                if b"rate limit" in resp.content.lower():
                    limiter.refused()
                    reset = resp.headers.get("x-ratelimit-reset")
                    if reset:
                        wait = limiter.hold_until(float(reset),
                                                  floor=min(60.0, 2.0 ** attempt))
                        if wait >= 1.0 and limiter.event.first(wait):
                            self.on_event("wait", seconds=wait, throttled=True,
                                          reason=f"rate limit {self._limit(resp)}")
                    else:
                        wait = 60 * (attempt + 1)
                        if limiter.event.first(wait):
                            self.on_event("wait", seconds=wait, throttled=True,
                                          reason=f"rate limit {self._limit(resp)}")
                        limiter.hold_for(wait)
                    continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt + random.random())
                continue
            return resp
        return resp

    @staticmethod
    def _limit(resp: requests.Response) -> str:
        """Which limit refused this, named from what the refusal carries.

        A primary limit is a spent budget and is answered by waiting; a
        secondary limit is priced on concurrency and server work and is
        answered by asking for less at once. GitHub names the secondary limit
        in the body and the pool in `x-ratelimit-resource`.
        """
        try:
            body = resp.content[:400].lower()
        except Exception:
            body = b""
        kind = "secondary" if b"secondary rate limit" in body else "primary"
        resource = resp.headers.get("x-ratelimit-resource")
        return f"{kind}/{resource}" if resource else kind

    @staticmethod
    def _message(resp: requests.Response) -> str:
        """GitHub's error text, without trusting the body to be JSON (a 502
        comes back as an HTML page)."""
        try:
            return str(resp.json().get("message", ""))[:200]
        except ValueError:
            return resp.text[:200]

    # -- endpoints --------------------------------------------------------

    def search_code(self, query: str, page: int = 1, per_page: int = 100,
                    fragments: bool = False) -> dict[str, Any]:
        """Code search. `fragments` asks for text-match snippets.

        Off by default. The secondary rate limit is priced on server work, and
        computing highlight snippets for 100 results is enough work to draw
        `retry-after` refusals at a request rate the primary limit allows.
        """
        accept = ("application/vnd.github.text-match+json" if fragments
                  else "application/vnd.github+json")
        resp = self._request(
            "GET",
            f"{API}/search/code",
            self.search_pacer,
            params={"q": query, "page": page, "per_page": per_page},
            headers={"Accept": accept},
        )
        if resp is None:
            return {"error": 0, "message": "no response after retries"}
        if resp.status_code != 200:
            return {"error": resp.status_code, "message": self._message(resp)}
        try:
            return resp.json()
        except ValueError:
            return {"error": resp.status_code, "message": "malformed json body"}

    def blobs(self, rows: list[dict[str, Any]]) -> list[bytes | None]:
        """Blob contents for many files in one request, parallel to `rows`.

        Each row needs `full_name` and `blob_sha`. An entry comes back None
        when the caller must fall back to a range request:

          * `text` is null: GitHub considers the blob binary (git's heuristic,
            a NUL in the first 8000 bytes), which includes UTF-16 files.
          * the bytes do not account for: `text` is a string, so non-UTF-8
            content can come back re-encoded. `byteSize` is checked against it.
          * the object is missing, or the query erred for that alias.

        Never pass an LFS row. Its git blob is the pointer, which is valid UTF-8
        of exactly its declared size -- it passes every check above and is the
        wrong content. `probe` routes those to media.githubusercontent.com.

        Raises when the request as a whole fails, so the caller can tell a
        failed batch from blobs GitHub declined one by one.
        """
        if not rows:
            return []
        # Grouped by repository: one `repository()` per repo with an `object()`
        # per blob inside it. Repository resolution is the expensive term in a
        # query, so a flat one-repository-per-row query is refused at a few
        # hundred aliases (see config.GRAPHQL_BATCH_REPOS).
        #
        # `nodes(ids: [...])` with derived blob ids would avoid the repository
        # term, but it is capped at 100 ids per field, and aliasing several
        # `nodes` fields was slower and less reliable than this.
        groups: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            groups.setdefault(row["full_name"], []).append(i)
        members = list(groups.values())

        parts = []
        for r, (full_name, idxs) in enumerate(groups.items()):
            owner, _, name = full_name.partition("/")
            inner = " ".join(
                f'b{j}: object(oid:{json.dumps(rows[i]["blob_sha"])}) '
                f'{{ ... on Blob {{ byteSize isBinary isTruncated text }} }}'
                for j, i in enumerate(idxs)
            )
            parts.append(
                f'  r{r}: repository(owner:{json.dumps(owner)},'
                f'name:{json.dumps(name)}) {{ {inner} }}'
            )
        query = "query {\n" + "\n".join(parts) + "\n}"

        # A long timeout: batches are tens of MiB, and a timeout costs the whole
        # batch. Retried once on a 200 whose body will not parse -- a response
        # cut off mid-stream, which `_request` does not see as a failure.
        resp = None
        for attempt in range(2):
            resp = self._request("POST", GRAPHQL, self.graphql_bucket,
                                 timeout=180, json={"query": query})
            if resp is not None and resp.status_code == 200:
                try:
                    resp.json()
                except ValueError:
                    if attempt == 0:
                        continue
            break
        if resp is None:
            raise RuntimeError("graphql: no response after retries")
        if resp.status_code != 200:
            raise RuntimeError(f"graphql: http {resp.status_code} "
                               f"{resp.content[:120]!r}")
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError("graphql: malformed json body") from None
        data = payload.get("data") or {}

        # An over-large batch is answered with HTTP 200, a mostly-null `data`
        # and one RESOURCE_LIMITS_EXCEEDED error per refused alias. Unread, that
        # looks exactly like GitHub declining those blobs on their merits.
        # Raised rather than returned partial so the caller counts a failed
        # batch and every row takes the fallback path.
        refused = sum(1 for e in payload.get("errors") or []
                      if e.get("type") == "RESOURCE_LIMITS_EXCEEDED")
        if refused:
            raise RuntimeError(
                f"graphql: resource limits exceeded on {refused} of {len(rows)} "
                f"blobs across {len(members)} repos -- batch too large")

        out: list[bytes | None] = [None] * len(rows)
        for r, idxs in enumerate(members):
            repo = data.get(f"r{r}") or {}
            for j, i in enumerate(idxs):
                out[i] = _blob_body(repo.get(f"b{j}") or {})
        return out

    def tree(self, full_name: str, ref: str = "HEAD") -> dict[str, Any]:
        """Whole-repo file listing in one call: path, blob sha and exact size.

        `HEAD` saves resolving the default branch first, and costs no pin: the
        response's top-level `sha` is the commit the ref resolved to, which
        `trees` records so later reads address the same snapshot.
        """
        resp = self._request(
            "GET",
            f"{API}/repos/{full_name}/git/trees/{ref}",
            self.core_bucket,
            params={"recursive": "1"},
        )
        if resp is None:
            return {"error": 0, "message": "no response after retries"}
        if resp.status_code != 200:
            return {"error": resp.status_code, "message": resp.text[:200]}
        try:
            return resp.json()
        except ValueError:
            return {"error": resp.status_code, "message": "malformed json body"}

    def repo(self, full_name: str) -> dict[str, Any]:
        resp = self._request("GET", f"{API}/repos/{full_name}", self.core_bucket)
        if resp is None:
            return {"error": 0}
        if resp.status_code != 200:
            return {"error": resp.status_code}
        try:
            return resp.json()
        except ValueError:
            return {"error": resp.status_code}

    # raw and media: paced by `RawPool`, not by an API budget.

    def raw_file(self, full_name: str, ref: str, path: str) -> bytes | None:
        return self._unmetered(f"{RAW}/{full_name}/{ref}/{requests.utils.quote(path)}")

    def lfs_media(self, full_name: str, ref: str, path: str) -> bytes | None:
        """Real content behind an LFS pointer. Billed to the *repo owner's*
        LFS bandwidth quota, so over-quota repos fail here and get backfilled."""
        return self._unmetered(f"{MEDIA}/{full_name}/{ref}/{requests.utils.quote(path)}")

    def stream(self, url: str, sink: Any, chunk: int = 1 << 20) -> tuple[int, str] | None:
        """Stream a response to `sink.write`, returning (bytes, sha256).

        Content is never held in memory: the corpus contains multi-hundred-MB
        files and LFS objects past a gigabyte.

        A retry re-sends the body from byte zero, so whatever a failed attempt
        left in `sink` is discarded first; otherwise the attempts concatenate
        while the digest covers only the last one. A sink that cannot be
        rewound is abandoned (None) rather than appended to.
        """
        import hashlib

        try:
            origin = sink.tell()
        except (AttributeError, OSError):
            origin = None
        written = 0

        for attempt in range(4):
            if written:
                if origin is None:
                    return None
                sink.seek(origin)
                sink.truncate()
                written = 0
            try:
                # The slot is held for the whole body, not just the request:
                # a stream occupies a connection for as long as it runs.
                with self.raw.slot(), self.anon.get(url, timeout=300, stream=True) as resp:
                    if resp.status_code != 200:
                        if resp.status_code in (403, 429):
                            self.raw.penalise(resp.headers.get("retry-after"), resp.headers)
                            continue
                        if resp.status_code >= 500:
                            time.sleep(2 ** attempt + random.random())
                            continue
                        return None
                    self.raw.reward()
                    digest = hashlib.sha256()
                    for block in resp.iter_content(chunk_size=chunk):
                        if not block:
                            continue
                        sink.write(block)
                        digest.update(block)
                        written += len(block)
                    return written, digest.hexdigest()
            except requests.RequestException:
                time.sleep(2 ** attempt + random.random())
        return None

    def head_bytes(self, full_name: str, path: str, n: int,
                   is_lfs: bool = False,
                   ref: str = "HEAD") -> tuple[bytes | None, int | None]:
        """First n bytes of a file, via an HTTP Range request.

        `ref` should be the commit the tree scan saw, so the bytes are those of
        the blob the index names; both hosts accept a commit sha as the ref.

        Returns (bytes, status). The status lets the caller tell a permanent
        failure (404: the path does not resolve at that ref) from a transient
        one (429, 503) worth another pass.

        One request whatever the file size, so a head of a 1.6 GB LFS object
        costs the same as one of a 2 KB file.

        `Accept-Encoding: identity` is required: otherwise the range applies to
        the gzip-encoded representation and returns the wrong number of
        decoded bytes.
        """
        base = MEDIA if is_lfs else RAW
        url = f"{base}/{full_name}/{ref}/{requests.utils.quote(path)}"
        headers = {"Range": f"bytes=0-{n - 1}", "Accept-Encoding": "identity"}
        status = None
        for attempt in range(config.RAW_ATTEMPTS):
            try:
                with self.raw.slot():
                    resp = self.anon.get(url, headers=headers, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            status = resp.status_code
            if status in (200, 206):
                self.raw.reward()
                return resp.content[:n], status
            # A throttle is about the rate every worker shares and parks all of
            # them; a server error is about this request and sleeps only here.
            if status in (403, 429):
                self.raw.penalise(resp.headers.get("retry-after"), resp.headers)
                continue
            if status >= 500:
                time.sleep(2 ** attempt + random.random())
                continue
            return None, status
        return None, status

    def _unmetered(self, url: str) -> bytes | None:
        for attempt in range(config.RAW_ATTEMPTS):
            try:
                with self.raw.slot():
                    resp = self.anon.get(url, timeout=120)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                self.raw.reward()
                return resp.content
            if resp.status_code in (403, 429):
                self.raw.penalise(resp.headers.get("retry-after"), resp.headers)
                continue
            if resp.status_code >= 500:
                time.sleep(2 ** attempt + random.random())
                continue
            return None
        return None
