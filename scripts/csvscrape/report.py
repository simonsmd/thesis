"""Live terminal progress for long-running phases.

Progress goes to stderr so stdout stays clean for the final summary. On a tty
the in-shard page counter rewrites one line; when redirected it is dropped
entirely, leaving one durable line per shard so a piped log stays readable.

Waits are drawn *onto* that rewritten line rather than replacing it, so a
wait is visible exactly while it is happening, the status line survives it,
and a throttled run's scrollback is not a wall of `holding 1s`. A wait long
enough to look like a stall (`_DURABLE_WAIT`) also gets a durable line, which
still reads correctly hours later in a piped log.
"""
from __future__ import annotations

import collections
import re
import sys
import time
from typing import Any

# Long enough to be legible, short enough that lines stay under 80 columns.
_QUERY_TRIM = 22
_EXTENSION = re.compile(r"extension:\w+ ")


def _short(query: str) -> str:
    """`extension:csv size:357888..358143` -> `357888..358143`.

    The extension is stripped because it is shown in its own column, and the
    size range is the part that differs between shards.
    """
    if "gitattributes" in query:
        return ".gitattributes lfs"
    trimmed = _EXTENSION.sub("", query).replace("size:", "")
    if len(trimmed) > _QUERY_TRIM:
        trimmed = trimmed[: _QUERY_TRIM - 1] + "…"
    return trimmed


# A wait at least this long gets a durable line as well as the overlay: long
# enough that the run looks hung, and long enough to be worth finding again in
# a log. Everything shorter is ordinary pacing or a brief server hold, which
# the overlay already shows while it is happening.
_DURABLE_WAIT = 30.0

# How often the fetch rate gets a durable line of its own. The transient status
# line carries it live, but a pass runs for days and the history -- climbing,
# sawtoothing against a ceiling, pinned at the floor -- is only visible in
# durable lines.
_HEARTBEAT = 300.0

# How far back the *displayed* rate looks. The ETA divides the work left by the
# average over the whole run, which predicts a remainder of mixed small and
# large files better than any recent slice. But that average barely moves when
# throughput collapses late in a long run, so the rate shown on the line and in
# the heartbeat is a trailing window, sized to the heartbeat.
_RATE_WINDOW = _HEARTBEAT
# Seconds over which `probe --resniff` measures its current speed.
_RESNIFF_WINDOW = 10.0


def _fmt_seconds(seconds: float) -> str:
    """Two significant figures in whatever unit a human would use.

    The same function times a two-second backoff and a four-day scan.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


class Reporter:
    """Formats scraper events onto a terminal."""

    def __init__(self, stream: Any = None):
        self.stream = stream or sys.stderr
        self.tty = getattr(self.stream, "isatty", lambda: False)()
        self.started = time.monotonic()
        self._open = False  # a transient line is currently on screen
        self._last = ""     # its text, so an overlay can be drawn over it
        self._waits = self._throttled = 0
        self._held = 0.0
        self._why: dict[str, int] = {}
        # The shard columns, which `_on_shard_start` fills and every line about
        # a shard reads. Initialised so events arriving in another order
        # cannot fail inside the display.
        self.shard = self.tier = self.query = ""
        self._closed = False              # stragglers must not print past close
        self._rates: dict[str, dict] = {}  # last reading per pool, by name
        self._beat = time.monotonic()     # when that was last written durably
        self._done = self._of = 0
        # Where this session's progress counter started, and when. See `_rate`.
        self._done0: int | None = None
        self._t0 = time.monotonic()
        # (monotonic, done) readings inside `_RATE_WINDOW`, for the rate that
        # is shown as against the one the ETA is built on. See `_observe`.
        self._samples: collections.deque[tuple[float, int]] = collections.deque()
        # (monotonic, sniffed) readings of a `probe --resniff` run.
        self._resniff: collections.deque[tuple[float, int]] = collections.deque()
        self._resniff_start = (0.0, 0)

    def __call__(self, event: str, **f: Any) -> None:
        # An interrupted pass leaves daemon workers running after the summary
        # is printed; their late events must not print after it.
        if self._closed:
            return
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(**f)

    # -- line plumbing ----------------------------------------------------

    def _paint(self, text: str) -> None:
        """Draw over the current transient line without remembering the text."""
        if not self.tty:
            return
        self.stream.write("\r\033[K" + text)
        self.stream.flush()
        self._open = True

    def _transient(self, text: str) -> None:
        self._last = text
        self._paint(text)

    def _overlay(self, note: str) -> None:
        """Append to the status line for as long as something else is happening.

        `_last` is deliberately not updated, so a second overlay replaces the
        first instead of accumulating, and the next real status event repaints
        the line clean.
        """
        self._paint(self._last + note)

    def _line(self, text: str) -> None:
        if self._open:
            self.stream.write("\r\033[K")
            self._open = False
        self.stream.write(text + "\n")
        self.stream.flush()
        # Put the status back underneath: a durable line is written during a
        # phase, not between phases.
        if self._last:
            self._paint(self._last)

    # -- events -----------------------------------------------------------

    def _on_start(self, retired: int = 0, partial: int = 0, known: int = 0, **_: Any) -> None:
        self._line(
            f"discovery: {known:,} repos known · "
            f"{retired:,} shards retired · {partial:,} partial"
        )

    def _on_shard_start(self, index: int = 0, of: int | None = None, tier: str = "",
                        query: str = "", resume: int = 0, ext: str = "",
                        **_: Any) -> None:
        self.shard = f"[{index}/{of}]" if of else f"[{index}]"
        # Both are worth seeing: the tier says which repos, the extension which
        # dialect.
        self.tier = f"{ext:3s} {tier:4s}" if ext else f"{tier:8s}"
        self.query = _short(query)
        note = f" resuming at page {resume + 1}" if resume else ""
        self._transient(f"  {self.shard} {self.tier} {self.query}{note} …")

    def _on_page(self, page: int = 0, hits: int = 0, new: int = 0,
                 total: int = 0, **_: Any) -> None:
        self._transient(
            f"  {self.shard} {self.tier} {self.query}  "
            f"p{page} {hits:>3} hits +{new:<3} ({total:,} total)"
        )

    def _on_shard_end(self, state: str = "", reason: str = "", pages: int = 0,
                      total_count: int | None = None, total: int = 0, **_: Any) -> None:
        mark = "✓" if state == "done" else "→"
        available = f"{total_count:,}" if total_count is not None else "?"
        # The shard is over, so its page counter must not be repainted under
        # the line that retires it.
        self._last = ""
        self._line(
            f"  {mark} {self.shard} {self.tier} {self.query:<{_QUERY_TRIM}} "
            f"p{pages} of {available:>9}  {reason:<11} {total:,} repos"
        )

    def _on_shard_error(self, message: str = "", **_: Any) -> None:
        self._last = ""
        self._line(f"  ✗ {self.shard} {self.tier} {self.query}  {message[:44]}")

    def _on_wait(self, seconds: float = 0.0, reason: str = "",
                 throttled: bool = True, **_: Any) -> None:
        # ⏳ is this program pacing itself at the rate it chose; ⏸ is the server
        # having refused something. A screen of ⏳ is a healthy run at its
        # ceiling; a screen of ⏸ is the thing to go and fix.
        self._waits += 1
        self._held += seconds
        mark = "⏸" if throttled else "⏳"
        if seconds >= _DURABLE_WAIT:
            self._line(f"  {mark}  {reason}: holding {_fmt_seconds(seconds)}")
        else:
            self._overlay(f"  {mark} {reason} {_fmt_seconds(seconds)}")
        if throttled:
            self._throttled += 1
            # Which limit, not just how many: different limits call for
            # different fixes, and the overlays that showed them are gone.
            self._why[reason] = self._why.get(reason, 0) + 1

    def _on_repo(self, done: int = 0, of: int = 0, note: str = "",
                 name: str = "", **_: Any) -> None:
        self._done, self._of = done, of
        self._observe(done)
        rate, shown = self._rate(done), self._recent_rate()
        # An unmeasured rate renders as "eta ?", not "eta 0s".
        eta = (f"eta {_fmt_seconds((of - done) / rate)}"
               if rate > 0 and of else "eta ?")
        # The busiest pool's request rate rides on this line; the heartbeat
        # names them all.
        busy = max(self._rates.values(), key=lambda r: r["achieved"],
                   default=None)
        now = f" · {busy['achieved']:.0f}/s" if busy else ""
        # The pass names its own totals.
        note = f"{note} · " if note else ""
        # `files/h` rather than a bare `/h`: the request rate beside it counts
        # something else, and one request can carry a thousand files.
        self._transient(
            f"  [{done:,}/{of:,}] {note}"
            f"{shown * 3600:,.0f} files/h{now} · {eta}  {name[:24]}"
        )

    def _observe(self, done: int) -> None:
        """Record a progress reading. Only the progress line advances this.

        The heartbeat reads both rates without contributing to either, so its
        five-minute cadence cannot thin out the window it is quoting.
        """
        now = time.monotonic()
        if self._done0 is None:
            self._done0, self._t0 = done, now
        self._samples.append((now, done))
        cutoff = now - _RATE_WINDOW
        while len(self._samples) > 2 and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def _rate(self, done: int) -> float:
        """Items per second over the whole session. The ETA's estimator.

        Measured from this session's first reading, because a counter may not
        start at zero: `probe` reports its position in the whole pool.
        """
        if self._done0 is None:
            return 0.0
        return (done - self._done0) / max(time.monotonic() - self._t0, 1e-6)

    def _recent_rate(self) -> float:
        """Items per second over the last `_RATE_WINDOW`. What gets displayed.

        Short by design: this is the number that has to move when the run does.
        Until the window fills it spans whatever has been seen, which makes it
        the cumulative rate early on -- the same figure, arrived at honestly.

        The window ends *now*, not at the last sample: only progress advances
        the samples, so dividing by the time actually elapsed is what lets the
        rate decay toward zero during a stall instead of freezing.
        """
        if len(self._samples) < 2:
            return 0.0
        then, before = self._samples[0]
        after = self._samples[-1][1]
        return (after - before) / max(time.monotonic() - then, 1e-6)

    # Pools shown in the unit their limit is stated in, so the number can be
    # checked against the documented one.
    _HOURLY = ("core", "graphql", "kaggle_rest", "kaggle_rpc")

    _PER_MINUTE = ("kaggle_list",)

    @classmethod
    def _pool_note(cls, name: str, r: dict) -> str:
        """Each pool in the unit its own limit is stated in."""
        if name in cls._HOURLY:
            return (f"{name} {r['achieved'] * 3600:,.0f}/h "
                    f"of {r['permitted'] * 3600:,.0f}")
        if name in cls._PER_MINUTE:
            return (f"{name} {r['achieved'] * 60:,.0f}/m "
                    f"of {r['permitted'] * 60:,.0f}")
        return f"{name} {r['achieved']:.1f}/s of {r['permitted']:.1f}"

    def _on_rate(self, permitted: float = 0.0, achieved: float = 0.0,
                 wall: float | None = None, throttled: int = 0,
                 requests: int = 0, pool: str = "raw", **_: Any) -> None:
        """A closed measurement window from the fetch pool.

        Both rates are shown because the gap between them is the diagnosis. If
        `achieved` tracks `permitted`, the pacer is the constraint. If it sits
        well below, something inside the pass is the constraint instead -- too
        few threads, or the sniff stage -- and raising the permitted rate will
        do nothing.
        """
        self._rates[pool] = {"permitted": permitted, "achieved": achieved,
                             "wall": wall, "throttled": throttled,
                             "requests": requests}
        now = time.monotonic()
        if now - self._beat < _HEARTBEAT:
            return
        self._beat = now
        elapsed = now - self.started
        eta = speed = ""
        # The same measurements the progress line makes; read-only here, since
        # the line is what advances them.
        rate = self._rate(self._done)
        shown = self._recent_rate()
        if shown > 0:
            # File throughput, which the request rates below cannot give. This
            # is the only throughput figure in a piped log (the progress line
            # is dropped without a tty), so the window is named: it does not
            # divide into the ETA beside it.
            speed = (f" · {shown * 3600:,.0f} files/h"
                     f" ({_RATE_WINDOW / 60:.0f}m)")
        if self._of and rate > 0:
            eta = f" · eta {_fmt_seconds((self._of - self._done) / rate)}"
        pools = " · ".join(self._pool_note(name, r)
                           for name, r in sorted(self._rates.items()))
        bad = sum(r["throttled"] for r in self._rates.values())
        worst = max((r["wall"] for r in self._rates.values() if r["wall"]),
                    default=None)
        at = f" · wall {worst:.1f}/s" if worst else ""
        total = sum(r["requests"] for r in self._rates.values())
        self._line(f"  ⟳ {_fmt_seconds(elapsed)}{speed} · {total:,} requests · "
                   f"{pools} · {bad:,} rejected{at}{eta}")

    def _on_plan(self, strata: int = 0, files: int = 0, bytes: int = 0,
                 mode: str = "", dataset: str = "", selector: str = "",
                 **_: Any) -> None:
        # The projected size is worth seeing before the download, not after: a
        # log-uniform draw's equal count per octave puts almost all the bytes
        # in the top bucket, which is not obvious from the file count.
        from .index import human

        shape = (f"across {strata} size buckets" if strata > 1
                 else "drawn uniformly from the pool")
        self._line(f"plan [{dataset}]: {files:,} files {shape} · "
                   f"{mode} over {selector!r} · ~{human(bytes)} projected")

    def _on_sniff(self, done: int = 0, total: int = 0, **_: Any) -> None:
        # Transient: what the sniffers could not settle is written to
        # <name>.review.toml.
        self._transient(f"  sniffing [{done:,}/{total:,}]")

    def _on_resniff(self, done: int = 0, sniffed: int = 0, not_cached: int = 0,
                    **_: Any) -> None:
        # Two speeds, in sniffed heads per second: over the last few seconds,
        # which shows what the run is doing right now, and since the start,
        # which is what the remaining time depends on.
        now = time.monotonic()
        if not self._resniff:
            self._resniff_start = (now, sniffed)
        self._resniff.append((now, sniffed))
        while len(self._resniff) > 2 and self._resniff[0][0] < now - _RESNIFF_WINDOW:
            self._resniff.popleft()

        def speed(since: tuple[float, int]) -> float:
            return (sniffed - since[1]) / max(now - since[0], 1e-6)

        self._transient(
            f"  sniffing again: {done:,} records read · {sniffed:,} sniffed · "
            f"{not_cached:,} not in the cache · {speed(self._resniff[0]):,.0f}/s "
            f"now · {speed(self._resniff_start):,.0f}/s average")

    def _on_round(self, n: int = 0, wanted: int = 0, drawn: int = 0, **_: Any) -> None:
        self._line(f"round {n}: need {wanted:,} more, drew {drawn:,} candidates")

    def _on_download(self, kept: int = 0, target: int = 0, rejected: int = 0,
                     state: str = "", reason: str = "", name: str = "", **_: Any) -> None:
        if state != "ok":
            # Rejections are the point of the exercise, so they get a durable
            # line each rather than scrolling past in the transient counter.
            self._line(f"  ✗ {reason:<26} {name[:40]}")
            return
        self._transient(f"  [{kept:,}/{target:,}] kept · {rejected:,} rejected  {name[:34]}")

    def close(self) -> None:
        self._closed = True
        if self._open:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._open = False
        # The overlays are gone, so the durable summary of waiting is said
        # here: how long, and how much of it the server asked for.
        if self._waits:
            self._last = ""
            why = ", ".join(f"{n} {reason}" for reason, n
                            in sorted(self._why.items(), key=lambda kv: -kv[1]))
            # Backoffs are congestion *events*; one event can reject many
            # requests already in flight, so the two counts differ.
            rejected = sum(r["throttled"] for r in self._rates.values())
            counts = f"{self._throttled:,} backoffs"
            if rejected:
                counts += f" from {rejected:,} rejected requests"
            self._line(
                f"waited {_fmt_seconds(self._held)} over {self._waits:,} holds · "
                f"{counts}" + (f" ({why})" if why else "")
            )
