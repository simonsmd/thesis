"""Small helpers shared by the passes."""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

# Budget for a pass's whole Ctrl-C wind-down, shared by every worker rather
# than spent per thread. Enough for requests that are about to land; what has
# not landed by then is abandoned. Workers are daemon threads, so that costs
# their records and nothing else.
GRACE_SECONDS = 5.0

# How often the main thread wakes while it waits on the workers.
POLL = 0.2

_END = object()


def run_workers(threads: list[threading.Thread], stats: dict[str, Any], *,
                stopping: threading.Event, accepting: threading.Event,
                lock: threading.Lock,
                on_interrupt: Callable[[], None] | None = None,
                on_tick: Callable[[], None] | None = None) -> bool:
    """Start `threads` and wait for them. False if the pass was interrupted.

    Waits on one live thread at a time with a short timeout, so the main
    thread keeps returning to the interpreter and can act on a pending signal.
    Polars starts its own threads on import, a process-directed SIGINT may be
    delivered to one of them, and Python only runs the handler on the main
    thread -- which a bare `join()` leaves asleep in a lock nothing wakes.
    `on_tick` runs after every wake.

    On Ctrl-C (or SIGTERM, which `cli.main` turns into one): `stopping` is set
    so the workers draw no new work, `on_interrupt` runs, and the work already
    in flight gets GRACE_SECONDS in total to land. Then `accepting` is cleared
    and `lock` taken once, which waits out any write already inside it. Workers
    must check `accepting` under `lock` before writing, so a straggler cannot
    write into a log that is being sealed. A second Ctrl-C cuts the grace
    period short rather than unwinding, so the caller still prints its summary.
    `stats` gets `interrupted`, and `abandoned` for a second Ctrl-C.
    """
    for t in threads:
        t.start()
    try:
        while True:
            alive = next((t for t in threads if t.is_alive()), None)
            if alive is None:
                return True
            alive.join(timeout=POLL)
            if on_tick is not None:
                on_tick()
    except KeyboardInterrupt:
        stats["interrupted"] = True
        stopping.set()
        if on_interrupt is not None:
            on_interrupt()
        deadline = time.monotonic() + GRACE_SECONDS
        try:
            for t in threads:
                t.join(timeout=max(0.0, deadline - time.monotonic()))
        except KeyboardInterrupt:
            stats["abandoned"] = True
        accepting.clear()
        with lock:
            pass
        return False


def interleave(streams: dict[Any, Iterable[T]],
               weights: dict[Any, int] | None = None) -> Iterator[T]:
    """Weighted round-robin over `streams`, in their dict order.

    Each round takes `weights[key]` items (default 1) from every stream in
    turn. An exhausted stream drops out and the rest share the rounds, so any
    prefix of the output spans every stream that still has items.
    """
    cursors = {key: iter(items) for key, items in streams.items()}
    while cursors:
        for key in list(cursors):
            for _ in range((weights or {}).get(key, 1)):
                item = next(cursors[key], _END)
                if item is _END:
                    del cursors[key]
                    break
                yield item
