"""Phase 2: whole-repo file enumeration, plus Git LFS resolution.

One `git/trees?recursive=1` call per repo returns every path with its exact
size and blob sha, with no search-index size ceiling -- this is where every
file in the corpus actually comes from. A repo costs exactly one metered
request; the response names the commit `HEAD` resolved to, and later reads are
pinned to it.

LFS handling matters because an LFS-tracked 2 GB CSV appears in the tree as a
~130 byte pointer blob. Left alone those both pollute the small buckets with
files that are not CSVs at all and hide the largest files in the corpus. The
pointer itself carries the true size and a sha256 of the content, so resolving
one only needs the pointer's own bytes (batched over GraphQL):

    version https://git-lfs.github.com/spec/v1
    oid sha256:055c9ece2b8b760fe6917cc85a9186632decbd659789fd03d8b7d2d7e6a2bcd5
    size 16490
"""
from __future__ import annotations

import fnmatch
import hashlib
import queue
from concurrent.futures import ThreadPoolExecutor
import random
import threading
import time
from typing import Any, Iterator

from . import config, log
from .util import interleave, run_workers
from .client import GitHub


# HTTP statuses a retry can never fix. Everything else -- 403, 429, 5xx, a
# dropped connection -- is transient and the repo stays in the queue.
PERMANENT_FAILURES = frozenset({404, 451})


def scanned() -> set[int]:
    """Repos that will not be offered again.

    A repo retires when its newest record says the tree was actually
    enumerated, or when the failure is one no retry can fix: 404 (deleted or
    renamed) and 451 (blocked). Any other failure leaves it pending, so a
    rate-limit episode cannot drain the queue without scanning anything.

    Reads three scalars per record rather than decoding it (see
    `log.scan_fields`).
    """
    verdict: dict[int, bool] = {}
    for repo_id, state, http in log.scan_fields("trees", ("repo_id", "state", "http")):
        if repo_id is None:
            continue
        # Later records win: a repo that failed and was later scanned is done.
        verdict[repo_id] = state == "ok" or http in PERMANENT_FAILURES
    return {repo_id for repo_id, done in verdict.items() if done}


# Share of the scan budget each discovery size tier gets per round under
# `interleave`. Never zero, so a run stopped early has scanned every kind of
# repo.
#
# `lfs` is weighted heavily because it carries the scarce top buckets and is
# exhaustible: `.gitattributes` search reaches a finite set of repos, and
# `_interleave` redistributes a tier's share once it runs out, so a heavy
# weight is spent early and costs nothing after. `tiny` is weighted up against
# the middle tiers to bring the share of small files closer to GitHub's real
# distribution; `hi` repos already yield plenty of small files but also many
# large ones, so lifting `tiny` moves the small end without disturbing the
# large one. Re-derive these from `csvscrape calibrate` per-tier yields.
#
# `micro` is a prior rather than a measurement (`tiny`'s weight scaled by the
# two bands' populations); set it from `calibrate` once micro repos are
# scanned.
SCAN_WEIGHTS = {"lfs": 8, "hi": 4, "mid": 2, "low": 1, "tiny": 6, "micro": 2}


def _priority(repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Largest expected files first: fills the scarce top buckets soonest."""
    return sorted(
        repos, key=lambda r: (r["tier"] == "lfs", r["size_hint"]), reverse=True
    )


def _interleave(repos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Weighted round-robin across tiers, largest-first within each tier.

    `trees` is the multi-day phase, so it is always stopped early -- which
    makes the head of this queue, not its tail, the corpus. Strict priority
    ordering means an interrupted run has seen only data-publishing repos.
    """
    streams: dict[str, list[dict[str, Any]]] = {}
    for repo in _priority(repos):
        streams.setdefault(repo["tier"], []).append(repo)
    return list(interleave(streams, SCAN_WEIGHTS))


def pending(order: str = "interleave", seed: int = 0) -> list[dict[str, Any]]:
    """Discovered-but-unscanned repos, in scan order.

    `tier` and `size_hint` are *first-sighting* facts: `discover` writes each
    repo once, tagged by whichever shard reached it first. Usually that is the
    largest-size shard that could have (see `discover.shards`), but under the
    weighted interleave a repo can be caught by a `tiny` shard before the `hi`
    one that would describe it better. Anything reading `tier` as a property
    of the repo -- notably `index.calibrate` -- inherits that.

    A repo can appear twice only if a damaged log hid it from `discover`'s
    `seen` set; the larger hint wins then.
    """
    done = scanned()
    best: dict[int, dict[str, Any]] = {}
    for rec in log.read("discovery"):
        if rec["repo_id"] in done:
            continue
        prev = best.get(rec["repo_id"])
        if prev is None or rec["size_hint"] > prev["size_hint"]:
            best[rec["repo_id"]] = rec

    repos = list(best.values())
    if order == "priority":
        return _priority(repos)
    if order == "random":
        # Uniform inclusion probability given discovery -- the easiest sampling
        # design to defend in writing, at the cost of the large-file head start.
        shuffled = sorted(repos, key=lambda r: r["repo_id"])
        random.Random(seed).shuffle(shuffled)
        return shuffled
    return _interleave(repos)


def _lfs_patterns(text: str) -> list[str]:
    """Path globs marked with filter=lfs in a .gitattributes file."""
    patterns = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "filter=lfs" not in line:
            continue
        pattern = line.split()[0].lstrip("/")
        patterns.append(pattern)
    return patterns


def parse_pointer(body: bytes) -> tuple[str, int] | None:
    """(sha256 oid, real size) from an LFS pointer blob, or None if not one.

    `size 0` is a valid resolution (a pointer to an empty object), so the size
    is tested against None rather than for truth; otherwise the blob would be
    left unresolved and counted as `lfs_suspect`. A negative size or an empty
    oid is malformed.
    """
    if not body.startswith(config.LFS_MAGIC):
        return None
    oid = size = None
    for line in body.decode("utf-8", "replace").splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":", 1)[1].strip()
        elif line.startswith("size "):
            try:
                size = int(line.split()[1])
            except (IndexError, ValueError):
                return None
    if not oid or size is None or size < 0:
        return None
    return oid, size


def _is_tracked(path: str, patterns: list[str], prefix: str) -> bool:
    rel = path[len(prefix):] if prefix and path.startswith(prefix) else path
    return any(
        fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pat)
        for pat in patterns
    )


def _blob_bodies(gh: GitHub, full_name: str, items: list[tuple[str, str]],
                 batch: int = config.LFS_BATCH,
                 ref: str = "HEAD") -> list[bytes | None]:
    """Contents for (path, sha) pairs, batched over GraphQL.

    A GraphQL query costs one point whatever it carries, where `raw_file`
    spends one request per blob against the per-IP raw limit. Pointer-heavy
    repos hold hundreds of candidates, so batching moves that load off the
    scarcer budget. (For one or two blobs a raw request is cheaper, which is
    why `.gitattributes` is read directly.)

    Here an LFS pointer *is* the content being read, which is what makes the
    git blob the right thing to fetch (unlike in `probe`).

    Whatever GraphQL declines -- a binary blob, a body that does not account
    for -- falls back to a raw request individually.
    """
    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]

    def one(chunk: list[tuple[str, str]]) -> list[bytes | None]:
        try:
            return gh.blobs([{"full_name": full_name, "blob_sha": sha}
                             for _, sha in chunk])
        except Exception:
            # A failed batch says nothing about any blob in it; they all take
            # the raw fallback below.
            return [None] * len(chunk)

    # In parallel, because the pointer count per repo is very skewed and a
    # large repo would otherwise wait on several multi-second round trips in
    # series. Requests still leave at whatever the graphql bucket allows.
    out: list[bytes | None] = []
    if len(chunks) == 1:
        out = one(chunks[0])
    else:
        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as pool:
            for got in pool.map(one, chunks):
                out.extend(got)

    misses = [i for i, body in enumerate(out) if body is None]
    if misses:
        with ThreadPoolExecutor(max_workers=min(len(misses), 16)) as pool:
            for i, body in zip(misses, pool.map(
                    lambda i: gh.raw_file(full_name, ref, items[i][0]), misses)):
                out[i] = body
    return out


def resolve_lfs(gh: GitHub, full_name: str, blobs: list[tuple[str, str, int]],
                attr_blobs: list[tuple[str, str]],
                ref: str = "HEAD") -> Iterator[dict[str, Any]]:
    """Turn pointer blobs into real (size, sha256) pairs.

    Gated on .gitattributes actually existing in the tree, so the 99% of repos
    that do not use LFS cost nothing. A size-range backstop catches files whose
    tracking rules have drifted out of .gitattributes; the magic-prefix check is
    what actually decides, since genuine 141-byte CSVs do exist in the wild.
    """
    # Read over raw, not batched: a repo rarely has more than one or two
    # .gitattributes files, and a GraphQL point is worth more than that.
    patterns: list[tuple[str, list[str]]] = []
    attr_paths = [path for path, _ in attr_blobs]
    with ThreadPoolExecutor(max_workers=min(len(attr_paths) or 1, 8)) as pool:
        fetched = list(pool.map(lambda p: gh.raw_file(full_name, ref, p),
                                attr_paths))
    for attr_path, body in zip(attr_paths, fetched):
        if not body:
            continue
        found = _lfs_patterns(body.decode("utf-8", "replace"))
        if found:
            patterns.append((attr_path[: -len(".gitattributes")], found))

    candidates = [
        (path, sha, size)
        for path, sha, size in blobs
        # Resolve against the wider keep-list, not the index list, so widening
        # INDEX_EXTENSIONS later does not require re-resolving pointers either.
        if config.is_kept(path)
        # Never fetch anything that cannot possibly be a pointer: a
        # .gitattributes glob may widen the window, but not past this bound.
        and size <= config.LFS_CANDIDATE_MAX_BYTES
        and (
            config.LFS_POINTER_MIN <= size <= config.LFS_POINTER_MAX
            or any(_is_tracked(path, pats, pre) for pre, pats in patterns)
        )
    ]
    if not candidates:
        return

    bodies = _blob_bodies(gh, full_name,
                          [(path, sha) for path, sha, _ in candidates], ref=ref)

    for (path, sha, size), body in zip(candidates, bodies):
        if body is not None and not body.startswith(config.LFS_MAGIC):
            # Genuine small CSVs of pointer-like size do exist. Recording the
            # negative keeps them out of the "unresolved pointer" bin
            # (`lfs_suspect`), which the index excludes from the pool.
            yield {
                "repo_id": None, "full_name": full_name, "path": path,
                "blob_sha": sha, "pointer_size": size, "size": size,
                "oid": None, "state": "plain", "ts": int(time.time()),
            }
            continue
        if not body:
            continue
        parsed = parse_pointer(body)
        if parsed:
            oid, real = parsed
            yield {
                "repo_id": None,  # filled by the caller
                "full_name": full_name,
                "path": path,
                "blob_sha": sha,
                "pointer_size": size,
                "size": real,
                "oid": oid,
                "state": "lfs",
                "ts": int(time.time()),
            }


def scan_one(gh: GitHub, repo: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_name = repo["full_name"]
    tree = gh.tree(full_name)
    now = int(time.time())
    if "error" in tree:
        return (
            {"repo_id": repo["repo_id"], "full_name": full_name, "state": "error",
             "http": tree["error"], "ts": now},
            [],
        )

    # The commit this scan saw. For a ref argument, `git/trees/{ref}` echoes the
    # *commit* sha the ref resolved to (not the tree's own sha), so the pin is
    # free. It is used for the .gitattributes and pointer reads below too, so a
    # push landing mid-scan cannot mix two commits.
    #
    # A response without one is treated as a failed scan rather than an
    # unpinned success: `probe` and `download` both key on the pin, and an
    # unpinned row would read whatever the branch points at later. `http: 0`
    # is not in PERMANENT_FAILURES, so the repo stays pending.
    ref = tree.get("sha")
    if not ref:
        return (
            {"repo_id": repo["repo_id"], "full_name": full_name, "state": "error",
             "http": 0, "message": "tree response carried no commit sha",
             "ts": now},
            [],
        )

    # `mode` comes along because a symlink is a `blob` too, whose bytes are the
    # link target (mode 120000). Recorded rather than filtered here, so the
    # policy lives in `index._projection` and can change with a `compact`.
    all_blobs = [
        (e["path"], e["sha"], e.get("size", 0), e.get("mode", ""))
        for e in tree.get("tree", [])
        if e.get("type") == "blob"
    ]
    # Read .gitattributes off the *unfiltered* tree: it never reaches the log,
    # but LFS resolution depends on it.
    attr_blobs = [(p, sha) for p, sha, _, _ in all_blobs
                  if p.rsplit("/", 1)[-1] == ".gitattributes"]

    blobs = [b for b in all_blobs if config.is_kept(b[0])]
    capped = len(blobs) > config.MAX_BLOBS_PER_REPO
    if capped:
        # A tree comes back path-sorted, so slicing the front would keep one
        # directory. A hash of the path is a deterministic sample instead:
        # reproducible across re-scans, and spread across the repo's layout.
        blobs = sorted(blobs, key=lambda b: hashlib.blake2b(
            b[0].encode("utf-8", "replace"), digest_size=8).digest()
        )[: config.MAX_BLOBS_PER_REPO]
        blobs.sort()
    pointers = []
    if attr_blobs or any(
        config.LFS_POINTER_MIN <= s <= config.LFS_POINTER_MAX
        for _, _, s, _ in blobs
    ):
        # Without the mode: pointer resolution does not need it.
        for rec in resolve_lfs(gh, full_name,
                               [(pa, sh, sz) for pa, sh, sz, _ in blobs],
                               attr_blobs, ref=ref):
            rec["repo_id"] = repo["repo_id"]
            pointers.append(rec)

    record = {
        "repo_id": repo["repo_id"],
        "full_name": full_name,
        # What `probe` and `download` pin their reads to, so a file edited
        # since the scan is still read as the blob the index names. Always a
        # commit sha on an `ok` record, so a reader can treat a mismatch
        # against it as corruption rather than drift.
        "ref": ref,
        "state": "ok",
        "truncated": bool(tree.get("truncated")),
        "capped": capped,
        # n_blobs is the *unfiltered* count, so the denominator for any
        # "how data-heavy is this repo" analysis survives the filtering.
        "n_blobs": len(all_blobs),
        "n_kept": len(blobs),
        "tier": repo.get("tier"),
        "ext": repo.get("ext"),
        # Parallel typed arrays, not arrays-of-arrays: Polars and DuckDB both
        # infer these as String/String/Int64 lists and zip them with a parallel
        # explode, whereas positional rows degrade to nested JSON with quoted
        # values. Contents are filtered to KEEP_EXTENSIONS, which is wider than
        # what gets indexed, so widening the index is a re-read not a rescan.
        "paths": [p for p, _, _, _ in blobs],
        "shas": [s for _, s, _, _ in blobs],
        "sizes": [z for _, _, z, _ in blobs],
        "modes": [m for _, _, _, m in blobs],
        "ts": now,
    }
    return record, pointers


def _note(stats: dict[str, Any]) -> str:
    """The two running totals worth watching, on the progress line.

    "kept" rather than "csv": these are KEEP_EXTENSIONS matches, and calling
    them csvs invites the reader to reconcile them against `status`, which
    counts the narrower INDEX_EXTENSIONS pool.
    """
    return f"{stats['kept_blobs']:,} kept · {stats['lfs_pointers']:,} lfs"


def run(gh: GitHub, limit: int | None = None, workers: int = 3,
        tier: str | None = None, on_event: Any = None,
        order: str = "interleave", seed: int = 0,
        ext: str | None = None) -> dict[str, Any]:
    todo = pending(order=order, seed=seed)
    if tier:
        # For calibration: scan a sample of one tier.
        todo = [r for r in todo if r.get("tier") == tier]
    if ext:
        # Independent of `--tier`: `--tier hi --ext tsv` is a valid question.
        todo = [r for r in todo if r.get("ext") == ext]
    if limit:
        todo = todo[:limit]
    work: queue.Queue = queue.Queue()
    for repo in todo:
        work.put(repo)

    # `kept_blobs` counts KEEP_EXTENSIONS matches, which is wider than what
    # `compact` indexes, so it sits well above the file count `status` reports.
    # `lfs_pointers` are resolved pointers; `lfs_plain` are pointer-sized blobs
    # that turned out to be genuine small files.
    stats: dict[str, Any] = {"repos": 0, "errors": 0, "blobs": 0,
                             "kept_blobs": 0, "lfs_pointers": 0, "lfs_plain": 0}
    lock = threading.Lock()
    emit = on_event or (lambda *a, **k: None)
    queued = len(todo)
    stopping = threading.Event()
    # Stop/accept handshake for `util.run_workers`.
    accepting = threading.Event()
    accepting.set()

    with log.Writer("trees") as trees_log, log.Writer("lfs", rotate=500) as lfs_log:
        def worker() -> None:
            while not stopping.is_set():
                try:
                    repo = work.get_nowait()
                except queue.Empty:
                    return
                try:
                    record, pointers = scan_one(gh, repo)
                except Exception as exc:  # a scan must never kill the run
                    record, pointers = (
                        {"repo_id": repo["repo_id"], "full_name": repo["full_name"],
                         "state": "error", "http": 0, "message": repr(exc)[:200],
                         "ts": int(time.time())},
                        [],
                    )
                # Under `lock` so the `accepting` check and the writes it guards
                # are one step.
                with lock:
                    if not accepting.is_set():
                        # Interrupted, and this scan missed the grace period.
                        # An unwritten repo is an unscanned one, so `pending`
                        # offers it again next run.
                        return
                    trees_log.write(record)
                    for pointer in pointers:
                        lfs_log.write(pointer)
                    stats["repos"] += 1
                    stats["blobs"] += record.get("n_blobs", 0)
                    stats["kept_blobs"] += record.get("n_kept", 0)
                    for pointer in pointers:
                        stats["lfs_pointers" if pointer["state"] == "lfs"
                              else "lfs_plain"] += 1
                    if record["state"] != "ok":
                        stats["errors"] += 1
                    emit("repo", done=stats["repos"], of=queued,
                         note=_note(stats), name=repo["full_name"])

        run_workers(
            [threading.Thread(target=worker, daemon=True) for _ in range(workers)],
            stats, stopping=stopping, accepting=accepting, lock=lock,
            on_interrupt=lambda: emit(
                "repo", done=stats["repos"], of=queued, note=_note(stats),
                name="interrupted, finishing in-flight repos"))
    # What each budget spent. Probed with `getattr` because a caller may hand
    # in a test double that only answers `tree`.
    for pool in ("raw", "core_bucket", "graphql_bucket"):
        if hasattr(getattr(gh, pool, None), "stats"):
            stats |= getattr(gh, pool).stats
    return stats
