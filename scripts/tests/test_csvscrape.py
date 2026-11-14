"""Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path


class KeepList(unittest.TestCase):
    def test_extensions_and_case(self):
        from csvscrape import config

        for path in ("a/b.csv", "X.CSV", "d/e.txt", "f.tsv", "g.data"):
            self.assertTrue(config.is_kept(path), path)
        for path in ("g.png", "i.parquet", "h.tar.gz", "Makefile", "x.csvx"):
            self.assertFalse(config.is_kept(path), path)

    def test_compressed_data_files_kept(self):
        from csvscrape import config

        self.assertTrue(config.is_kept("data/foo.csv.gz"))
        self.assertTrue(config.is_kept("data/foo.tsv.zst"))
        self.assertFalse(config.is_kept("archive.tar.gz"))

    def test_keep_list_is_wider_than_index_list(self):
        """The whole point of two lists: widening the index must not need a rescan."""
        from csvscrape import config

        self.assertTrue(set(config.INDEX_EXTENSIONS) < set(config.KEEP_EXTENSIONS))


class LfsPointers(unittest.TestCase):
    POINTER = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:055c9ece2b8b760fe6917cc85a9186632decbd659789fd03d8b7d2d7e6a2bcd5\n"
        b"size 16490\n"
    )

    def test_parses_real_pointer(self):
        from csvscrape import trees

        oid, size = trees.parse_pointer(self.POINTER)
        self.assertEqual(size, 16490)
        self.assertTrue(oid.startswith("055c9ece"))

    def test_rejects_genuine_small_csv(self):
        """A real CSV of pointer-like size must not read as a pointer."""
        from csvscrape import trees

        body = b"pseudo.r.squared,df.null,logLik,AIC\n0.42,18,7.53,-9.07\n"
        self.assertIsNone(trees.parse_pointer(body))

    def test_rejects_truncated_pointer(self):
        from csvscrape import trees

        self.assertIsNone(trees.parse_pointer(self.POINTER.split(b"\n")[0]))

    def _sized(self, size: bytes) -> bytes:
        return self.POINTER.replace(b"size 16490\n", b"size " + size + b"\n")

    def test_a_pointer_to_an_empty_object_resolves(self):
        """`size 0` is a valid resolution, not a parse failure."""
        from csvscrape import trees

        parsed = trees.parse_pointer(self._sized(b"0"))
        self.assertIsNotNone(parsed, "a pointer to an empty object went unresolved")
        oid, size = parsed
        self.assertEqual(size, 0)
        self.assertTrue(oid.startswith("055c9ece"))




    def test_gitattributes_glob_and_explicit_rules(self):
        from csvscrape import trees

        text = (
            "# comment\n"
            "*.csv filter=lfs diff=lfs merge=lfs -text\n"
            "/Data/Clean/active.csv filter=lfs diff=lfs merge=lfs -text\n"
            "*.md text\n"
        )
        patterns = trees._lfs_patterns(text)
        self.assertEqual(patterns, ["*.csv", "Data/Clean/active.csv"])
        self.assertTrue(trees._is_tracked("x/y.csv", patterns, ""))
        self.assertTrue(trees._is_tracked("Data/Clean/active.csv", patterns, ""))
        self.assertFalse(trees._is_tracked("readme.md", patterns, ""))

    def test_candidate_cap_blocks_large_fetches(self):
        """A `*.csv filter=lfs` glob must not make a large file a pointer
        candidate."""
        from csvscrape import config

        self.assertLessEqual(config.LFS_POINTER_MAX, config.LFS_CANDIDATE_MAX_BYTES)
        self.assertLess(config.LFS_CANDIDATE_MAX_BYTES, 100_000)


class LogDurability(unittest.TestCase):
    """The no-database design rests on these two properties."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, log

        importlib.reload(config)
        importlib.reload(log)
        self.log = log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_rotation_round_trips(self):
        with self.log.Writer("t", rotate=10) as writer:
            for i in range(25):
                writer.write({"repo_id": i})
        self.assertEqual([r["repo_id"] for r in self.log.read("t")], list(range(25)))

    def test_records_flushed_into_a_frame_survive_a_kill(self):
        """A killed process costs the records since the last frame, not the shard."""
        writer = self.log.Writer("t", rotate=1000)
        for i in range(3):
            writer.write({"repo_id": i})
        writer.flush()  # closes the frame; these three are now on disk
        writer.write({"repo_id": 3})  # still buffered inside the open frame
        # No close(): the file keeps whatever the last frame boundary sealed.
        # `read` tolerates the torn tail, so it sees the open shard.
        self.assertEqual([r["repo_id"] for r in self.log.read("t")], [0, 1, 2])
        writer._raw.close()

    def test_an_open_shard_is_hidden_from_parquet_readers(self):
        """Polars raises `OSError: incomplete frame` on a mid-frame zstd stream,
        which fails the whole scan -- so it must never be handed one."""
        writer = self.log.Writer("t", rotate=1000)
        for i in range(3):
            writer.write({"repo_id": i})
        writer.flush()
        self.assertEqual(self.log.sealed("t"), [], "open shard exposed to Polars")
        self.assertEqual(len(list(self.log.read("t"))), 3)
        writer.close()
        self.assertEqual(len(self.log.sealed("t")), 1, "sealed on close")

    def test_rotation_seals_the_finished_shard(self):
        with self.log.Writer("t", rotate=5) as writer:
            for i in range(12):
                writer.write({"repo_id": i})
        self.assertEqual(len(self.log.sealed("t")), 3)

    def test_a_crashed_shard_is_sealed_on_next_open(self):
        """An orphaned .active file is rewritten as whole frames, dropping only
        the torn tail the crash already cost."""
        writer = self.log.Writer("t", rotate=1000)
        for i in range(5):
            writer.write({"repo_id": i})
        writer.flush()
        writer._raw.close()          # simulate kill -9: no rename, no close
        self.assertEqual(self.log.sealed("t"), [])

        with self.log.Writer("t", rotate=1000) as second:
            second.write({"repo_id": 99})
        recovered = sorted(r["repo_id"] for r in self.log.read("t"))
        self.assertEqual(recovered, [0, 1, 2, 3, 4, 99])
        self.assertEqual(len(self.log.sealed("t")), 2, "orphan not sealed")

    def test_torn_trailing_frame_costs_only_its_tail(self):
        with self.log.Writer("t", rotate=1000) as writer:
            for i in range(50):
                writer.write({"repo_id": i})
        shard = next(p for p in (self.tmp / "t").iterdir() if p.name.endswith(".zst"))
        shard.write_bytes(shard.read_bytes()[:-4])  # kill -9 mid-frame
        recovered = [r["repo_id"] for r in self.log.read("t")]
        self.assertEqual(recovered, list(range(len(recovered))))

    def test_reopen_starts_a_new_shard_and_appends(self):
        writer = self.log.Writer("t", rotate=1000)
        writer.write({"repo_id": 0})
        writer.close()
        with self.log.Writer("t", rotate=1000) as second:
            second.write({"repo_id": 1})
        self.assertEqual(sorted(r["repo_id"] for r in self.log.read("t")), [0, 1])

    def test_rec_id_is_unique_across_shards(self):
        with self.log.Writer("t", rotate=10) as writer:
            for i in range(25):
                writer.write({"repo_id": i})
        ids = [r["rec_id"] for r in self.log.read("t")]
        self.assertEqual(len(set(ids)), 25)

    def test_empty_shard_is_not_left_behind(self):
        """An empty zstd frame breaks Polars' NDJSON reader for the whole glob."""
        with self.log.Writer("t", rotate=1000):
            pass
        self.assertEqual(list((self.tmp / "t").iterdir()), [])


try:
    import polars  # noqa: F401

    HAVE_POLARS = True
except ImportError:
    HAVE_POLARS = False


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class LfsResolutionIsKeyedByTheBlob(unittest.TestCase):
    """A pointer blob is `version .../oid sha256:.../size N`, so its sha1
    determines the oid, the real size, and whether it is a pointer at all.
    Keying the log by (repo_id, path) instead made the same blob a separate
    row in every repo holding it -- and made a stale record win a join it had
    no business winning."""

    def setUp(self):
        import importlib

        from csvscrape import config, index, log

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(os.environ.pop, "CSVSCRAPE_DATA", None)
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        for mod in (config, log, index):
            importlib.reload(mod)
        self.index = index

    def _build(self, lfs_records):
        trees = self.tmp / "trees"
        trees.mkdir(parents=True)
        # The same pointer blob `p1` in two repos, at different paths.
        rows = [
            {"repo_id": 1, "full_name": "o/a", "ref": "a" * 40, "state": "ok",
             "truncated": False, "capped": False, "n_blobs": 1, "n_kept": 1,
             "tier": "hi", "ts": 1, "paths": ["one.csv"], "shas": ["p1"],
             "sizes": [130]},
            {"repo_id": 2, "full_name": "o/b", "ref": "b" * 40, "state": "ok",
             "truncated": False, "capped": False, "n_blobs": 1, "n_kept": 1,
             "tier": "hi", "ts": 2, "paths": ["nested/two.csv"], "shas": ["p1"],
             "sizes": [130]},
        ]
        with (trees / "part-00000.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        lfs = self.tmp / "lfs"
        lfs.mkdir(parents=True)
        with (lfs / "part-00000.jsonl").open("w") as fh:
            for rec in lfs_records:
                fh.write(json.dumps(rec) + "\n")
        self.index.build()

    def test_one_resolution_serves_every_repo_holding_the_blob(self):
        """Resolved in o/a only; o/b holds the identical blob, so the resolution
        applies there too."""

        self._build([{"repo_id": 1, "full_name": "o/a", "path": "one.csv",
                      "blob_sha": "p1", "pointer_size": 130,
                      "size": 5_000_000_000, "oid": "deadbeef",
                      "state": "lfs", "ts": 1}])
        frame = self.index.files().collect()
        self.assertEqual(frame.height, 1, "the two occurrences did not collapse")
        row = frame.to_dicts()[0]
        self.assertEqual(row["occurrences"], 2)
        self.assertEqual(row["n_repos"], 2)
        self.assertTrue(row["is_lfs"])
        self.assertFalse(row["lfs_suspect"], "a resolved blob left suspect")
        self.assertEqual(row["size"], 5_000_000_000)
        self.assertEqual(row["content_key"], "deadbeef")

    def test_a_record_for_a_blob_no_longer_present_joins_to_nothing(self):
        """The staleness property the positional key did not have: a record can
        only ever describe its own blob, so one left over from an earlier scan
        cannot attach itself to whatever now sits at that path."""

        self._build([
            {"repo_id": 1, "full_name": "o/a", "path": "one.csv",
             "blob_sha": "p1", "pointer_size": 130, "size": 5_000_000_000,
             "oid": "deadbeef", "state": "lfs", "ts": 2},
            # Same (repo, path), different blob -- the shape a rescan leaves
            # behind when the file at that path changed.
            {"repo_id": 1, "full_name": "o/a", "path": "one.csv",
             "blob_sha": "gone", "pointer_size": 130, "size": 999,
             "oid": "stale", "state": "lfs", "ts": 1},
        ])
        frame = self.index.files().collect()
        self.assertEqual(frame.height, 1)
        row = frame.to_dicts()[0]
        self.assertEqual(row["content_key"], "deadbeef",
                         "a record for another blob won the join")
        self.assertEqual(row["size"], 5_000_000_000)


class IndexBuild(unittest.TestCase):
    """End-to-end compaction against a hand-built log: no network involved."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, index, log

        importlib.reload(config)
        importlib.reload(log)
        importlib.reload(index)
        self.index = index

        trees = self.tmp / "trees"
        trees.mkdir(parents=True)
        rows = [
            # a plain csv, a duplicate of it in another repo, a non-csv, an
            # LFS pointer, a genuine pointer-sized csv, and an unchecked one
            {"repo_id": 1, "full_name": "o/a", "ref": "HEAD", "state": "ok",
             "truncated": False, "capped": False, "n_blobs": 9, "n_kept": 3,
             "tier": "hi", "ts": 1,
             "paths": ["d/x.csv", "d/n.txt", "d/ptr.csv"],
             "shas": ["s1", "s2", "s3"],
             "sizes": [4096, 500, 130]},
            {"repo_id": 2, "full_name": "o/b", "ref": "b" * 40, "state": "ok",
             "truncated": False, "capped": False, "n_blobs": 4, "n_kept": 3,
             "tier": "lfs", "ts": 2,
             "paths": ["x.csv", "small.csv", "unchecked.csv"],
             "shas": ["s1", "s5", "s6"],
             "sizes": [4096, 141, 130]},
            {"repo_id": 3, "full_name": "o/c", "ref": "HEAD", "state": "error",
             "http": 404, "ts": 3},
        ]
        with (trees / "part-00000.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

        lfs = self.tmp / "lfs"
        lfs.mkdir(parents=True)
        with (lfs / "part-00000.jsonl").open("w") as fh:
            fh.write(json.dumps({"repo_id": 1, "full_name": "o/a", "path": "d/ptr.csv",
                                 "blob_sha": "s3", "pointer_size": 130,
                                 "size": 2_000_000_000, "oid": "deadbeef",
                                 "state": "lfs", "ts": 1}) + "\n")
            fh.write(json.dumps({"repo_id": 2, "full_name": "o/b", "path": "small.csv",
                                 "blob_sha": "s5", "pointer_size": 141, "size": 141,
                                 "oid": None, "state": "plain", "ts": 2}) + "\n")
        self.index.build()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_errored_repo_excluded(self):

        names = self.index.files().select("full_name").collect()["full_name"]
        self.assertNotIn("o/c", set(names))

    def test_non_csv_filtered_from_index(self):
        paths = set(self.index.files().select("path").collect()["path"])
        self.assertNotIn("d/n.txt", paths)

    def test_duplicate_content_collapses_to_one_row(self):
        """Same blob sha in two repos is one indexed row, with the duplication
        recorded as counts rather than as extra rows."""
        import polars as pl

        frame = self.index.files().filter(pl.col("blob_sha") == "s1").collect()
        self.assertEqual(frame.height, 1)
        self.assertEqual(frame["occurrences"][0], 2)
        self.assertEqual(frame["n_repos"][0], 2)

    def test_the_scanned_commit_travels_with_its_occurrence(self):
        """`ref` rides with the representative occurrence, not the content: o/a
        wins x.csv on (full_name, path), so its unpinned ref shows and
        o/b's pin does not leak onto it."""

        frame = self.index.files().collect()
        self.assertIn("ref", frame.columns)
        by_path = {r["path"]: r for r in frame.to_dicts()}
        self.assertEqual(by_path["d/x.csv"]["ref"], "HEAD",
                         "the representative did not keep its own ref")
        self.assertEqual(by_path["small.csv"]["ref"], "b" * 40,
                         "the pin was lost between the log and the index")

    def test_no_row_is_lost_to_sharding(self):
        """The dedup shards the keyspace by hash. A prefix match would drop any
        key outside [0-9a-f] silently -- this fixture uses keys like 's1'."""
        import polars as pl

        # Five csv occurrences across the two fixture repos; `pool()` reports
        # four because it drops the lfs_suspect row, but none may go missing
        # from the index itself.
        occurrences = self.index.files().select(
            pl.col("occurrences").sum()).collect().item()
        self.assertEqual(occurrences, 5)
        self.assertEqual(self.index.files().collect().height, 4)

    def test_lfs_size_overrides_pointer_size(self):
        import polars as pl

        row = (self.index.files().filter(pl.col("path") == "d/ptr.csv")
               .collect().to_dicts()[0])
        self.assertEqual(row["raw_size"], 130)
        self.assertEqual(row["size"], 2_000_000_000)
        self.assertTrue(row["is_lfs"])
        self.assertEqual(row["bucket"], 30)

    def test_checked_genuine_small_csv_survives(self):
        import polars as pl

        row = (self.index.files().filter(pl.col("path") == "small.csv")
               .collect().to_dicts()[0])
        self.assertFalse(row["lfs_suspect"])
        self.assertEqual(row["bucket"], 7)

    def test_unchecked_pointer_sized_file_is_suspect(self):
        import polars as pl

        row = (self.index.files().filter(pl.col("path") == "unchecked.csv")
               .collect().to_dicts()[0])
        self.assertTrue(row["lfs_suspect"])

    def test_pool_excludes_suspects_and_counts_unique(self):
        pool = self.index.pool()
        by_bucket = {r["bucket"]: r for r in pool.to_dicts()}
        self.assertEqual(by_bucket[12]["occurrences"], 2)
        self.assertEqual(by_bucket[12]["unique_files"], 1)
        self.assertNotIn(None, by_bucket)
        self.assertEqual(sum(r["occurrences"] for r in pool.to_dicts()), 4)

    def test_render_draws_empty_buckets(self):
        text = self.index.render()
        self.assertIn("·", text)  # buckets 8..29 are empty and must still appear
        self.assertIn("smallest", text)
        self.assertIn("largest", text)

    def test_render_reports_unresolved_lfs(self):
        # One unchecked pointer-sized file in the fixture: excluded from every
        # count, so the headline has to say it is there.
        text = self.index.render()
        self.assertIn("1 unresolved lfs excluded", text.splitlines()[0])


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class TheCoverageNoteSeparatesUntriedFromFailed(unittest.TestCase):
    """A probed count alone cannot tell a run that has not finished from a
    pool that is as complete as it will get: files still queued and files
    permanently 404 print the same "unprobed" number."""

    def _note(self, rows, columns=None):
        import polars as pl

        from csvscrape import index

        schema = columns or {"fetched": pl.Boolean, "http": pl.Int64,
                             "via": pl.String}
        return index._coverage_note(pl.LazyFrame(rows, schema=schema)) or ""

    def test_dead_paths_are_not_reported_as_work(self):
        note = self._note([{"fetched": True, "http": 200, "via": "graphql"}]
                          + [{"fetched": False, "http": 404, "via": "raw"}] * 3)
        self.assertIn("every file attempted", note)
        self.assertIn("http 404:3", note)
        self.assertIn("nothing left to probe", note)

    def test_a_retryable_failure_is_still_work(self):
        note = self._note([{"fetched": False, "http": 429, "via": "raw"}] * 2
                          + [{"fetched": False, "http": 404, "via": "raw"}] * 5)
        self.assertIn("2 for the next probe run", note,
                      "a 404 was counted as something a later pass can fix")

    def test_a_file_never_drawn_is_told_from_one_that_failed(self):
        """`http` is null for both a dropped connection and a file the probe
        never reached; `via` is written by every record and so tells them
        apart."""
        note = self._note([{"fetched": False, "http": None, "via": None}] * 4
                          + [{"fetched": False, "http": None, "via": "raw"}])
        self.assertIn("4 remaining", note)
        self.assertIn("1 failed (no status:1)", note)
        self.assertIn("5 for the next probe run", note)



if __name__ == "__main__":
    unittest.main()


class StubGitHub:
    """Deterministic search results; records every query it is asked for."""

    def __init__(self, per_page=3, incomplete=False):
        self.calls = []
        self.per_page = per_page
        self.incomplete = incomplete

    def search_code(self, query, page=1, per_page=100):
        self.calls.append((query, page))
        base = abs(hash(query)) % 10_000
        return {
            "total_count": 500,
            "incomplete_results": self.incomplete,
            "items": [
                {
                    "path": f"d/f{i}.csv",
                    "sha": f"sha-{base}-{i}",
                    "repository": {"id": base + i, "full_name": f"o{base}/r{i}"},
                }
                for i in range(self.per_page)
            ],
        }


class DiscoveryResumability(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, discover, log

        importlib.reload(config)
        importlib.reload(log)
        importlib.reload(discover)
        self.discover = discover

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_restart_skips_completed_shards(self):
        first = StubGitHub()
        self.discover.run(first, max_shards=3, max_pages=1)
        done_first = self.discover.completed()
        self.assertEqual(len(done_first), 3)

        second = StubGitHub()
        self.discover.run(second, max_shards=3, max_pages=1)
        queried_again = {q for q, _ in second.calls}
        self.assertFalse(queried_again & done_first, "re-queried a completed shard")
        self.assertEqual(len(self.discover.completed()), 6)

    def test_repos_are_durable_before_shard_marked_done(self):
        """A shard recorded as done must have its repos already readable."""
        gh = StubGitHub()
        self.discover.run(gh, max_shards=2, max_pages=1)
        from csvscrape import log

        shards_done = {r["query"] for r in log.read("shards") if r["state"] == "done"}
        repos_by_shard = {}
        for rec in log.read("discovery"):
            repos_by_shard.setdefault(rec["shard"], 0)
            repos_by_shard[rec["shard"]] += 1
        for query in shards_done:
            self.assertGreater(repos_by_shard.get(query, 0), 0, query)

    def test_seen_repos_are_not_rewritten(self):
        gh = StubGitHub()
        self.discover.run(gh, max_shards=2, max_pages=2)
        from csvscrape import log

        ids = [r["repo_id"] for r in log.read("discovery")]
        self.assertEqual(len(ids), len(set(ids)), "duplicate repo records")


class ShardPageBudget(unittest.TestCase):
    """A shard cut short by --pages must stay resumable, not retire as done."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, discover, log

        importlib.reload(config)
        importlib.reload(log)
        importlib.reload(discover)
        self.discover = discover

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_page_budget_marks_partial_not_done(self):
        gh = StubGitHub(per_page=100)  # a full page => never intrinsically exhausted
        self.discover.run(gh, max_shards=1, max_pages=2, min_new_per_page=0)
        self.assertEqual(self.discover.completed(), set())
        self.assertEqual(list(self.discover.progress().values()), [2])

    def test_resumes_from_next_page(self):
        gh = StubGitHub(per_page=100)
        self.discover.run(gh, max_shards=1, max_pages=2, min_new_per_page=0)
        first_query = gh.calls[0][0]
        self.assertEqual([p for q, p in gh.calls if q == first_query], [1, 2])

        again = StubGitHub(per_page=100)
        self.discover.run(again, max_shards=1, max_pages=4, min_new_per_page=0)
        self.assertEqual([p for q, p in again.calls if q == first_query], [3, 4])

    def test_exhausted_shard_retires(self):
        gh = StubGitHub(per_page=3)  # short page => genuinely out of results
        self.discover.run(gh, max_shards=1, max_pages=5)
        self.assertEqual(len(self.discover.completed()), 1)
        self.assertEqual(self.discover.progress(), {})

    def test_backend_timeout_stays_resumable(self):
        """`incomplete_results` is a timed-out backend, not the end of the shard.
        """
        from csvscrape import log

        gh = StubGitHub(per_page=3, incomplete=True)
        self.discover.run(gh, max_shards=1, max_pages=5)
        self.assertEqual(self.discover.completed(), set(),
                         "a backend timeout must not retire the shard")
        rec = [r for r in log.read("shards") if r.get("state")][0]
        self.assertEqual(rec["state"], "partial")
        self.assertEqual(rec["reason"], "incomplete")
        # The timed-out page is not counted as harvested, so the retry refetches
        # it whole rather than resuming past a page it only half received.
        self.assertEqual(rec["pages"], 0)

        again = StubGitHub(per_page=3)          # backend healthy this time
        self.discover.run(again, max_shards=1, max_pages=5)
        first_query = gh.calls[0][0]
        self.assertEqual([p for q, p in again.calls if q == first_query], [1],
                         "the shard must be re-offered from page 1")

    def test_result_cap_retires_shard(self):
        gh = StubGitHub(per_page=100)
        self.discover.run(gh, max_shards=1, max_pages=10, min_new_per_page=0)
        self.assertEqual(len(self.discover.completed()), 1)


class FailingGitHub(StubGitHub):
    """Full pages until `fail_at`, then a permanent error like a rate limit."""

    def __init__(self, fail_at=3, **kw):
        super().__init__(**kw)
        self.fail_at = fail_at

    def search_code(self, query, page=1, per_page=100):
        self.calls.append((query, page))
        if page >= self.fail_at:
            return {"error": 403, "message": "API rate limit exceeded for user ID 1"}
        base = abs(hash(query)) % 10_000
        return {
            "total_count": 5000,
            "items": [
                {"path": f"d/f{i}.csv", "sha": f"sha-{base}-{page}-{i}",
                 "repository": {"id": base * 1000 + page * 100 + i,
                                "full_name": f"o{base}/r{page}_{i}"}}
                for i in range(self.per_page)
            ],
        }


class ErroredShardResume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, discover, log

        importlib.reload(config)
        importlib.reload(log)
        importlib.reload(discover)
        self.discover = discover

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_error_records_pages_completed(self):
        gh = FailingGitHub(fail_at=3, per_page=100)
        self.discover.run(gh, max_shards=1, max_pages=10, min_new_per_page=15)
        self.assertEqual(list(self.discover.progress().values()), [2])
        self.assertEqual(self.discover.completed(), set())

    def test_retry_resumes_instead_of_retiring_as_low_yield(self):
        """The regression: pages 1-2 already populated `seen`, so restarting at
        page 1 would yield zero new repos and retire the shard after one page."""
        gh = FailingGitHub(fail_at=3, per_page=100)
        self.discover.run(gh, max_shards=1, max_pages=10, min_new_per_page=15)
        query = gh.calls[0][0]

        # fail_at beyond max_pages == a healthy client, and unlike StubGitHub it
        # varies repo ids per page, so novelty does not collapse artificially.
        healthy = FailingGitHub(fail_at=99, per_page=100)
        self.discover.run(healthy, max_shards=1, max_pages=10, min_new_per_page=15)
        pages = [p for q, p in healthy.calls if q == query]
        self.assertEqual(pages[0], 3, f"restarted at page {pages[0]}, losing 3-10")
        self.assertEqual(pages, [3, 4, 5, 6, 7, 8, 9, 10])
        from csvscrape import log as _log

        retired = {r["query"]: r.get("reason") for r in _log.read("shards")
                   if r.get("state") == "done"}
        self.assertEqual(retired.get(query), "result_cap",
                         "should retire on the 1000-result cap, not low_yield")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class Allocation(unittest.TestCase):
    def test_equal_split_when_supply_is_ample(self):
        from csvscrape.sample import allocate

        alloc = allocate({b: 1000 for b in range(10)}, 1000)
        self.assertEqual(set(alloc.values()), {100})
        self.assertEqual(sum(alloc.values()), 1000)

    def test_starved_bucket_returns_surplus(self):
        """Bucket 30 has 2 files; its unused quota must be redistributed."""
        from csvscrape.sample import allocate

        supply = {28: 1000, 29: 1000, 30: 2}
        alloc = allocate(supply, 300)
        self.assertEqual(alloc[30], 2)
        self.assertEqual(sum(alloc.values()), 300)

    def test_never_exceeds_supply(self):
        from csvscrape.sample import allocate

        supply = {1: 3, 2: 5, 3: 1}
        alloc = allocate(supply, 1000)
        self.assertEqual(alloc, {1: 3, 2: 5, 3: 1})
        self.assertEqual(sum(alloc.values()), 9)

    def test_fewer_slots_than_buckets(self):
        from csvscrape.sample import allocate

        alloc = allocate({b: 10 for b in range(10)}, 3)
        self.assertEqual(sum(alloc.values()), 3)


class FileValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, data: bytes) -> Path:
        p = self.tmp / "f.csv"
        p.write_bytes(data)
        return p

    def _describe(self, data: bytes):
        from csvscrape.features import describe

        return describe(data, size=len(data))



    def test_binary_and_html_are_not_rejected_by_check(self):
        """What the bytes *are* is the selector's call, not base validation's."""
        from csvscrape import validate

        for data in (b"\x1f\x8b\x08\x00" + b"\x00" * 60,
                     b"PK\x03\x04" + b"\x00" * 60,
                     b"PAR1" + b"\x00" * 60,
                     b"<!DOCTYPE html>\n<html><body>404</body></html>\n",
                     "a;b\ncafé;1\n".encode("latin-1")):
            self.assertIsNone(validate.check(self._write(data)), data[:8])

    def test_lfs_pointer_rejected(self):
        """A pointer hashes correctly as its own blob but is not the file."""
        from csvscrape import validate

        data = (b"version https://git-lfs.github.com/spec/v1\n"
                b"oid sha256:" + b"a" * 64 + b"\nsize 10\n")
        self.assertEqual(validate.check(self._write(data)), "lfs_pointer")

    def test_empty_and_degenerate_files_kept(self):
        """The index admits 0-byte files on purpose (INDEX_MIN_BYTES = 0), and
        a file of only comments or blank lines is a parser input too."""
        from csvscrape import validate

        for data in (b"", b"\n\n\n\n", b"# generated by foo\n# nothing here\n"):
            self.assertIsNone(validate.check(self._write(data)), data)
            # must not raise; without a table next to them, the `#` lines
            # are taken as records and not as comments
            self._describe(data)


    def test_git_blob_sha1_matches_git(self):
        from csvscrape import validate

        data = b"a,b\n1,2\n"
        # git hash-object equivalent
        import hashlib
        expected = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
        self.assertEqual(validate.git_blob_sha1(len(data), [data]), expected)


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class SelectorVerifiers(unittest.TestCase):
    """Whole-file gates: the checks a head sample cannot make."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _file(self, data: bytes) -> Path:
        p = self.tmp / "f.csv"
        p.write_bytes(data)
        return p

    def test_utf8_valid_whole_file(self):
        from csvscrape.selectors import whole_file_utf8
        self.assertIsNone(whole_file_utf8(self._file(b"a,b\n1,2\n" * 5000), {}))

    def test_utf8_bad_byte_past_the_head_sample(self):
        """The whole point: a stray byte at 40 KB clears every head check."""
        from csvscrape.selectors import whole_file_utf8
        data = b"a,b\n" + b"x,y\n" * 10000 + "café,1\n".encode("latin-1")
        why = whole_file_utf8(self._file(data), {})
        self.assertIsNotNone(why)
        self.assertTrue(why.startswith("invalid_utf8@"))

    def test_utf8_multibyte_across_chunk_boundary_is_not_an_error(self):
        """An incremental decoder must carry state across 1 MiB reads."""
        from csvscrape.selectors import whole_file_utf8
        pad = (1 << 20) - 2
        data = b"a\n" * (pad // 2) + "é,x\n".encode() * 10
        self.assertIsNone(whole_file_utf8(self._file(data), {}))

    def test_nul_found_deep_in_file(self):
        from csvscrape.selectors import no_nul_bytes
        data = b"a,b\n" + b"1,2\n" * 10000 + b"3,\x004\n"
        self.assertTrue(no_nul_bytes(self._file(data), {}).startswith("nul_byte@"))

    def test_dialect_drift_detected(self):
        from csvscrape.selectors import dialect_holds
        data = b"a,b,c\n" + b"1,2,3\n" * 3000 + b"\n" + b"p;q;r\n" * 3000
        why = dialect_holds(self._file(data), {"delimiter": ","})
        self.assertIn("dialect_drift", why)

    def test_stable_dialect_accepted(self):
        from csvscrape.selectors import dialect_holds
        data = b"a,b,c\n" + b"1,2,3\n" * 6000
        self.assertIsNone(dialect_holds(self._file(data), {"delimiter": ","}))

    def test_tail_window_never_slides_back_to_the_head(self):
        """On a file smaller than the window, the tail read must not slide back
        over the head and agree with itself."""
        from csvscrape.selectors import dialect_holds
        data = b"a,b\n" + b"1,2\n" * 1200 + b"\n" + b"p;q;r\n" * 1200
        self.assertIn("dialect_drift", dialect_holds(self._file(data), {"delimiter": ","}))


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class WeightedShuffle(unittest.TestCase):
    """1/size weighting within an octave. The textbook `u ** (1/w)` key underflows
    for large files, so the shuffle must still work there."""

    def setUp(self):
        from csvscrape import sample

        self.sample = sample

    def _order(self, size: int, seed: int) -> list[int]:
        import random

        rows = [{"i": i, "size": size} for i in range(40)]
        weights = [1.0 / size] * len(rows)
        picked = self.sample._weighted_shuffle(rows, weights, random.Random(seed))
        return [r["i"] for r in picked]

    def test_large_files_are_actually_shuffled(self):
        for exponent in (11, 16, 20, 30):
            size = 2 ** exponent
            self.assertNotEqual(self._order(size, 0), list(range(40)),
                                f"bucket {exponent} came back in index order")

    def test_seed_changes_the_order(self):
        size = 2 ** 30
        self.assertNotEqual(self._order(size, 0), self._order(size, 1))

    def test_same_seed_reproduces(self):
        self.assertEqual(self._order(2 ** 24, 7), self._order(2 ** 24, 7))

    def test_smaller_files_are_favoured_within_a_bucket(self):
        """1/size weighting must survive into the large buckets too."""
        import random

        rows = [{"i": i, "size": 2 ** 30 if i % 2 else 2 ** 29} for i in range(200)]
        weights = [1.0 / r["size"] for r in rows]
        wins = 0
        for seed in range(300):
            order = self.sample._weighted_shuffle(rows, weights, random.Random(seed))
            wins += order[0]["size"] == 2 ** 29
        # Chance is 150, the 2:1 weighting predicts ~200. Anything comfortably
        # above chance shows the weight survived into a 2**30-sized bucket.
        self.assertGreater(wins, 175)


class DiversityBudgetOrder(unittest.TestCase):
    """The per-repo and per-owner caps are one global budget, so the scarce large
    buckets must claim it first."""

    @unittest.skipUnless(HAVE_POLARS, "polars not installed")
    def test_scarce_buckets_claim_the_budget_first(self):
        import polars as pl

        from csvscrape import sample

        # One prolific owner supplies everything. Bucket 30 has two files;
        # bucket 10 has fifty and could exhaust the cap on its own.
        rows = [{"content_key": f"k{i}", "repo_id": 1, "owner": "megacorp",
                 "full_name": "megacorp/data", "path": f"p{i}.csv",
                 "size": 2 ** 10, "bucket": 10, "is_lfs": False}
                for i in range(50)]
        rows += [{"content_key": f"big{i}", "repo_id": 2, "owner": "megacorp",
                  "full_name": "megacorp/big", "path": f"b{i}.csv",
                  "size": 2 ** 30, "bucket": 30, "is_lfs": False}
                 for i in range(2)]
        pool = pl.LazyFrame(rows)
        picked, short = sample.draw({10: 5, 30: 2}, seed=0, per_repo=100,
                                    per_owner=4, pool=pool)
        got = picked.group_by("bucket").len().to_dicts()
        by_bucket = {r["bucket"]: r["len"] for r in got}
        self.assertEqual(by_bucket.get(30), 2, "scarce bucket went short")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class ProbeCandidates(unittest.TestCase):
    """Successive draws advance through the pool rather than redrawing the same
    rows."""

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe, self.log = probe, log

        (self.tmp / "index").mkdir(parents=True)
        rows = []
        for bucket in (10, 11, 12):
            for i in range(400):
                rows.append({
                    "repo_id": i, "owner": "o", "full_name": f"o/r{i}",
                    "path": f"b{bucket}/f{i}.csv", "blob_sha": f"s{bucket}_{i}",
                    "content_key": f"k{bucket}_{i}", "size": 2 ** bucket,
                    "raw_size": 2 ** bucket, "bucket": bucket, "is_lfs": False,
                    "lfs_suspect": False, "tier": "hi", "truncated": False,
                    "ref": f"{i:040x}",
                })
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _mark_probed(self, rows):
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as writer:
            for row in rows:
                writer.write({"content_key": row["content_key"],
                              "fetched": True, "http": 200})

    def test_successive_runs_advance_through_the_pool(self):
        seen = set()
        for _ in range(6):
            batch = self.probe.candidates(limit=60)
            self.assertEqual(len(batch), 60, "probe stalled with pool remaining")
            keys = {r["content_key"] for r in batch}
            self.assertFalse(keys & seen, "re-drew already-probed files")
            seen |= keys
            self._mark_probed(batch)
        self.assertEqual(len(seen), 360)

    def test_sample_is_not_index_order(self):
        batch = self.probe.candidates(limit=90)
        drawn = [r["path"] for r in batch if r["bucket"] == 10]
        in_order = [f"b10/f{i}.csv" for i in range(len(drawn))]
        self.assertNotEqual(drawn, in_order,
                            "candidates came back in parquet order")

    def test_spread_covers_every_bucket(self):
        batch = self.probe.candidates(limit=30, order="spread")
        self.assertEqual(sorted({r["bucket"] for r in batch}), [10, 11, 12])

    def test_the_pin_survives_the_draw(self):
        """`candidates` is the only thing that puts `ref` in front of the range
        request, so a projection that dropped it would silently unpin every
        read without failing anything."""
        batch = self.probe.candidates(limit=30)
        self.assertTrue(batch)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", row["ref"])
                            for row in batch), "a drawn row lost its pin")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class AResumedDrawStillSkipsSettledFiles(unittest.TestCase):
    """The `--order repo` cursor narrows a draw; it must never replace the
    already-probed filter."""

    UNPROBED = {f"k{i}" for i in range(0, 200, 5)}
    GONE = {f"k{i}" for i in range(1, 200, 25)}

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe = probe

        (self.tmp / "index").mkdir(parents=True)
        # A compacted index: `fetched`/`http` are columns, the state every
        # re-run starts from.
        rows = []
        for i in range(200):
            key = f"k{i}"
            unprobed = key in self.UNPROBED
            rows.append({
                "repo_id": i, "owner": "o", "full_name": f"o/r{i}",
                "path": f"f{i}.csv", "blob_sha": f"s{i}", "content_key": key,
                "size": 4096, "raw_size": 4096, "bucket": 12, "is_lfs": False,
                "lfs_suspect": False, "tier": "hi", "truncated": False,
                "ref": f"{i:040x}",
                "fetched": False if unprobed or key in self.GONE else True,
                "http": None if unprobed else (404 if key in self.GONE else 200),
            })
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _resume(self, limit, after=None):
        batch = self.probe.candidates(limit=limit, order="repo", after=after)
        cursor = ((batch[-1]["_repo_rank"], batch[-1]["_key_rank"])
                  if batch else None)
        return [r["content_key"] for r in batch], cursor

    def test_a_resumed_pass_draws_only_what_is_outstanding(self):
        first, cursor = self._resume(5)
        self.assertEqual(len(first), 5)
        rest, _ = self._resume(1000, after=cursor)
        self.assertTrue(rest, "the resumed pass drew nothing at all")
        self.assertEqual(set(rest) - self.UNPROBED, set(),
                         "a resumed pass redrew files the index calls settled")

    def test_the_whole_residue_is_still_reachable_pass_by_pass(self):
        drawn, cursor = [], None
        for _ in range(len(self.UNPROBED) + 2):
            batch, moved = self._resume(4, after=cursor)
            if not batch:
                break
            drawn += batch
            cursor = moved
        self.assertEqual(sorted(drawn), sorted(self.UNPROBED),
                         "the passes did not cover the residue exactly once")

    def test_a_404_is_not_redrawn_by_a_resumed_pass(self):
        _, cursor = self._resume(1)
        rest, _ = self._resume(1000, after=cursor)
        self.assertEqual(set(rest) & self.GONE, set(),
                         "a retired 404 came back in a resumed draw")


class TheDrawPicksItsSamplingFrame(unittest.TestCase):
    """`spread` fills every size octave at the same rate; `uniform` follows the
    pool's own size distribution."""

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(os.environ.pop, "CSVSCRAPE_DATA", None)
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe = probe
        (self.tmp / "index").mkdir(parents=True)
        # Deliberately skewed, which the older fixture is not: with equal
        # counts per bucket the two draws are indistinguishable and a test
        # over one passes under the other.
        self.supply = {10: 900, 11: 90, 12: 10}
        rows = []
        for bucket, n in self.supply.items():
            for i in range(n):
                rows.append({
                    "repo_id": i, "owner": "o", "full_name": f"o/r{bucket}_{i}",
                    "path": f"b{bucket}/f{i}.csv", "blob_sha": f"s{bucket}_{i}",
                    "content_key": f"k{bucket}_{i}", "size": 2 ** bucket,
                    "raw_size": 2 ** bucket, "bucket": bucket, "is_lfs": False,
                    "lfs_suspect": False, "tier": "hi", "truncated": False,
                    "ref": f"{i:040x}",
                })
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def _shares(self, order, limit=200):
        import collections

        batch = self.probe.candidates(limit=limit, order=order, seed=3)
        got = collections.Counter(r["bucket"] for r in batch)
        return {b: got.get(b, 0) / len(batch) for b in self.supply}, len(batch)

    def test_uniform_tracks_the_pool(self):
        total = sum(self.supply.values())
        shares, n = self._shares("uniform")
        for bucket, count in self.supply.items():
            self.assertAlmostEqual(shares[bucket], count / total, delta=0.06,
                                   msg=f"bucket {bucket} is not pool-weighted "
                                       f"({shares[bucket]:.2%} of {n})")

    def test_spread_lifts_the_sparse_buckets_against_uniform(self):
        """The same property, inverted, and stated as a comparison rather than
        a threshold: whatever the exact shares, a spread must move weight off
        the populous buckets and onto the scarce ones, or it is not a spread.
        (A fixed cutoff here would be tuning against this fixture -- the first
        one I picked failed at 2.7% against an arbitrary 3%.)"""
        uniform, _ = self._shares("uniform")
        spread, _ = self._shares("spread")
        self.assertLess(spread[10], uniform[10],
                        "the populous bucket kept its share under the spread")
        for bucket in (11, 12):
            self.assertGreater(spread[bucket], uniform[bucket],
                               f"bucket {bucket} was not lifted by the spread")


class ASnifferWorkerNeverWritesToStdout(unittest.TestCase):
    """DuckDB draws a progress bar on fd 1, from C++, for slow queries; it would
    interleave with the pass's status line, so sniff workers drop stdout."""

    def test_worker_init_takes_the_descriptor_away(self):
        import os

        from csvscrape import sniffers

        # A pipe standing in for the terminal, so the redirect is observable
        # without running a subprocess.
        read_fd, write_fd = os.pipe()
        saved = os.dup(1)
        try:
            os.dup2(write_fd, 1)
            sniffers.worker_init(("custom",))
            os.write(1, b"this should go nowhere")
        finally:
            os.dup2(saved, 1)
            os.close(saved)
            os.close(write_fd)
            os.set_blocking(read_fd, False)
            try:
                leaked = os.read(read_fd, 4096)
            except BlockingIOError:
                leaked = b""
            os.close(read_fd)

        self.assertEqual(leaked, b"", "a worker's stdout still reaches the pipe")


class EverySnifferReadsTheSameDecoding(unittest.TestCase):
    """The dialect sniffers must disagree over dialect, never over encoding:
    CleverCSV fed a UTF-8 decode picked U+FFFD as the delimiter on gb18030 and
    failed on UTF-16, DuckDB fed raw UTF-16 lost the quote."""

    ROWS = "id;name;city\n" + "".join(
        f'{i};"Müller {i}";Köln\n' for i in range(300))

    @staticmethod
    def _encode(rows: str, codec: str) -> bytes:
        if codec == "utf-16-be-bom":
            return b"\xfe\xff" + rows.encode("utf-16-be")
        return rows.encode(codec)

    def test_one_dialect_whatever_the_encoding(self):
        from csvscrape import sniffers

        for codec in ("utf-8", "cp1252", "gb18030", "utf-16", "utf-16-le",
                      "utf-16-be-bom", "utf-32-le"):
            with self.subTest(codec=codec):
                head = self._encode(self.ROWS, codec)
                out = sniffers.run(head, size=len(head),
                                   which=("custom", "clevercsv", "duckdb"))
                self.assertEqual((out["custom"]["delimiter"],
                                  out["custom"]["quote"]), (";", '"'))
                self.assertEqual((out["clevercsv"]["delimiter"],
                                  out["clevercsv"]["quotechar"]), (";", '"'))
                self.assertEqual((out["duckdb"]["Delimiter"],
                                  out["duckdb"]["Quote"]), (";", '"'))
                self.assertEqual(out["duckdb"]["n_columns"], 3)

    def test_utf8_reaches_duckdb_untouched(self):
        from csvscrape import sniffers

        for head in (self.ROWS.encode("utf-8"),
                     b"\xef\xbb\xbf" + self.ROWS.encode("utf-8")):
            with self.subTest(bom=head.startswith(b"\xef")):
                # Not a re-encode: DuckDB skips a UTF-8 BOM itself.
                self.assertIs(sniffers.Sample(head).utf8, head)

    def test_legacy_pages_reach_duckdb_as_utf8(self):
        from csvscrape import sniffers

        for codec in ("cp1252", "utf-16", "gb18030"):
            with self.subTest(codec=codec):
                s = sniffers.Sample(self._encode(self.ROWS, codec))
                self.assertEqual(s.utf8, self.ROWS.encode("utf-8"))

    def test_pipe_flag_reads_characters_not_trail_bytes(self):
        from csvscrape import sniffers

        # A gb18030 character with `|` (0x7C) as its trail byte.
        char = next(c for c in map(chr, range(0x4E00, 0x9FFF))
                    if b"|" in c.encode("gb18030"))
        rows = "a,b\n" + "".join(f"{i},{char}{char}\n" for i in range(200))
        head = rows.encode("gb18030")
        self.assertIn(b"|", head)
        out = sniffers.run(head, size=len(head), which=("duckdb",))
        self.assertFalse(out["duckdb"]["head_has_pipe"])

    def test_charset_normalizer_runs_once_and_never_goes_missing(self):
        from csvscrape import sniffers, validate

        head = self.ROWS.encode("cp1252")
        s = sniffers.Sample(head)
        with unittest.mock.patch.object(
                validate, "charset_normalizer_verdict",
                wraps=validate.charset_normalizer_verdict) as detect:
            out = sniffers.run_sample(s, which=("custom", "charset_normalizer"))
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(out["charset_normalizer"]["encoding"], s.enc.detector["encoding"])

        # A BOM settles `probe_encoding` without the detector; the sniffer
        # still records the detector's own answer.
        bom = sniffers.run(self.ROWS.encode("utf-16"), which=("charset_normalizer",))
        self.assertEqual(bom["charset_normalizer"]["encoding"], "utf_16")

    def test_downloaded_files_go_to_duckdb_as_utf8(self):
        from csvscrape import sniffers

        with tempfile.TemporaryDirectory() as tmp:
            for codec in ("utf-8", "cp1252", "utf-16"):
                with self.subTest(codec=codec):
                    path = Path(tmp) / f"{codec}.csv"
                    path.write_bytes(self._encode(self.ROWS, codec))
                    data = path.read_bytes()
                    out = sniffers.duckdb_file(path, sniffers.Sample(data, len(data)))
                    self.assertEqual((out["Delimiter"], out["Quote"]), (";", '"'))


class RateBucket(unittest.TestCase):
    """`Bucket.sync` adopts the server's budget without losing time already
    earned."""


    def test_sync_adopts_a_smaller_server_remaining(self):
        import time

        from csvscrape.client import Bucket

        bucket = Bucket(3600.0, 5000, capacity=10)
        bucket.sync(remaining=3, reset=time.time() + 60)
        self.assertEqual(bucket._tokens, 3.0)

    def test_sync_never_inflates_the_local_budget(self):
        from csvscrape.client import Bucket

        bucket = Bucket(3600.0, 5000, capacity=10)
        bucket.sync(remaining=4999, reset=0)
        self.assertEqual(bucket._tokens, 10.0)

    def test_sync_keeps_the_refill_earned_since_the_last_take(self):
        """`sync` credits the time since the last refill before clamping, or each
        request's round trip is lost from the budget."""
        import time

        from csvscrape.client import Bucket

        bucket = Bucket(1.0, 10.0, capacity=10)  # 10 tokens/s
        bucket._tokens = 0.0
        bucket._updated = time.monotonic() - 0.5  # half a second in flight
        bucket.sync(remaining=4999, reset=0)
        self.assertGreaterEqual(bucket._tokens, 4.9)  # ~5 earned, none lost
        self.assertLessEqual(bucket._tokens, 5.5)


class AHoldDoesNotRefillTheBucket(unittest.TestCase):
    """A server-imposed hold must not earn tokens: coming off a hold with a full
    burst fires straight back into the limit."""

    def test_tokens_do_not_accrue_while_parked(self):
        from csvscrape.client import Bucket

        bucket = Bucket(1.0, 10.0, capacity=10)  # 10 tokens/s
        bucket._tokens = 0.0
        bucket.hold_for(0.2)
        time.sleep(0.25)
        bucket.take()  # the one token the hold's own expiry cannot avoid
        self.assertLess(bucket._tokens, 1.0,
                        "the backoff was credited to the bucket that caused it")

    def test_a_response_landing_mid_hold_does_not_remove_tokens(self):
        """`_updated` sits at the *end* of a hold, so an unclamped elapsed term
        would run backwards for a request still in flight when one began."""
        from csvscrape.client import Bucket

        bucket = Bucket(1.0, 10.0, capacity=10)
        bucket.hold_for(30.0)
        bucket._tokens = 5.0
        bucket.sync(remaining=4999, reset=0)
        self.assertEqual(bucket._tokens, 5.0)


    def test_time_outside_a_hold_is_still_credited(self):
        """The fix must not stop the clock generally, only for the backoff."""
        from csvscrape.client import Bucket

        bucket = Bucket(1.0, 10.0, capacity=10)
        bucket._tokens = 0.0
        bucket._updated = time.monotonic() - 0.5
        bucket.take()
        self.assertGreaterEqual(bucket._tokens, 3.5)


class TheWireTimeIsStillCredited(unittest.TestCase):
    """Time a request spends on the wire still earns tokens, so pacing reaches the
    permitted rate under latency. A fake clock, because under real threads
    the effect is easily hidden."""

    class Clock:
        def __init__(self):
            self.t = 1000.0

        def monotonic(self):
            return self.t

        def sleep(self, seconds):
            self.t += seconds

        def time(self):
            return 5_000_000.0 + self.t

    def _drive(self, bucket, clock, n=500, latency=0.35, hold_at=None):
        """`n` requests, each `latency` on the wire, `sync` on every response."""
        start = clock.t
        for i in range(n):
            if hold_at is not None and i == hold_at[0]:
                bucket.hold_for(hold_at[1])
            bucket.take()
            clock.t += latency
            bucket.sync(remaining=4000, reset=clock.time() + 1800)
        return n / (clock.t - start) * 3600.0

    def setUp(self):
        from csvscrape import client

        self.clock = self.Clock()
        real = client.time
        client.time = self.clock
        self.addCleanup(lambda: setattr(client, "time", real))

    def test_the_permitted_rate_is_actually_reached_under_latency(self):
        from csvscrape.client import Bucket

        bucket = Bucket(3600.0, 4750.0, capacity=10)
        bucket._updated = bucket._win_start = self.clock.t
        got = self._drive(bucket, self.clock)
        # The bug scored 1 / (1/rate + latency) -- about 3,260/h at these
        # numbers, which is what the docstring recorded from the live run.
        self.assertGreater(got, 4600.0, f"wire time is being lost: {got:,.0f}/h")
        self.assertLess(got, 4900.0, f"spending above the budget: {got:,.0f}/h")


class GraphqlConcurrencyIsBounded(unittest.TestCase):
    """GraphQL batches in flight are bounded separately from the request rate.
    """

    def test_the_bucket_caps_requests_in_flight(self):
        import threading

        from csvscrape.client import Bucket

        bucket = Bucket(3600.0, 5000, capacity=10, max_inflight=2)
        peak, live = [0], [0]
        lock = threading.Lock()

        def hit():
            with bucket.inflight():
                with lock:
                    live[0] += 1
                    peak[0] = max(peak[0], live[0])
                time.sleep(0.05)
                with lock:
                    live[0] -= 1

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(peak[0], 2, "the in-flight cap did not bind")



class OneBackoffPerCongestionEvent(unittest.TestCase):
    """One blown budget is answered with a rejection to every request already
    on the wire. `RawPool` learned that first -- fourteen 429s halving the rate
    fourteen times for one event -- and the metered path never got the same
    treatment, so a single GraphQL throttle printed five identical
    `holding 5.0m` lines."""

    def test_a_second_refusal_inside_the_window_is_not_announced(self):
        from csvscrape.client import Congestion

        event = Congestion()
        self.assertTrue(event.first(60.0))
        self.assertFalse(event.first(60.0))
        self.assertFalse(event.first(300.0))

    def test_a_later_event_is_announced_again(self):
        from csvscrape.client import Congestion

        event = Congestion()
        self.assertTrue(event.first(0.0))
        event._until = time.monotonic() - 1.0
        self.assertTrue(event.first(0.0), "the window never reopened")

    def test_every_caller_still_applies_the_hold(self):
        """Only the reporting is deduplicated. `hold_for` takes the maximum, so
        a straggler's larger `retry-after` must not be discarded because a
        smaller one arrived first."""
        from csvscrape.client import GitHub, Pacer

        holds, events = [], []
        pacer = Pacer(0.01)
        pacer.hold_for = holds.append
        pacer.take = lambda on_wait=None: None

        class Resp:
            status_code = 429
            headers = {"retry-after": "30"}
            content = b""

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda event, **kw: events.append(kw)
        gh.session = type(
            "S", (), {"request": staticmethod(lambda *a, **k: Resp())})()
        gh._request("GET", "https://example.invalid", pacer)

        self.assertEqual(len(holds), 6, "an attempt skipped its backoff")
        self.assertEqual(len(events), 1, f"one event reported {len(events)}x")


class RefusalsNameTheLimit(unittest.TestCase):
    """A primary limit is a spent hourly budget and is answered by waiting; a
    secondary limit is priced in concurrency and server work and is answered by
    asking for less at once. A bare `retry-after` in the log distinguished
    neither, so a run of 5-minute holds could not be attributed."""

    def test_a_secondary_limit_is_named_from_the_body_and_resource(self):
        from csvscrape.client import GitHub

        class Resp:
            headers = {"x-ratelimit-resource": "graphql"}
            content = b'{"message":"You have exceeded a secondary rate limit"}'

        self.assertEqual(GitHub._limit(Resp()), "secondary/graphql")

    def test_a_primary_limit_is_not_mislabelled(self):
        from csvscrape.client import GitHub

        class Resp:
            headers = {"x-ratelimit-resource": "core"}
            content = b'{"message":"API rate limit exceeded"}'

        self.assertEqual(GitHub._limit(Resp()), "primary/core")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class EmptyMetadata(unittest.TestCase):
    """An unprobed pool must yield a typed empty frame, not an untyped one.

    An untyped empty frame gives content_key the Null dtype, and the left join
    in sample.candidates then fails on a dtype mismatch instead of producing
    null metadata -- so every metadata selector errored rather than returning
    nothing.
    """

    def setUp(self):
        import importlib

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe = probe

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)



    def test_it_joins_against_a_pool_without_erroring(self):
        import polars as pl

        pool = pl.LazyFrame({"content_key": ["a", "b"], "size": [10, 20]})
        joined = pool.join(self.probe.metadata(), on="content_key", how="left")
        self.assertEqual(joined.collect().height, 2)

    def test_flat_fields_match_the_shape(self):
        """The flattened columns must all exist on Shape, or the struct access
        fails at query time on a real log."""
        from csvscrape.features import Shape

        for name in self.probe.FLAT_FIELDS:
            self.assertIn(name, Shape.__annotations__, name)


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class ProbeRetryPolicy(unittest.TestCase):
    """Only a fetch or a 404 settles a file; transient failures stay in the queue.
    """

    def setUp(self):
        import importlib

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe, self.log = probe, log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _write(self, rows):
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as w:
            for r in rows:
                w.write(r)

    def _settled(self):
        return set(self.probe.retired_keys().collect()["content_key"].to_list())

    def test_fetched_rows_are_settled(self):
        self._write([{"content_key": "done", "fetched": True, "http": 200}])
        self.assertIn("done", self._settled())

    def test_404_is_settled_and_never_retried(self):
        """The path does not resolve at the pinned ref; retrying it is waste.
        """
        self._write([{"content_key": "gone", "fetched": False, "http": 404}])
        self.assertIn("gone", self._settled())

    def test_transient_failures_stay_in_the_queue(self):
        for status in (429, 500, 503, None):
            with self.subTest(status=status):
                shutil.rmtree(self.tmp / self.probe.HEADS_LOG, ignore_errors=True)
                self._write([{"content_key": "blip", "fetched": False,
                              "http": status}])
                self.assertNotIn("blip", self._settled())

    def test_a_blip_does_not_remove_the_file_from_candidates(self):
        import polars as pl

        from csvscrape import index

        (self.tmp / "index").mkdir(parents=True, exist_ok=True)
        pl.DataFrame([{
            "repo_id": 1, "owner": "o", "full_name": "o/r", "path": "a.csv",
            "blob_sha": "s", "content_key": "blip", "size": 4096,
            "raw_size": 4096, "bucket": 12, "is_lfs": False,
            "lfs_suspect": False, "tier": "hi", "truncated": False,
            "ref": "d" * 40,
        }]).write_parquet(index.INDEX_PATH)
        self._write([{"content_key": "blip", "fetched": False, "http": 503}])
        keys = [r["content_key"] for r in self.probe.candidates()]
        self.assertEqual(keys, ["blip"], "transient failure was not retried")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class TextOnlySelector(unittest.TestCase):
    """`text-only`: `main`, plus dropping HTML and XML documents."""

    def _pool(self, **over):
        import polars as pl

        row = {"content_key": "k", "magic_kind": None, "looks_like_html": False,
               "is_lfs_pointer": False, "control_frac": 0.0,
               "wide_encoding": None}
        return pl.LazyFrame([row | over])

    def _kept(self, **over):
        from csvscrape import selectors

        return selectors.get("text-only").filter(self._pool(**over)).collect().height

    def test_plain_text_kept(self):
        self.assertEqual(self._kept(), 1)

    def test_magic_prefixed_dropped(self):
        self.assertEqual(self._kept(magic_kind="gzip"), 0)

    def test_html_dropped(self):
        self.assertEqual(self._kept(looks_like_html=True), 0)

    def test_lfs_pointer_dropped(self):
        self.assertEqual(self._kept(is_lfs_pointer=True), 0)

    def test_control_heavy_dropped(self):
        self.assertEqual(self._kept(control_frac=0.4), 0)

    def test_utf16_is_exempt_from_the_control_test(self):
        """UTF-16 is ~50% NUL by construction. Scoring it on control bytes
        alone would drop every Excel export."""
        self.assertEqual(self._kept(control_frac=0.5,
                                    wide_encoding="utf-16-le"), 1)

    def test_it_matches_the_python_helper(self):
        """`Encoding.looks_binary` and this selector must not drift: one is the
        Python statement of what the other does in a query."""
        from csvscrape.validate import probe_encoding

        cases = [
            ("a,b\n1,2\n" * 200).encode(),
            b"\x1f\x8b\x08\x00" + b"\xde\xad" * 500,
            bytes(range(256)) * 40,
            ("a,b\n" + "1,2\n" * 200).encode("utf-16-le"),
            ("код;город\n" + "1;Москва\n" * 300).encode("koi8-r"),
        ]
        for data in cases:
            enc = probe_encoding(data)
            kept = self._kept(magic_kind=enc.magic, control_frac=enc.control_frac,
                              wide_encoding=enc.wide)
            self.assertEqual(bool(kept), not enc.looks_binary, data[:8])


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class WhichSelectorsNeedAProbe(unittest.TestCase):
    """Whether an unprobed file is drawable is the selector's own answer.

    A selector that picks on probe metadata cannot describe a row whose
    metadata is all null, so it draws from the probed fraction only. One that
    does not -- `main`, and both headline corpora with it -- draws from the
    whole pool and lets its whole-file verifier settle each file. Requiring a
    probe for every draw capped the main corpora at whatever the probe pass had
    reached, which over a 12M-file pool is the difference between a corpus and
    a subset of one."""

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, dataset, index, log, probe, sample

        for mod in (config, log, index, probe, sample):
            importlib.reload(mod)
        self.sample, self.probe, self.log = sample, probe, log

        # `dataset.ROOT` is package-relative and does *not* follow
        # CSVSCRAPE_DATA, so redirecting the data root leaves it pointing at
        # the repository's own `datasets/` directory. `download.run` ends by
        # building a dataset definition, so a test that calls it writes
        # `datasets/<name>.json` and `datasets/needs-review.toml` over
        # whatever is really there -- and the default name is `bench`, which
        # is a corpus someone actually has. Verified by putting sentinel
        # content in both and running the suite: both were replaced, the
        # definition with an empty one.
        self.addCleanup(setattr, dataset, "ROOT", dataset.ROOT)
        self.addCleanup(setattr, dataset, "OVERRIDES", dataset.OVERRIDES)
        dataset.ROOT = self.tmp / "datasets"
        dataset.OVERRIDES = dataset.ROOT / "overrides.toml"

        (self.tmp / "index").mkdir(parents=True)
        rows = [{
            "repo_id": i, "owner": "o", "full_name": f"o/r{i}", "path": f"f{i}.csv",
            "blob_sha": f"s{i}", "content_key": f"k{i}", "size": 4096,
            "raw_size": 4096, "bucket": 12, "is_lfs": False,
            "lfs_suspect": False, "tier": "hi", "truncated": False,
        } for i in range(10)]
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _probe(self, keys, fetched=True):
        """Mark rows probed the way `compact` does: as columns on the index.

        Metadata is folded into files.parquet now rather than joined from the
        heads log at query time, so this writes the post-compact state.
        """
        import polars as pl

        from csvscrape import index

        frame = pl.read_parquet(index.INDEX_PATH)
        flags = pl.col("content_key").is_in(list(keys))
        prior = (pl.col("fetched") if "fetched" in frame.columns
                 else pl.lit(None, dtype=pl.Boolean))
        prior_http = (pl.col("http") if "http" in frame.columns
                      else pl.lit(None, dtype=pl.Int64))
        # Cumulative: successive calls mark more rows without clearing earlier ones.
        prior_agreement = (pl.col("agreement") if "agreement" in frame.columns
                           else pl.lit(None, dtype=pl.Int64))
        cols = {"fetched": pl.when(flags).then(fetched).otherwise(prior),
                "http": pl.when(flags).then(200).otherwise(prior_http),
                # Cumulative like `fetched`: a second call marking k2 must not
                # take the verdicts off k0 and k1, or a selector filtering on
                # `agreement` sees a pool this fixture never meant to build.
                "agreement": pl.when(flags & pl.lit(fetched)).then(3)
                               .otherwise(prior_agreement)}
        # Typed off `metadata_schema`, not all String: the selectors compare
        # `control_frac` against a float, so a null column of the wrong dtype
        # would make this fixture disagree with a real compacted index in the
        # one way that matters.
        for field, dtype in self.probe.metadata_schema().items():
            if field not in frame.columns and field not in cols:
                cols[field] = pl.lit(None, dtype=dtype)
        frame.with_columns(**cols).write_parquet(index.INDEX_PATH)

    def test_a_dialect_selector_draws_only_probed_files(self):
        self._probe(["k0", "k1", "k2"])
        drawn = self.sample.candidates(select="unanimous").collect()
        self.assertEqual(sorted(drawn["content_key"]), ["k0", "k1", "k2"])

    def test_nothing_probed_means_nothing_for_a_dialect_selector(self):
        self.assertEqual(
            self.sample.candidates(select="unanimous").collect().height, 0)

    def test_the_main_corpus_draws_unprobed_files(self):
        """The change this class is named for: `main` filters on nothing a
        probe supplies, so an unprobed pool is a full pool, not an empty one."""
        self._probe(["k0"])
        self.assertEqual(self.sample.candidates(select="main").collect().height, 10)

    def test_the_main_corpus_still_drops_what_the_probe_called_binary(self):
        """Null is "unknown" and stays in; a recorded magic prefix is a fact
        and goes."""
        import polars as pl

        from csvscrape import index

        self._probe(["k0", "k1"])
        frame = pl.read_parquet(index.INDEX_PATH)
        frame.with_columns(
            magic_kind=pl.when(pl.col("content_key") == "k0")
            .then(pl.lit("gzip")).otherwise(pl.lit(None, dtype=pl.String))
        ).write_parquet(index.INDEX_PATH)
        drawn = set(self.sample.candidates(select="main").collect()["content_key"])
        self.assertNotIn("k0", drawn)
        self.assertIn("k1", drawn)
        self.assertEqual(len(drawn), 9)

    def test_the_utf8_flag_keeps_the_undecided_and_drops_the_decided(self):
        """A head sample can only rule UTF-8 *out*; `whole_file_utf8` is what
        rules it in, so an unprobed row has to survive the pool filter."""
        import polars as pl

        from csvscrape import index

        self._probe(["k0", "k1"])
        frame = pl.read_parquet(index.INDEX_PATH)
        frame.with_columns(
            encoding_class=pl.when(pl.col("content_key") == "k0").then(pl.lit("wide"))
            .when(pl.col("content_key") == "k1").then(pl.lit("utf-8"))
            .otherwise(pl.lit(None, dtype=pl.String))
        ).write_parquet(index.INDEX_PATH)
        drawn = set(self.sample.candidates(select="main", utf8=True)
                    .collect()["content_key"])
        self.assertNotIn("k0", drawn)
        self.assertIn("k1", drawn)
        self.assertEqual(len(drawn), 9, "unprobed rows must not be assumed non-UTF-8")


    def test_a_failed_fetch_does_not_count_as_probed(self):
        """A row exists for it, but it carries no verdicts."""
        self._probe(["k0", "k1"])
        self._probe(["k2"], fetched=False)
        self.assertEqual(
            sorted(self.sample.candidates(select="unanimous")
                   .collect()["content_key"]), ["k0", "k1"])

    def test_coverage_counts_the_drawable_share(self):
        self._probe(["k0", "k1", "k2", "k3"])
        cov = self.sample.coverage().to_dicts()
        self.assertEqual(len(cov), 1)
        self.assertEqual(cov[0]["unique_files"], 10)
        self.assertEqual(cov[0]["probed"], 4)


    def test_download_reports_coverage_for_a_selector_that_needs_it(self):
        """An empty pool under a dialect selector is almost always coverage.
        Saying so beats 'no candidates'."""
        from csvscrape import download

        stats = download.run(gh=None, n=5, select="unanimous")
        self.assertIn("probed", stats["error"])
        self.assertIn("0/10", stats["error"].replace(",", ""))

    def test_download_does_not_blame_coverage_for_a_selector_that_does_not(self):
        """`main` draws unprobed files, so an unprobed pool is not its problem
        and telling the user to go and probe would send them nowhere."""
        from csvscrape import download

        stats = download.run(gh=None, n=5, select="main", max_rounds=0)
        self.assertNotIn("error", stats, stats.get("error"))


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class NewestRecordWins(unittest.TestCase):
    """Retries append; readers pick a winner. It must be picked from recorded
    fields, not from whatever order the shards happened to be scanned in."""

    def setUp(self):
        import importlib

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe, self.log = probe, log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _write(self, rows, rotate=10_000):
        with self.log.Writer(self.probe.HEADS_LOG, rotate=rotate) as w:
            for r in rows:
                w.write(r)

    def _rec(self, delim, ts, fetched=True):
        flat = {f: None for f in self.probe.FLAT_FIELDS}
        return {"content_key": "k", "fetched": fetched, "http": 200, "ts": ts,
                "agreement": 1, "is_lfs_pointer": False, "looks_like_html": False,
                "custom": flat | {"delimiter": delim}}

    def test_newest_by_timestamp_wins(self):
        self._write([self._rec(",", 100), self._rec(";", 200)])
        m = self.probe.metadata().collect()
        self.assertEqual(m.height, 1)
        self.assertEqual(m["delimiter"][0], ";")

    def test_an_older_record_appended_later_does_not_win(self):
        """The case scan order gets wrong: a stale record written after a newer
        one -- a slow worker finishing late, or shards read out of order."""
        self._write([self._rec(";", 200), self._rec(",", 100)])
        m = self.probe.metadata().collect()
        self.assertEqual(m["delimiter"][0], ";", "scan order beat the timestamp")

    def test_same_timestamp_breaks_on_rec_id(self):
        self._write([self._rec(",", 100), self._rec(";", 100)])
        m = self.probe.metadata().collect()
        self.assertEqual(m["delimiter"][0], ";")

    def test_winner_survives_being_split_across_shards(self):
        self._write([self._rec(";", 200), self._rec(",", 100)], rotate=1)
        shards = list((self.tmp / self.probe.HEADS_LOG).iterdir())
        self.assertGreater(len(shards), 1)
        self.assertEqual(self.probe.metadata().collect()["delimiter"][0], ";")


    def _keyed(self, key, delim, ts):
        return self._rec(delim, ts) | {"content_key": key}

    def test_deduping_per_bucket_matches_deduping_the_whole_log(self):
        """`compact` dedups one content-key bucket at a time. That gives the same
        answer only because every record for a key lands in the same
        bucket."""
        import polars as pl
        from polars.testing import assert_frame_equal

        shards = 8
        rows = []
        for i in range(50):
            key = f"k{i}"
            # Losers written after winners, so scan order disagrees with ts and
            # only the recorded ordering can get this right.
            rows.append(self._keyed(key, ";", 200))
            rows.append(self._keyed(key, ",", 100))
        self._write(rows, rotate=7)

        whole = self.probe.metadata().collect().sort("content_key")
        raw = self.probe.metadata_rows()
        per_bucket = pl.concat([
            self.probe.dedupe(
                raw.filter((pl.col("content_key").hash(seed=0) % shards) == b)
            ).collect()
            for b in range(shards)
        ]).sort("content_key")

        self.assertEqual(whole.height, 50)
        self.assertEqual(per_bucket.height, 50)
        self.assertEqual(set(per_bucket["delimiter"].to_list()), {";"})
        assert_frame_equal(whole, per_bucket)



@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class CompactFoldsMetadata(unittest.TestCase):
    """`compact` folds probe verdicts into the index as columns."""

    def setUp(self):
        import importlib

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe, sample

        for mod in (config, log, index, probe, sample):
            importlib.reload(mod)
        self.index, self.probe, self.log, self.sample = index, probe, log, sample

        trees = self.tmp / "trees"
        trees.mkdir(parents=True)
        with self.log.Writer("trees", rotate=1000) as w:
            w.write({"repo_id": 1, "full_name": "o/a", "ref": "HEAD", "state": "ok",
                     "truncated": False, "capped": False, "n_blobs": 2, "n_kept": 2,
                     "tier": "hi", "ts": 1,
                     "paths": ["a.csv", "b.csv"], "shas": ["sa", "sb"],
                     "sizes": [4096, 8192]})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _probe_log(self, key, delim, ts=10):
        flat = {f: None for f in self.probe.FLAT_FIELDS}
        with self.log.Writer(self.probe.HEADS_LOG, rotate=1000) as w:
            w.write({"content_key": key, "fetched": True, "http": 200, "ts": ts,
                     "agreement": 3, "is_lfs_pointer": False,
                     "looks_like_html": False, "custom": flat | {"delimiter": delim}})

    def _failed_log(self, key, http=404, ts=20, via="raw"):
        """What the probe writes when there are no bytes: an outcome and
        nothing else. No `custom`, no `agreement` -- `describe` never gets far
        enough to sniff anything."""
        with self.log.Writer(self.probe.HEADS_LOG, rotate=1000) as w:
            w.write({"content_key": key, "fetched": False, "http": http,
                     "via": via, "ts": ts})


    def test_a_selector_survives_an_index_that_has_no_metadata_columns(self):
        """`main` names columns `compact` only writes once there is a heads log,
        so on an unprobed pool they must be supplied as nulls."""
        self.index.build()
        drawn = self.sample.candidates(select="main").collect()
        self.assertEqual(sorted(drawn["content_key"]), ["sa", "sb"])

    def test_compact_folds_verdicts_into_the_index(self):
        self._probe_log("sa", ";")
        self.index.build()
        rows = {r["content_key"]: r for r in self.index.files().collect().to_dicts()}
        self.assertEqual(rows["sa"]["delimiter"], ";")
        self.assertTrue(rows["sa"]["fetched"])
        self.assertIsNone(rows["sb"]["fetched"])

    def test_a_dead_path_survives_the_fold(self):
        """A 404 must reach the index, or once the log falls behind the watermark
        nothing remembers the path is dead."""
        self._failed_log("sa")
        self.index.build()
        row = {r["content_key"]: r
               for r in self.index.files().collect().to_dicts()}["sa"]
        self.assertFalse(row["fetched"])
        self.assertEqual(row["http"], 404)
        self.assertEqual(row["via"], "raw")
        self.assertIsNone(row["delimiter"], "a failure carried a verdict")

    def test_a_folded_404_stays_retired(self):
        """The property the fold exists for: after the log falls behind the
        watermark, the index alone has to answer that this path is dead."""
        self._failed_log("sa")
        self.index.build()
        self.assertIn("sa", set(self.probe.retired_keys()
                                .collect()["content_key"]))
        self.assertEqual(self.probe.outstanding_size(), 1)
        self.assertEqual([r["content_key"] for r in self.probe.candidates()],
                         ["sb"])

    def test_a_later_failure_does_not_erase_the_verdicts(self):
        """A content key is a content address: the bytes were described, and
        the path failing afterwards does not undescribe them."""
        self._probe_log("sa", ";", ts=10)
        self._failed_log("sa", ts=20)
        self.index.build()
        row = {r["content_key"]: r
               for r in self.index.files().collect().to_dicts()}["sa"]
        self.assertTrue(row["fetched"], "a 404 overwrote a described file")
        self.assertEqual(row["delimiter"], ";")

    def test_a_later_success_replaces_the_failure(self):
        self._failed_log("sa", http=503, ts=10)
        self._probe_log("sa", "|", ts=20)
        self.index.build()
        row = {r["content_key"]: r
               for r in self.index.files().collect().to_dicts()}["sa"]
        self.assertTrue(row["fetched"])
        self.assertEqual(row["delimiter"], "|")

    def test_a_log_of_nothing_but_failures_still_folds(self):
        """A log with no successful fetch has no `custom` or `agreement` column,
        and must still fold its outcomes."""
        self._failed_log("sa")
        self._failed_log("sb", http=403)
        self.index.build()
        rows = {r["content_key"]: r
                for r in self.index.files().collect().to_dicts()}
        self.assertEqual(rows["sa"]["http"], 404)
        self.assertEqual(rows["sb"]["http"], 403)
        self.assertEqual(self.probe.outstanding_size(), 1,
                         "the 403 is transient and stays in the queue")

    def test_only_folded_rows_are_drawable_by_a_dialect_selector(self):
        self._probe_log("sa", ",")
        self.index.build()
        drawn = self.sample.candidates(select="unanimous").collect()
        self.assertEqual(drawn["content_key"].to_list(), ["sa"])
        # ...and the unfolded row is still in the pool for a corpus that does
        # not need its dialect.
        self.assertEqual(
            sorted(self.sample.candidates(select="main").collect()["content_key"]),
            ["sa", "sb"])

    def test_watermark_records_what_was_folded(self):
        self._probe_log("sa", ",")
        self.index.build()
        mark = self.index.heads_watermark()
        self.assertTrue(mark, "no watermark written")
        self.assertRegex(mark, r"^\d{5}:\d{8}$")

    def test_probe_sees_records_written_after_the_watermark(self):
        """The whole point of the watermark: fresh probes must count as
        retired without re-reading the compacted part of the log."""
        self._probe_log("sa", ",")
        self.index.build()
        self._probe_log("sb", "|")          # after compact, not in the index
        retired = set(self.probe.retired_keys().collect()["content_key"])
        self.assertEqual(retired, {"sa", "sb"})


    def _bulk_log(self, records):
        flat = {f: None for f in self.probe.FLAT_FIELDS}
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as w:
            for key, extra, top in records:
                w.write({"content_key": key, "fetched": True, "ts": 10,
                         "agreement": 3, "is_lfs_pointer": False,
                         "looks_like_html": False,
                         "custom": flat | {"delimiter": ","} | extra} | top)

    def test_shape_field_null_across_the_inference_window(self):
        """`scan_ndjson` types a field from the first rows of the log, and most
        of a Shape is null on most files. The first CSV carrying a comment
        character then arrived as String against a Null column and took the
        whole of `compact` down with a SchemaError."""
        self._bulk_log(
            [(f"x{i}", {"comment": None}, {"http": 200}) for i in range(150)]
            + [("sa", {"comment": "#"}, {"http": 200})]
        )
        self.index.build()
        rows = {r["content_key"]: r for r in self.index.files().collect().to_dicts()}
        self.assertEqual(rows["sa"]["comment"], "#",
                         "the comment character was lost or mistyped")

    def test_top_level_field_null_across_the_inference_window(self):
        """Same trap one level up: `http` is null on every record until the
        first fetch that returns one, which is what `_fresh_log` reads."""
        self._bulk_log(
            [(f"x{i}", {}, {"http": None, "fetched": False}) for i in range(150)]
            + [("sa", {}, {"http": 404, "fetched": False})]
        )
        retired = set(self.probe.retired_keys().collect()["content_key"])
        self.assertIn("sa", retired, "a 404 must retire the key")
        self.assertNotIn("x0", retired, "a null http is transient, not settled")


class CodeSearchRateLimit(unittest.TestCase):
    """Search pacing: a fixed interval, plus the budget the server reports."""


    def test_a_hold_leaves_no_credit_to_burst_with(self):
        """Why search is paced rather than bucketed: a pacer's whole state is one
        instant, so a hold leaves no credit to burst with afterwards."""
        from csvscrape.client import Pacer

        pacer = Pacer(0.05)
        pacer.hold_for(0.2)
        start = time.monotonic()
        stamps = []
        for _ in range(3):
            pacer.take()
            stamps.append(time.monotonic() - start)

        self.assertGreaterEqual(stamps[0], 0.2, "the hold was not enforced")
        for before, after in zip(stamps, stamps[1:]):
            self.assertGreaterEqual(after - before, 0.05,
                                    "burst released on the far side of a hold")


    def test_the_budget_is_learned_from_response_headers(self):
        """From the x-ratelimit-* headers on every response, which also show
        another machine's use of a shared token. (/rate_limit lags
        them.)"""
        from csvscrape.client import GitHub, Pacer

        pacer = Pacer(6.67)
        pacer.take = lambda on_wait=None: None
        reset = time.time() + 20

        class Resp:
            status_code = 200
            headers = {"x-ratelimit-remaining": "2",
                       "x-ratelimit-reset": str(reset)}
            @staticmethod
            def json():
                return {}

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda *a, **k: None
        gh.session = type("S", (), {"request": staticmethod(lambda *a, **k: Resp())})()
        gh._request("GET", "https://example.invalid", pacer)

        # Two left cannot be spread over the 20s remaining at 6.67s spacing, so
        # the window is waited out rather than overspent.
        self.assertAlmostEqual(pacer._next - time.monotonic(), 20.0, delta=1.5)

    def test_sync_spends_a_budget_that_covers_the_rest_of_the_window(self):
        """The other side of it: waiting out a window we are entitled to use
        would forfeit real requests on every restart."""
        from csvscrape import config
        from csvscrape.client import Pacer

        pacer = Pacer(config.CODE_SEARCH_INTERVAL)
        pacer.sync(remaining=9, reset=time.time() + 20)
        self.assertLessEqual(pacer._next, time.monotonic(),
                             "held despite budget enough for the window")


class RetryAfterFloor(unittest.TestCase):
    """GitHub answers a spent search budget with `retry-after: 1`. Retrying that
    soon cannot succeed and still counts against the limit, so the wait is
    floored at the pacing interval."""

    def test_a_short_retry_after_is_raised_to_the_pacing_interval(self):
        from csvscrape.client import GitHub, Pacer

        pacer = Pacer(6.0)
        holds, events = [], []
        pacer.hold_for = holds.append
        pacer.take = lambda on_wait=None: None  # the floor is under test

        class Resp:
            status_code = 403
            headers = {"retry-after": "1"}
            content = b""

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda event, **kw: events.append((event, kw))
        gh.session = type(
            "S", (), {"request": staticmethod(lambda *a, **k: Resp())})()

        gh._request("GET", "https://example.invalid", pacer)

        self.assertTrue(holds, "no backoff applied to a 403")
        self.assertTrue(all(h >= pacer.min_interval for h in holds),
                        f"retried sooner than the pacing allows: {holds}")
        self.assertEqual(len(set(holds)), 1,
                         f"the interval moved during one request: {holds}")
        announced = [kw["seconds"] for event, kw in events if event == "wait"]
        self.assertTrue(all(w >= pacer.min_interval for w in announced),
                        "announced the server's figure, not the wait applied")
        # One announcement, six holds: every attempt of one request is refused
        # by the same congestion event, and only the first of them reports it.
        # The floor still has to hold for all six, which is what `holds` checks
        # above -- what is under test here is that what was *said* is what was
        # done, not how many times it was said.
        self.assertEqual(len(announced), 1,
                         f"one event announced more than once: {announced}")
        self.assertEqual(set(announced), set(holds),
                         "the announced wait is not the one applied")

    def test_a_long_retry_after_is_honoured_as_given(self):
        """The floor must never shorten a wait the server actually asked for."""
        from csvscrape.client import Pacer

        pacer = Pacer(6.0)
        self.assertEqual(max(120.0, pacer.min_interval), 120.0)


class WaitsAreAttributed(unittest.TestCase):
    """Waits are reported as pacing (ours) or throttling (the server's)."""


    def test_a_server_hold_is_attributed_to_the_server(self):
        from csvscrape.client import Pacer

        pacer = Pacer(0.01)
        pacer.hold_for(0.2)
        seen = []
        pacer.take(lambda seconds, throttled: seen.append(throttled))
        self.assertEqual(seen, [True])

    def test_the_attribution_is_consumed_by_the_wait_that_serves_it(self):
        """Otherwise every request after one 403 would report as throttled."""
        from csvscrape.client import Pacer

        pacer = Pacer(0.05)
        pacer.hold_for(0.1)
        pacer.take()
        seen = []
        pacer.take(lambda seconds, throttled: seen.append(throttled))
        self.assertEqual(seen, [False], "the hold was already waited out")


class SizeBucketIsSharedNotDuplicated(unittest.TestCase):
    """`download` compares an as-downloaded size against the bucket the index
    recorded. Two definitions that drifted apart would silently mis-stratify
    the corpus."""

    def test_it_agrees_with_the_index_expression(self):
        import math

        from csvscrape.index import size_bucket

        cases = ([0, 1, 2, 3, 63, 64, 65, 1023, 1024, 512_000]
                 + [2 ** k for k in range(31)]
                 + [2 ** k - 1 for k in range(1, 31)])
        for n in cases:
            want = 0 if n < 1 else int(math.floor(math.log2(n)))
            self.assertEqual(size_bucket(n), want, n)

    def test_an_empty_file_does_not_land_below_zero(self):
        from csvscrape.index import size_bucket
        self.assertEqual(size_bucket(0), 0)


class TheBytesMustBeTheOnesTheIndexMeasured(unittest.TestCase):
    """A downloaded file must hash to the content key the index measured: anything
    else carries another file's size, bucket and dialect, and would be
    filed in the store under a key it does not hash to. Fetches are pinned
    to the scanned commit, so a mismatch is a failure, not drift."""

    def _fetch(self, served: bytes, indexed_size: int, key: str | None = None,
               dataset_verify=None):
        """Run fetch_one against a server serving `served` for a row the index
        recorded at `indexed_size`, and return the outcome record.

        `key` defaults to a content key `served` does *not* hash to, which is
        the mismatch case; pass the real one for the agreeing case.
        """
        import hashlib

        from csvscrape import config, download, index

        # `download.CORPUS` is bound at import from config.DATA_ROOT, so both
        # have to move together or `relative_to` fails on the mismatch.
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        self.addCleanup(setattr, download, "CORPUS", download.CORPUS)
        config.DATA_ROOT = root
        download.CORPUS = root / "corpus"
        (root / "corpus").mkdir(parents=True, exist_ok=True)

        class GH:
            def stream(self, url, sink, chunk=1 << 20):
                sink.write(served)
                return len(served), hashlib.sha256(served).hexdigest()

        row = {"content_key": key or "k" * 40, "repo_id": 1, "full_name": "o/r",
               "path": "a.csv", "size": indexed_size,
               "bucket": index.size_bucket(indexed_size),
               "is_lfs": False, "blob_sha": "0" * 40, "ref": "e" * 40}
        return download.fetch_one(GH(), row, dataset_verify)

    @staticmethod
    def _key(data: bytes) -> str:
        from csvscrape.validate import git_blob_sha1

        return git_blob_sha1(len(data), iter([data]))

    def test_the_download_is_pinned_to_the_scanned_commit(self):
        """`fetch_one` addressed a branch tip, so a repo pushed to since the
        scan served different bytes and the identity check below rejected them.
        Only changing repos are ever rejected, so an unpinned fetch biased the
        corpus towards dormant ones -- dropping exactly the regularly-published
        datasets the pass exists to collect."""
        import hashlib

        from csvscrape import config, download, index

        seen = {}

        class GH:
            def stream(self, url, sink, chunk=1 << 20):
                seen["url"] = url
                sink.write(b"a,b\n")
                return 4, hashlib.sha256(b"a,b\n").hexdigest()

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        self.addCleanup(setattr, download, "CORPUS", download.CORPUS)
        config.DATA_ROOT = root
        download.CORPUS = root / "corpus"
        (root / "corpus").mkdir(parents=True, exist_ok=True)

        sha = "f" * 40
        for is_lfs, host in ((False, "raw.githubusercontent.com"),
                             (True, "media.githubusercontent.com")):
            download.fetch_one(GH(), {
                "content_key": "k" * 64, "repo_id": 1, "full_name": "o/r",
                "path": "a.csv", "size": 4, "bucket": index.size_bucket(4),
                "is_lfs": is_lfs, "blob_sha": "0" * 40, "ref": sha})
            self.assertIn(host, seen["url"])
            self.assertIn(f"/o/r/{sha}/", seen["url"])
            self.assertNotIn("/HEAD/", seen["url"])

    def test_the_indexed_blob_is_accepted(self):
        data = b"a,b\n1,2\n" * 900
        rec = self._fetch(data, len(data), key=self._key(data))
        self.assertEqual(rec["state"], "ok", rec.get("reason"))
        self.assertEqual(rec["bytes"], len(data))

    def test_different_bytes_are_rejected_even_inside_the_same_octave(self):
        """A mismatch is rejected even when the size stays in the same bucket: the
        row's dialect and encoding would still describe another file."""
        from csvscrape.index import size_bucket

        served = b"a,b\n1,2\n" * 900
        self.assertEqual(size_bucket(8_000), size_bucket(len(served)))
        rec = self._fetch(served, 8_000)
        self.assertEqual(rec["state"], "rejected")
        self.assertEqual(rec["reason"], "content_mismatch")
        self.assertEqual(rec["indexed_size"], 8_000)
        self.assertEqual(rec["actual_size"], len(served))
        self.assertEqual(rec["ref"], "e" * 40, "the pin that failed is not recorded")

    def test_a_rejected_blob_leaves_nothing_in_the_store(self):
        """The store is keyed by content, so a near-miss left under the key it
        was drawn for would be served as that file to every later dataset."""
        from csvscrape import download

        rec = self._fetch(b"a,b\n1,2\n" * 900, 8_000)
        self.assertEqual(rec["state"], "rejected")
        self.assertFalse([p for p in download.CORPUS.rglob("*") if p.is_file()],
                         "rejected bytes were left in the corpus")

    def test_a_dataset_reuses_a_blob_another_dataset_already_fetched(self):
        """One shared content-addressed store under many manifests: the second
        corpus to draw a file must not spend the transfer again."""
        import hashlib

        from csvscrape import config, download, index

        data = b"a,b\n1,2\n" * 900
        key = self._key(data)
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        self.addCleanup(setattr, download, "CORPUS", download.CORPUS)
        config.DATA_ROOT = root
        download.CORPUS = root / "corpus"

        calls = []

        class GH:
            def stream(self, url, sink, chunk=1 << 20):
                calls.append(url)
                sink.write(data)
                return len(data), hashlib.sha256(data).hexdigest()

        row = {"content_key": key, "repo_id": 1, "full_name": "o/r",
               "path": "a.csv", "size": len(data),
               "bucket": index.size_bucket(len(data)), "is_lfs": False,
               "blob_sha": "0" * 40, "ref": "e" * 40}
        first = download.fetch_one(GH(), row)
        second = download.fetch_one(GH(), row)
        self.assertEqual(first["state"], "ok", first.get("reason"))
        self.assertEqual(second["state"], "ok", second.get("reason"))
        self.assertEqual(len(calls), 1, "the second dataset re-downloaded it")
        self.assertFalse(first["reused"])
        self.assertTrue(second["reused"])

    def test_a_reused_blob_still_faces_this_dataset_s_own_verifier(self):
        """Reuse is of the bytes, not of the verdict: the selector that admits
        a file to the second corpus is not the one that admitted it to the
        first."""
        from csvscrape import download

        data = b"a,b\n1,2\n" * 900
        key = self._key(data)
        # `_fetch` puts it in the store (and leaves CORPUS pointed there).
        self.assertEqual(self._fetch(data, len(data), key=key)["state"], "ok")
        self.assertTrue(download._target(key).exists())
        # A second dataset draws the same file under a selector that says no.
        # `gh=None` is the assertion that nothing reaches the network.
        rec = download.fetch_one(
            None, {"content_key": key, "repo_id": 1, "full_name": "o/r",
                   "path": "a.csv", "size": len(data), "bucket": 12,
                   "is_lfs": False, "blob_sha": "0" * 40, "ref": "e" * 40},
            verify=lambda path, row: "nope")
        self.assertEqual(rec["state"], "rejected")
        self.assertEqual(rec["reason"], "verify/nope")


class ARetryRewindsTheSink(unittest.TestCase):
    """A retry re-sends the body from byte zero, so the failed attempt's bytes
    have to go first. Appending them instead put prefix-then-whole-file on disk
    under a digest computed over the whole file alone -- and for LFS, where the
    oid is checked against that digest rather than against the bytes, nothing
    downstream could tell."""

    def gh(self, session):
        from csvscrape.client import GitHub

        gh = GitHub.__new__(GitHub)
        gh.anon = session
        gh.raw = self.pool()
        return gh

    @staticmethod
    def pool():
        class Raw:
            def slot(self):
                return contextlib.nullcontext()

            def reward(self):
                pass

            def penalise(self, *a, **k):
                pass

        return Raw()

    @staticmethod
    def session(script):
        """A session whose n-th response follows `script[n]`.

        Each entry is (blocks, die_at): the body to yield, and the block index
        at which the connection drops, or None for a clean response.
        """
        import requests

        class Resp:
            def __init__(self, blocks, die_at):
                self.status_code, self.headers = 200, {}
                self._blocks, self._die_at = blocks, die_at

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def iter_content(self, chunk_size=0):
                for i, block in enumerate(self._blocks):
                    if i == self._die_at:
                        raise requests.ConnectionError("reset")
                    yield block

        class Session:
            def __init__(self):
                self.calls = []

            def get(self, url, **kw):
                self.calls.append(url)
                return Resp(*script[len(self.calls) - 1])

        return Session()

    def test_a_dropped_body_does_not_prepend_itself_to_the_retry(self):
        import io

        body = [b"A" * 10, b"B" * 10, b"C" * 10]
        sink = io.BytesIO()
        gh = self.gh(self.session([(body, 1), (body, None)]))

        result = gh.stream("http://x", sink)

        self.assertIsNotNone(result, "the clean retry should still succeed")
        written, sha256 = result
        self.assertEqual(sink.getvalue(), b"".join(body),
                         "the failed attempt's bytes survived into the file")
        self.assertEqual(written, len(sink.getvalue()),
                         "reported byte count disagrees with the file")

    def test_the_digest_describes_the_bytes_that_landed(self):
        """The check that matters for LFS: `fetch_one` compares the oid against
        this digest and never re-reads the file, so the two agreeing is the
        only thing standing between a torn transfer and the corpus."""
        import hashlib
        import io

        body = [b"A" * 10, b"B" * 10, b"C" * 10]
        sink = io.BytesIO()
        gh = self.gh(self.session([(body, 2), (body, None)]))

        _, sha256 = gh.stream("http://x", sink)

        self.assertEqual(sha256, hashlib.sha256(sink.getvalue()).hexdigest())


    def test_a_sink_that_cannot_rewind_is_abandoned_not_appended_to(self):
        """`None` is a failed fetch, which costs one redraw. Appending would
        cost a corrupt file that every check downstream reports as clean."""

        class Unrewindable:
            def __init__(self):
                self.buf = bytearray()

            def write(self, block):
                self.buf += block

            def tell(self):
                raise OSError("not seekable")

        body = [b"A" * 10, b"B" * 10]
        sink = Unrewindable()
        gh = self.gh(self.session([(body, 1), (body, None)]))

        self.assertIsNone(gh.stream("http://x", sink))



class BigFilesTakeTheRangeRequest(unittest.TestCase):
    """Files above PROBE_GRAPHQL_MAX_BYTES take a range request instead of a
    GraphQL batch, and cost the batch nothing."""

    def test_a_file_over_the_threshold_is_routed_out(self):
        from csvscrape import config, probe

        under = {"is_lfs": False, "size": config.PROBE_GRAPHQL_MAX_BYTES}
        over = {"is_lfs": False, "size": config.PROBE_GRAPHQL_MAX_BYTES + 1}
        self.assertTrue(probe._via_graphql(under))
        self.assertFalse(probe._via_graphql(over))

    def test_a_routed_out_row_does_not_consume_the_byte_budget(self):
        """The batch closes on `GRAPHQL_BATCH_BYTES`, which is a bound on the
        *response*. Counting files that are not going into the response ended
        the batch early and sent a nearly empty request."""
        from csvscrape import probe

        big = {"is_lfs": False, "size": 50 << 20}
        self.assertEqual(probe._graphql_cost(big), 0)
        self.assertEqual(probe._graphql_cost({"is_lfs": True, "size": 133}), 0)
        small = {"is_lfs": False, "size": 4096}
        self.assertEqual(probe._graphql_cost(small), 4096)


    def test_a_batch_of_only_large_files_still_makes_progress(self):
        """Every row routed out means no GraphQL call at all -- which must read
        as "all of them take the range request", not as an empty batch that
        drops the work."""
        from csvscrape import probe

        rows = [{"is_lfs": False, "size": 50 << 20} for _ in range(5)]
        self.assertEqual([r for r in rows if probe._via_graphql(r)], [])
        self.assertEqual(sum(probe._graphql_cost(r) for r in rows), 0)


class BatchedBlobReads(unittest.TestCase):
    """When a GraphQL blob body must not be trusted, so the caller falls back to a
    range request."""

    def gh_with(self, blobs, status=200, errors=None):
        from csvscrape.client import Bucket, GitHub

        # One repo per blob, each holding a single object, because `rows` below
        # gives every row its own repo. The aliases are `r{repo}.b{blob}` since
        # the query became repo-grouped; they were `r{row}.o` when every row
        # carried its own `repository()`.
        data = {f"r{i}": ({"b0": b} if b is not None else None)
                for i, b in enumerate(blobs)}

        class Resp:
            status_code = status
            headers: dict = {}
            content = b""
            @staticmethod
            def json():
                return {"data": data} | ({"errors": errors} if errors else {})

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda *a, **k: None
        gh.graphql_bucket = Bucket(3600.0, 5000, capacity=10)
        gh.session = type("S", (), {"request": staticmethod(lambda *a, **k: Resp())})()
        return gh

    def rows(self, n):
        return [{"full_name": f"o{i}/r{i}", "blob_sha": f"{i:040x}"} for i in range(n)]

    def test_whole_small_blobs_come_back(self):
        body = b"a,b,c\n1,2,3\n"
        gh = self.gh_with([{"text": body.decode(), "byteSize": len(body),
                            "isBinary": False, "isTruncated": False}])
        self.assertEqual(gh.blobs(self.rows(1)), [body])

    def test_a_truncated_blob_is_a_head_not_a_failure(self):
        """GitHub caps `text` at GRAPHQL_TEXT_CAP and returns a clean byte
        prefix, with `byteSize` still the file's true length. Checking the
        length against `byteSize` there would reject every large file and
        quietly undo the batching."""
        from csvscrape import config

        body = b"x" * config.GRAPHQL_TEXT_CAP
        gh = self.gh_with([{"text": body.decode(), "byteSize": 9_000_000,
                            "isBinary": True and False, "isTruncated": True}])
        self.assertEqual(gh.blobs(self.rows(1)), [body])

    def test_a_truncated_blob_may_miss_the_cap_by_a_few_bytes(self):
        """GitHub cuts the byte stream at the cap and then decodes, so a cut
        through a multi-byte character can leave the body a few bytes
        off the cap."""
        from csvscrape import config

        for delta in (-2, 0, +2):
            body = b"x" * (config.GRAPHQL_TEXT_CAP + delta)
            gh = self.gh_with([{"text": body.decode(), "byteSize": 9_000_000,
                                "isBinary": False, "isTruncated": True}])
            self.assertEqual(gh.blobs(self.rows(1)), [body], f"delta {delta}")

    def test_the_boundary_replacement_is_trimmed_to_a_clean_prefix(self):
        """GitHub cuts the byte stream at the cap and the trailing partial
        character becomes U+FFFD -- verified against the same blob over raw,
        where the real bytes continue. Trimming it makes the body a byte-exact
        prefix instead of one with three foreign bytes on the end."""
        from csvscrape import config

        real = b"a,b\n" * ((config.GRAPHQL_TEXT_CAP // 4) - 1)
        body = real + b"\xef\xbf\xbd"
        gh = self.gh_with([{"text": body.decode(), "byteSize": 9_000_000,
                            "isBinary": False, "isTruncated": True}])
        self.assertEqual(gh.blobs(self.rows(1)), [real])

    def test_a_replacement_anywhere_else_falls_back(self):
        """U+FFFD in the body means the bytes were not valid UTF-8 and came
        back re-encoded. A length check cannot catch that -- a substitution can
        land on the same length -- and silently mangled content is the one
        outcome worse than a missing file."""
        body = b"a,b\n1,\xef\xbf\xbd2\n" + b"c,d\n" * 40
        gh = self.gh_with([{"text": body.decode(), "byteSize": len(body),
                            "isBinary": False, "isTruncated": False}])
        self.assertEqual(gh.blobs(self.rows(1)), [None],
                         "kept a body the server had re-encoded")


    def test_a_short_truncated_blob_is_rejected(self):

        gh = self.gh_with([{"text": "x" * 100, "byteSize": 9_000_000,
                            "isBinary": False, "isTruncated": True}])
        self.assertEqual(gh.blobs(self.rows(1)), [None])

    def test_binary_falls_back(self):
        """git calls a blob binary on a NUL in the first 8000 bytes, so a
        UTF-16 CSV lands here -- exactly the population a corpus about
        encodings cannot afford to drop silently."""
        gh = self.gh_with([{"text": None, "byteSize": 40,
                            "isBinary": True, "isTruncated": False}])
        self.assertEqual(gh.blobs(self.rows(1)), [None])

    def test_a_body_that_does_not_account_for_falls_back(self):
        """`text` is a string; a body that is not valid UTF-8 could come back re-
        encoded, and a silently mangled file is worse than a missing
        one."""
        gh = self.gh_with([{"text": "abc", "byteSize": 99,
                            "isBinary": False, "isTruncated": False}])
        self.assertEqual(gh.blobs(self.rows(1)), [None])


    def test_a_failed_request_is_raised_not_silently_nulled(self):
        """A failed request raises rather than returning None for every row, which
        would look like GitHub declining each blob."""
        # 400, not 5xx: `_request` retries a server error six times with
        # exponential sleep, which would put 63 seconds into the suite.
        gh = self.gh_with([None] * 3, status=400)
        with self.assertRaises(RuntimeError):
            gh.blobs(self.rows(3))


    def test_no_rows_makes_no_request(self):
        from csvscrape.client import GitHub
        gh = GitHub.__new__(GitHub)
        self.assertEqual(gh.blobs([]), [])


class OneResolutionPerRepository(unittest.TestCase):
    """Blob queries are grouped by repository: a `repository()` resolution costs
    far more than an `object()` lookup inside one."""

    def gh(self, payload, capture):
        from csvscrape.client import Bucket, GitHub

        class Resp:
            status_code = 200
            headers: dict = {}
            content = b""
            @staticmethod
            def json():
                return payload

        def request(method, url, **kw):
            capture.append(kw["json"]["query"])
            return Resp()

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda *a, **k: None
        gh.graphql_bucket = Bucket(3600.0, 5000, capacity=10)
        gh.session = type("S", (), {"request": staticmethod(request)})()
        return gh

    @staticmethod
    def blob(text):
        return {"text": text, "byteSize": len(text.encode()),
                "isBinary": False, "isTruncated": False}

    def test_one_repository_field_per_repo_not_per_row(self):
        rows = [{"full_name": "o/a", "blob_sha": "1" * 40},
                {"full_name": "o/a", "blob_sha": "2" * 40},
                {"full_name": "o/a", "blob_sha": "3" * 40},
                {"full_name": "o/b", "blob_sha": "4" * 40}]
        seen: list = []
        payload = {"data": {"r0": {"b0": self.blob("a"), "b1": self.blob("bb"),
                                   "b2": self.blob("ccc")},
                            "r1": {"b0": self.blob("dddd")}}}
        got = self.gh(payload, seen).blobs(rows)

        self.assertEqual(got, [b"a", b"bb", b"ccc", b"dddd"])
        self.assertEqual(seen[0].count("repository("), 2,
                         "one repository() per row rather than per repo")
        self.assertEqual(seen[0].count("object("), 4)

    def test_results_come_back_in_row_order_not_grouped_order(self):
        """The grouping is an implementation detail of the request; the caller
        indexes the result against the rows it passed in. `probe.run` zips this
        with its batch, so a permutation here mislabels every file in it."""
        rows = [{"full_name": "o/a", "blob_sha": "1" * 40},
                {"full_name": "o/b", "blob_sha": "2" * 40},
                {"full_name": "o/a", "blob_sha": "3" * 40}]
        payload = {"data": {"r0": {"b0": self.blob("first"),
                                   "b1": self.blob("third")},
                            "r1": {"b0": self.blob("second")}}}
        got = self.gh(payload, []).blobs(rows)

        self.assertEqual(got, [b"first", b"second", b"third"],
                         "interleaved repos came back in grouped order")

    def test_a_missing_repo_leaves_only_its_own_rows_none(self):
        rows = [{"full_name": "o/gone", "blob_sha": "1" * 40},
                {"full_name": "o/here", "blob_sha": "2" * 40}]
        payload = {"data": {"r0": None, "r1": {"b0": self.blob("ok")}}}
        self.assertEqual(self.gh(payload, []).blobs(rows), [None, b"ok"])

    def test_resource_limits_raise_rather_than_returning_nones(self):
        """An over-large batch comes back HTTP 200 with mostly-null `data` and one
        RESOURCE_LIMITS_EXCEEDED error per refused alias. It must
        raise, not look like declined blobs."""
        rows = [{"full_name": f"o/r{i}", "blob_sha": f"{i:040x}"} for i in range(3)]
        payload = {
            "data": {"r0": {"b0": self.blob("served")}, "r1": None, "r2": None},
            "errors": [{"type": "RESOURCE_LIMITS_EXCEEDED", "path": ["r1"],
                        "message": "Resource limits for this query exceeded."},
                       {"type": "RESOURCE_LIMITS_EXCEEDED", "path": ["r2"],
                        "message": "Resource limits for this query exceeded."}],
        }
        with self.assertRaises(RuntimeError) as caught:
            self.gh(payload, []).blobs(rows)
        self.assertIn("resource limits", str(caught.exception).lower())

    def test_an_unrelated_error_is_not_a_batch_failure(self):
        """A repo deleted since the scan errors for its own alias and nothing
        else. Raising on that would send the whole batch down the metered path
        because one row went missing."""
        rows = [{"full_name": "o/gone", "blob_sha": "1" * 40},
                {"full_name": "o/here", "blob_sha": "2" * 40}]
        payload = {
            "data": {"r0": None, "r1": {"b0": self.blob("ok")}},
            "errors": [{"type": "NOT_FOUND", "path": ["r0"],
                        "message": "Could not resolve to a Repository"}],
        }
        self.assertEqual(self.gh(payload, []).blobs(rows), [None, b"ok"])


class ThePassLoopStopsOnAnEmptyDraw(unittest.TestCase):
    """A probe run ends on an empty draw or on a pass that settles nothing -- not
    on a short draw, which `--order repo` produces routinely."""

    def _passes(self, draws, retired_per_pass=None):
        """Run cli's loop against scripted `probe.run` results; count passes."""
        import argparse

        from csvscrape import cli, config

        seen = []

        def fake_run(gh, **kw):
            # The last entry repeats rather than the script running out, so a
            # loop that fails to terminate actually fails to terminate -- a
            # script that ends by returning 0 would stop even a broken
            # condition, and prove nothing about the stall case.
            i = len(seen)
            queued = draws[min(i, len(draws) - 1)]
            retired = (retired_per_pass[min(i, len(retired_per_pass) - 1)]
                       if retired_per_pass else queued)
            seen.append(queued)
            if len(seen) > 50:
                raise AssertionError("the pass loop never terminated")
            return {"probed": queued, "queued": queued, "retired": retired}

        args = argparse.Namespace(
            limit=None, workers=1, raw_workers=1, cache=False, cache_dir=None,
            head_bytes=None, sniffers="custom", seed=0, sniff_procs=0,
            order="repo", quiet=True)
        with unittest.mock.patch.object(cli, "_github", lambda rep: None), \
             unittest.mock.patch("csvscrape.probe.run", fake_run), \
             unittest.mock.patch("csvscrape.probe.pool_size", lambda: 10_000), \
             unittest.mock.patch("csvscrape.probe.outstanding_size",
                                 lambda: 9_000), \
             unittest.mock.patch.object(config, "PROBE_DRAW", 100):
            cli._cmd_probe(args)
        return seen

    def test_a_short_draw_does_not_end_the_run(self):
        """A clustered draw routinely comes back short on a pool with plenty left.
        """
        self.assertEqual(self._passes([90, 80, 0]), [90, 80, 0],
                         "stopped on a draw that was merely short")

    def test_an_empty_draw_ends_the_run(self):
        self.assertEqual(self._passes([100, 0, 100]), [100, 0],
                         "kept drawing after the pool came back empty")

    def test_a_pass_that_settles_nothing_ends_the_run(self):
        """A transient failure does not retire its key, by design -- so files
        that keep failing are redrawn every pass and the queue never empties.
        Without this the loop spins forever on them."""
        seen = self._passes([100, 5, 5, 5, 5], retired_per_pass=[100, 0, 0, 0, 0])
        self.assertEqual(seen, [100, 5], "spun on files that never settle")

    def test_a_full_draw_keeps_going(self):
        self.assertEqual(len(self._passes([100, 100, 100, 0])), 4)


class ProgressIsReportedAgainstTheWholePool(unittest.TestCase):
    """Progress is a position in the whole pool: done plus outstanding is the
    pool."""

    REPOS, PER_REPO = 40, 25

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe, self.log = probe, log

        (self.tmp / "index").mkdir(parents=True)
        rows = [{
            "repo_id": r, "owner": "o", "full_name": f"o/r{r}",
            "path": f"f{i}.csv", "blob_sha": f"s{r}_{i}",
            "content_key": f"k{r}_{i}", "size": 4096, "raw_size": 4096,
            "bucket": 12, "is_lfs": False, "lfs_suspect": False, "tier": "hi",
            "truncated": False, "ref": f"{r:040x}",
        } for r in range(self.REPOS) for i in range(self.PER_REPO)]
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _mark_probed(self, rows):
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as writer:
            for row in rows:
                writer.write({"content_key": row["content_key"],
                              "fetched": True, "http": 200})


    def test_an_untouched_pool_has_everything_outstanding(self):
        self.assertEqual(self.probe.pool_size(), self.REPOS * self.PER_REPO)
        self.assertEqual(self.probe.outstanding_size(),
                         self.REPOS * self.PER_REPO)


    def test_done_plus_outstanding_is_the_pool(self):
        """The invariant the progress line rests on. It holds because the
        position counts `retired` -- files taken out of the pool -- rather than
        `probed`, which also counts a transient failure that comes round
        again."""
        pool = self.probe.pool_size()
        settled = 0
        for drawn in (100, 250, 400):
            self._mark_probed(self.probe.candidates(limit=drawn, order="repo"))
            settled += drawn
            with self.subTest(settled=settled):
                left = self.probe.outstanding_size()
                self.assertEqual(pool - left, settled, "the position drifted")
                self.assertEqual(settled + left, pool)

    def test_a_404_counts_as_done_because_it_will_never_be_redrawn(self):
        """`retired` is fetched-or-404, and the pool figure has to agree with
        it or the last few files never appear to finish."""
        rows = self.probe.candidates(limit=10, order="repo")
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as writer:
            for row in rows:
                writer.write({"content_key": row["content_key"],
                              "fetched": False, "http": 404})

        self.assertEqual(self.probe.outstanding_size(),
                         self.REPOS * self.PER_REPO - 10)

    def test_a_killed_runs_records_are_not_probed_twice(self):
        """A killed run leaves its shard `.active`, which every reader skips, so
        the pass must seal orphans before drawing or it re-fetches what
        was already probed."""
        drawn = self.probe.candidates(limit=100, order="repo")
        writer = self.log.Writer(self.probe.HEADS_LOG, rotate=10_000)
        for row in drawn:
            writer.write({"content_key": row["content_key"],
                          "fetched": True, "http": 200})
        writer.flush()
        del writer                      # killed: the shard stays `.active`

        self.assertEqual(self.probe.outstanding_size(),
                         self.REPOS * self.PER_REPO - 100,
                         "an unsealed shard was counted as unprobed")
        again = {r["content_key"] for r in
                 self.probe.candidates(limit=100, order="repo")}
        self.assertFalse(again & {r["content_key"] for r in drawn},
                         "the draw re-probed a killed run's files")

    def test_a_transient_failure_stays_outstanding(self):
        rows = self.probe.candidates(limit=10, order="repo")
        with self.log.Writer(self.probe.HEADS_LOG, rotate=10_000) as writer:
            for row in rows:
                writer.write({"content_key": row["content_key"],
                              "fetched": False, "http": 503})

        self.assertEqual(self.probe.outstanding_size(),
                         self.REPOS * self.PER_REPO,
                         "a 503 was counted as done and will never be redrawn")


class TheRateIsMeasuredFromWhereTheSessionStarted(unittest.TestCase):
    """The rate counts only this session's progress: a resumed run's counter
    starts at its position in the pool, not at zero."""

    def _reporter(self):
        import io

        from csvscrape.report import Reporter

        return Reporter(stream=io.StringIO())

    def test_a_resumed_run_does_not_count_the_backlog_as_its_own_work(self):
        rep = self._reporter()
        rep._on_repo(done=40_000_000, of=65_000_000)   # first reading
        rep._t0 -= 10.0                                 # ten seconds pass
        rep._on_repo(done=40_000_100, of=65_000_000)

        self.assertAlmostEqual(rep._rate(40_000_100), 10.0, delta=1.0,
                               msg="the backlog was counted as this session's")


    def test_the_shown_rate_notices_a_collapse_the_eta_average_cannot(self):
        """Why there are two estimators: a long run's cumulative average barely
        moves when throughput collapses, while the ETA wants exactly
        that long-run average."""
        import collections

        from csvscrape.report import _RATE_WINDOW

        rep = self._reporter()
        rep._on_repo(done=0, of=10_000_000)
        # Two days at 1,000/s, then the window's worth of near-nothing.
        day = 2 * 86_400
        rep._t0 -= day
        rep._samples = collections.deque([(rep._t0, 0)])
        rep._on_repo(done=day * 1000, of=10_000_000)
        rep._samples = collections.deque(
            [(time.monotonic() - _RATE_WINDOW, day * 1000)])
        rep._on_repo(done=day * 1000 + 10, of=10_000_000)

        self.assertAlmostEqual(rep._rate(day * 1000 + 10), 1000.0, delta=1.0,
                               msg="the ETA's average moved on one bad window")
        self.assertLess(rep._recent_rate(), 1.0,
                        "the shown rate did not notice the collapse")


class AStalledPassReadsAsStalledRatherThanFrozen(unittest.TestCase):
    """Only progress advances the rate samples, so the shown rate must decay
    during a stall rather than freeze."""

    def _reporter(self):
        import io

        from csvscrape.report import Reporter

        return Reporter(stream=io.StringIO())

    def test_the_shown_rate_decays_while_nothing_is_written(self):
        import collections

        rep = self._reporter()
        rep._on_repo(done=0, of=10_000_000)
        # 1,000 files over ten seconds, then a stall.
        rep._samples = collections.deque(
            [(time.monotonic() - 10.0, 0), (time.monotonic(), 10_000)])
        before = rep._recent_rate()
        self.assertAlmostEqual(before, 1000.0, delta=50.0)

        # Sixty seconds pass with no further readings at all.
        rep._samples = collections.deque(
            [(time.monotonic() - 70.0, 0), (time.monotonic() - 60.0, 10_000)])
        after = rep._recent_rate()

        self.assertLess(after, before / 5,
                        "the rate froze at its pre-stall value")
        self.assertAlmostEqual(after, 10_000 / 70.0, delta=20.0)




class TheDisplayCarriesThroughputAndAnEta(unittest.TestCase):
    """A run over the whole pool needs to say how fast it is going and how much
    longer it has, on both lines -- the heartbeat because it is the only one
    that survives being piped to a file, the progress line being a terminal
    repaint that is dropped without a tty.

    `eta` is the whole of the second half. A `full` beside it -- the same
    arithmetic over the pool rather than over what is left -- said nearly the
    same thing on any run that starts near the beginning, which is every run
    now that `probe` sweeps to exhaustion on its own."""

    POOL = 65_578_908

    def _drawn(self, done_then, done_now, seconds, heartbeat=False):
        import io
        import re

        from csvscrape.report import Reporter

        class Tty(io.StringIO):
            def isatty(self):
                return True

        import collections

        screen = Tty()
        rep = Reporter(stream=screen)
        rep._on_repo(done=done_then, of=self.POOL)
        # Wind the clock back on both estimators: the cumulative one keeps its
        # own baseline, the windowed one its samples, and shifting only the
        # first would leave them describing different intervals.
        rep._t0 -= seconds
        rep._samples = collections.deque((t - seconds, d)
                                         for t, d in rep._samples)
        rep._on_repo(done=done_now, of=self.POOL)
        if heartbeat:
            rep._beat -= 400
            rep._on_rate(pool="graphql", permitted=1.3, achieved=1.3, requests=99)
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", screen.getvalue())
        return [l.strip() for l in text.replace("\r", "\n").split("\n") if l.strip()]

    def test_the_progress_line_carries_the_rate_and_what_is_left(self):
        # 900,000 files in half an hour = 1.8M/h, and 63.2M left is 35.1h.
        line = self._drawn(1_460_000, 2_360_000, 1800.0)[-1]

        self.assertIn("1,800,000 files/h", line)
        self.assertIn("eta 35.1h", line)
        self.assertNotIn("full", line, "the redundant whole-pool figure is back")

    def test_the_heartbeat_carries_the_throughput(self):
        beat = [l for l in self._drawn(1_460_000, 2_360_000, 1800.0,
                                       heartbeat=True) if l.startswith("⟳")]

        self.assertEqual(len(beat), 1)
        self.assertIn("1,800,000 files/h", beat[0])
        self.assertIn("eta 35.1h", beat[0])
        self.assertNotIn("full", beat[0])


    def test_an_unmeasured_rate_says_so_rather_than_reporting_no_time_left(self):
        first = self._drawn(1_460_000, 2_360_000, 1800.0)[0]

        self.assertIn("eta ?", first)
        self.assertNotIn("eta 0s", first)


class SigtermWindsDownLikeCtrlC(unittest.TestCase):
    """`timeout`, `kill`, systemd and a closed terminal all send SIGTERM, and
    its default action terminates without unwinding -- so a pass's `finally`
    never runs, the shard is never sealed, and the records since the last zstd
    frame boundary go with it. `cli.main` turns it into a KeyboardInterrupt
    for every command."""

    def _run_recording_signal_calls(self):
        import signal

        from csvscrape import cli

        calls = []
        real = signal.signal

        def recording(signum, handler):
            calls.append((signum, handler))
            return real(signum, handler)

        with unittest.mock.patch.object(cli, "_cmd_selectors", lambda args: 0), \
             unittest.mock.patch("signal.signal", recording):
            cli.main(["selectors"])
        return calls, signal

    def test_the_handler_is_installed_and_then_restored(self):
        calls, signal = self._run_recording_signal_calls()
        term = [handler for signum, handler in calls if signum == signal.SIGTERM]

        self.assertEqual(len(term), 2, "not installed-then-restored exactly once")
        self.assertNotIn(term[0], (signal.SIG_DFL, signal.SIG_IGN),
                         "SIGTERM was left on its default, which does not unwind")
        self.assertEqual(term[1], signal.getsignal(signal.SIGTERM),
                         "the caller's handler was not put back")

    def test_the_handler_raises_so_the_existing_wind_down_catches_it(self):
        calls, signal = self._run_recording_signal_calls()
        handler = [h for s, h in calls if s == signal.SIGTERM][0]

        with self.assertRaises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)


class GaugesDoNotAddUpAcrossPasses(unittest.TestCase):
    """A multi-pass run sums counters but not measurements (peaks, rates)."""

    @staticmethod
    def _fold(passes):
        from csvscrape import cli

        total: dict = {}
        for stats in passes:
            for key, value in stats.items():
                total[key] = cli._fold(key, total, value)
        return total

    def test_counts_add_and_measurements_do_not(self):
        total = self._fold([
            {"probed": 200000, "peak_rss_mb": 6359, "raw_rate_min": 10.0,
             "raw_rate_max": 28.0, "raw_rate_final": 28.0},
            {"probed": 34, "peak_rss_mb": 7462, "raw_rate_min": 10.0,
             "raw_rate_max": 56.0, "raw_rate_final": 56.0},
        ])
        self.assertEqual(total["probed"], 200034, "a count stopped adding up")
        self.assertEqual(total["peak_rss_mb"], 7462, "peak RSS was summed")
        self.assertEqual(total["raw_rate_min"], 10.0, "a floor was summed")
        self.assertEqual(total["raw_rate_max"], 56.0, "a ceiling was summed")
        self.assertEqual(total["raw_rate_final"], 56.0,
                         "the settled rate is the last pass's, not a total")

    def test_a_gauge_no_pass_set_stays_none(self):
        """`raw_rate_wall` is None until a 429 is seen, and `max` of None and a
        number raises rather than reporting anything."""
        total = self._fold([{"raw_rate_wall": None}, {"raw_rate_wall": None}])
        self.assertIsNone(total["raw_rate_wall"])


class RepoOrderClustersTheDraw(unittest.TestCase):
    """`--order repo` trades a uniform sample for throughput: GraphQL is
    charged per repository resolution, so a draw that revisits few repos packs
    far more files into one request. It is a cluster sample and not a sample of
    the pool, which is why `uniform` stays the default."""

    REPOS, PER_REPO = 40, 25

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe

        for mod in (config, log, index, probe):
            importlib.reload(mod)
        self.probe = probe

        (self.tmp / "index").mkdir(parents=True)
        rows = [{
            "repo_id": r, "owner": "o", "full_name": f"o/r{r}",
            "path": f"f{i}.csv", "blob_sha": f"s{r}_{i}",
            "content_key": f"k{r}_{i}", "size": 4096, "raw_size": 4096,
            "bucket": 12, "is_lfs": False, "lfs_suspect": False, "tier": "hi",
            "truncated": False, "ref": f"{r:040x}",
        } for r in range(self.REPOS) for i in range(self.PER_REPO)]
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    @staticmethod
    def _repos(rows):
        return {r["full_name"] for r in rows}

    def test_it_touches_far_fewer_repos_than_uniform(self):
        clustered = self.probe.candidates(limit=100, order="repo")
        uniform = self.probe.candidates(limit=100, order="uniform")

        self.assertEqual(len(clustered), 100)
        self.assertLess(len(self._repos(clustered)), len(self._repos(uniform)) / 4,
                        "the clustered draw is as scattered as the uniform one")



    def test_the_same_seed_reproduces_the_draw(self):
        first = self.probe.candidates(limit=100, order="repo", seed=7)
        again = self.probe.candidates(limit=100, order="repo", seed=7)
        self.assertEqual([r["content_key"] for r in first],
                         [r["content_key"] for r in again])




    # -- the resume cursor ------------------------------------------------

    def _cursor(self, rows):
        return (rows[-1]["_repo_rank"], rows[-1]["_key_rank"])


    def test_a_resumed_draw_takes_what_the_first_one_did_not(self):
        """The point of the cursor: successive passes partition the pool
        without anyone asking the log what has been probed."""
        first = self.probe.candidates(limit=100, order="repo")
        second = self.probe.candidates(limit=100, order="repo",
                                       after=self._cursor(first))

        a = {r["content_key"] for r in first}
        b = {r["content_key"] for r in second}
        self.assertEqual(len(a & b), 0, "the second pass redrew the first's rows")
        self.assertEqual(len(second), 100)
        self.assertGreater((second[0]["_repo_rank"], second[0]["_key_rank"]),
                           self._cursor(first))

    def test_cursored_passes_cover_the_pool_exactly_once(self):
        """Walked to exhaustion, the passes are a partition -- nothing drawn
        twice and nothing skipped. A cursor comparing on the repo rank alone
        would drop the tail of whichever repo straddled a pass boundary."""
        seen, cursor, rows = [], None, None
        while True:
            rows = self.probe.candidates(limit=100, order="repo", after=cursor)
            if not rows:
                break
            seen += [r["content_key"] for r in rows]
            cursor = self._cursor(rows)

        self.assertEqual(len(seen), len(set(seen)), "a row was drawn twice")
        self.assertEqual(len(seen), self.REPOS * self.PER_REPO,
                         "the cursored walk skipped rows")

    def test_a_partly_probed_pool_still_draws(self):
        """The draw has to find the k-th smallest rank among the rows that are
        *left*, not at a fixed point in the rank range.

        `--order repo` consumes that range front-first, so once a run has been
        going a while every row below any absolute cut is already probed and
        the anti-join removes all of them. A draw written that way returns zero
        rows against a pool with millions left -- and `cli` reads an empty draw
        as an exhausted pool and ends the run reporting success."""
        from csvscrape import log

        # Half the pool probed, taken off the front of the rank order the way
        # a real run consumes it. The draw that follows is small enough that a
        # fixed-fraction cut would fall entirely inside that probed half, which
        # is the whole point: the bug only shows when `limit` is small against
        # what is left, and that is every pass of a long run.
        done = self.probe.candidates(limit=500, order="repo")
        with log.Writer(self.probe.HEADS_LOG, rotate=10_000) as writer:
            for row in done:
                writer.write({"content_key": row["content_key"],
                              "fetched": True, "http": 200})

        rows = self.probe.candidates(limit=10, order="repo")
        self.assertEqual(len(rows), 10,
                         "the draw came back empty with the pool far from done")
        self.assertFalse({r["content_key"] for r in rows}
                         & {r["content_key"] for r in done},
                         "redrew rows the log says are probed")


class TheProbeQueueHoldsWholeRepos(unittest.TestCase):
    """`candidates` can cluster the draw all it likes; if the queue hands out
    single rows then `workers` threads deal one repo's files across all of
    them, and every batch reaches `blobs` spanning as many repos as it has
    rows -- which is the shape that could not be served."""

    @staticmethod
    def rows(spec):
        return [{"full_name": name, "content_key": f"{name}-{i}",
                 "size": size, "is_lfs": False}
                for name, n, size in spec for i in range(n)]

    def test_rows_are_grouped_by_repo(self):
        from csvscrape import probe

        chunks = probe._repo_chunks(self.rows([("o/a", 3, 100),
                                               ("o/b", 2, 100)]))
        self.assertEqual([len(c) for c in chunks], [3, 2])
        self.assertEqual([{r["full_name"] for r in c} for c in chunks],
                         [{"o/a"}, {"o/b"}])


    def test_a_repo_too_large_for_one_batch_is_split(self):
        """Split here rather than in the worker, so every chunk is already
        inside the limits when it reaches the queue -- which is what lets the
        worker count chunks as repo-equivalents."""
        from csvscrape import config, probe

        big = config.PROBE_BATCH + 50
        chunks = probe._repo_chunks(self.rows([("o/a", big, 10)]))
        self.assertGreater(len(chunks), 1, "an oversized repo was not split")
        self.assertTrue(all(len(c) <= config.PROBE_BATCH for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), big, "rows were lost")

    def test_the_byte_budget_splits_a_repo_too(self):
        from csvscrape import config, probe

        # Sized off what the row will cost the *response*, not off the file.
        # The two were the same only while the routing threshold sat below the
        # truncation cap; now that everything GraphQL will serve is routed to
        # it, a 10 MB file still costs one GRAPHQL_TEXT_CAP head, and rows
        # picked by file size made a batch that never reached the budget.
        size = config.GRAPHQL_TEXT_CAP
        cost = probe._graphql_cost({"is_lfs": False, "size": size})
        per = max(1, config.GRAPHQL_BATCH_BYTES // cost)
        chunks = probe._repo_chunks(self.rows([("o/a", per * 3, size)]))
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(sum(probe._graphql_cost(r) for r in chunk[:-1]),
                                 config.GRAPHQL_BATCH_BYTES)

    def test_no_row_is_lost_or_duplicated(self):
        from csvscrape import probe

        rows = self.rows([("o/a", 7, 100), ("o/b", 1, 100), ("o/c", 4, 100)])
        keys = [r["content_key"] for c in probe._repo_chunks(rows) for r in c]
        self.assertEqual(sorted(keys), sorted(r["content_key"] for r in rows))

    def _drain(self, rows):
        """Every batch a worker would assemble from these rows, in order."""
        import queue as _q

        from csvscrape import probe

        work: _q.Queue = _q.Queue()
        for chunk in probe._repo_chunks(rows):
            work.put(chunk)
        out, carry = [], None
        while True:
            batch, carry = probe._next_batch(work, carry)
            if not batch:
                return out
            out.append(batch)

    def test_a_batch_never_exceeds_the_byte_budget(self):
        """Limits are checked before a chunk goes in, so a batch near the budget
        cannot admit a whole further chunk."""
        from csvscrape import config, probe

        size = config.PROBE_GRAPHQL_MAX_BYTES
        per = max(1, config.GRAPHQL_BATCH_BYTES // size)
        rows = self.rows([(f"o/r{i}", per, size) for i in range(12)])
        for batch in self._drain(rows):
            cost = sum(probe._graphql_cost(r) for r in batch)
            self.assertLessEqual(cost, config.GRAPHQL_BATCH_BYTES,
                                 f"batch of {len(batch)} rows overshot the budget")

    def test_a_batch_never_exceeds_the_repo_cap(self):
        from csvscrape import config

        rows = self.rows([(f"o/r{i}", 2, 10)
                          for i in range(config.GRAPHQL_BATCH_REPOS * 3)])
        for batch in self._drain(rows):
            self.assertLessEqual(len({r["full_name"] for r in batch}),
                                 config.GRAPHQL_BATCH_REPOS)

    def test_the_chunk_that_does_not_fit_is_kept_not_dropped(self):
        from csvscrape import config

        size = config.PROBE_GRAPHQL_MAX_BYTES
        per = max(1, config.GRAPHQL_BATCH_BYTES // size)
        rows = self.rows([(f"o/r{i}", per, size) for i in range(12)])
        got = [r["content_key"] for b in self._drain(rows) for r in b]
        self.assertEqual(sorted(got), sorted(r["content_key"] for r in rows))

    def test_an_oversized_chunk_still_makes_progress(self):
        """A chunk is always accepted into an empty batch, or a worker whose
        first chunk is larger than the budget never drains its queue."""
        from csvscrape import config

        rows = self.rows([("o/a", 4, config.GRAPHQL_BATCH_BYTES)])
        batches = self._drain(rows)
        self.assertTrue(batches, "the worker stalled on an oversized chunk")
        self.assertEqual(sum(len(b) for b in batches), 4)


class TheScanRecordsTheCommitItSaw(unittest.TestCase):
    """`trees` records the commit `git/trees/HEAD` resolved to, so later reads are
    pinned to the scanned snapshot."""

    def _scan(self, tree):
        from csvscrape import trees

        class GH:
            def tree(self, full_name, ref="HEAD"):
                return tree
            def raw_file(self, *a):
                return None
            def blobs(self, rows):
                return [None] * len(rows)

        record, _ = trees.scan_one(GH(), {"repo_id": 1, "full_name": "o/r",
                                          "tier": "hi"})
        return record

    def test_the_commit_sha_is_recorded_not_the_literal_head(self):
        sha = "b272558c486188ea8516b4aa65b65baeee1b0248"
        rec = self._scan({"sha": sha, "truncated": False, "tree": [
            {"type": "blob", "path": "a.csv", "sha": "0" * 40, "size": 10}]})
        self.assertEqual(rec["ref"], sha)

    def test_a_response_without_one_is_a_failed_scan_not_an_unpinned_one(self):
        """A tree response without a commit sha is a failed scan: an unpinned row
        would read whatever the branch points at later."""
        rec = self._scan({"truncated": False, "tree": [
            {"type": "blob", "path": "a.csv", "sha": "0" * 40, "size": 10}]})
        self.assertEqual(rec["state"], "error")
        self.assertNotIn("ref", rec, "wrote a row with no usable pin")



class TheScanKeepsTheEntryMode(unittest.TestCase):
    """The scan records each blob's mode: a symlink is `type: blob` too, and only
    mode 120000 tells it apart."""

    def _scan(self, entries):
        from csvscrape import trees

        class GH:
            def tree(self, full_name, ref="HEAD"):
                return {"sha": "a" * 40, "truncated": False, "tree": entries}
            def raw_file(self, *a):
                return None
            def blobs(self, rows):
                return [None] * len(rows)

        rec, _ = trees.scan_one(GH(), {"repo_id": 1, "full_name": "o/r",
                                       "tier": "hi"})
        return rec

    def test_modes_are_recorded_and_aligned_with_paths(self):
        rec = self._scan([
            {"type": "blob", "path": "real.csv", "sha": "1" * 40,
             "size": 4096, "mode": "100644"},
            {"type": "blob", "path": "link.csv", "sha": "2" * 40,
             "size": 20, "mode": "120000"},
            {"type": "tree", "path": "d", "sha": "3" * 40, "mode": "040000"},
        ])
        self.assertEqual(len(rec["modes"]), len(rec["paths"]))
        self.assertEqual(dict(zip(rec["paths"], rec["modes"])),
                         {"real.csv": "100644", "link.csv": "120000"})

    def test_a_symlink_is_still_indexed_not_dropped(self):
        """Recorded, not filtered -- same rule the rest of the scan follows.
        A selector can change its mind about symlinks; a rescan cannot."""
        rec = self._scan([
            {"type": "blob", "path": "link.csv", "sha": "2" * 40,
             "size": 20, "mode": "120000"}])
        self.assertEqual(rec["paths"], ["link.csv"])


class TheRefReachesTheRangeRequest(unittest.TestCase):
    """The recorded pin reaches the range request URL."""

    def test_head_bytes_puts_the_ref_in_the_url(self):
        from csvscrape.client import GitHub

        seen = {}

        class Resp:
            status_code = 206
            content = b"x" * 64
            headers: dict = {}

        gh = GitHub.__new__(GitHub)
        gh.raw = type("P", (), {
            "slot": staticmethod(lambda: contextlib.nullcontext()),
            "reward": staticmethod(lambda: None)})()
        gh.anon = type("S", (), {"get": staticmethod(
            lambda url, **kw: seen.update(url=url) or Resp())})()

        sha = "b272558c486188ea8516b4aa65b65baeee1b0248"
        gh.head_bytes("o/r", "d/a.csv", 64, ref=sha)
        self.assertIn(f"/o/r/{sha}/", seen["url"])
        self.assertNotIn("/HEAD/", seen["url"])



class TreesReadsPointersInBatches(unittest.TestCase):
    """LFS pointer bodies are read in GraphQL batches rather than one raw request
    each. Here the pointer *is* the content being read, unlike in `probe`."""

    def test_pointer_bodies_come_from_one_batched_read(self):
        from csvscrape import trees

        asked, raw_calls = [], []

        class GH:
            def blobs(self, rows):
                asked.append(len(rows))
                return [b"version https://git-lfs.github.com/spec/v1\n"] * len(rows)
            def raw_file(self, *a):
                raw_calls.append(a)
                return b""

        items = [(f"f{i}.csv", f"{i:040x}") for i in range(150)]
        out = trees._blob_bodies(GH(), "o/r", items)
        self.assertEqual(len(out), 150)
        self.assertEqual(asked, [150], "not batched into one request")
        self.assertEqual(raw_calls, [], "fell back to raw needlessly")



    def test_parallel_batches_still_come_back_in_order(self):
        from csvscrape import config, trees

        class GH:
            def blobs(self, rows):
                return [r["blob_sha"].encode() for r in rows]

        n = config.LFS_BATCH * 3
        items = [(f"f{i}", f"{i:040x}") for i in range(n)]
        got = trees._blob_bodies(GH(), "o/r", items)
        self.assertEqual(got, [sha.encode() for _, sha in items],
                         "parallel batches were reassembled out of order")

    def test_what_graphql_declines_still_gets_read(self):
        from csvscrape import trees

        class GH:
            def blobs(self, rows):
                return [None if i == 1 else b"ok" for i in range(len(rows))]
            def raw_file(self, full_name, ref, path):
                return b"from-raw:" + path.encode()

        items = [("a.csv", "0" * 40), ("b.csv", "1" * 40), ("c.csv", "2" * 40)]
        self.assertEqual(trees._blob_bodies(GH(), "o/r", items),
                         [b"ok", b"from-raw:b.csv", b"ok"])

    def test_a_failed_batch_falls_back_rather_than_losing_the_repo(self):
        from csvscrape import trees

        class GH:
            def blobs(self, rows):
                raise RuntimeError("graphql: http 502")
            def raw_file(self, full_name, ref, path):
                return b"from-raw"

        items = [("a.csv", "0" * 40), ("b.csv", "1" * 40)]
        self.assertEqual(trees._blob_bodies(GH(), "o/r", items),
                         [b"from-raw", b"from-raw"])


class LfsNeverGoesThroughGraphQL(unittest.TestCase):
    """The git blob for an LFS file is the 133-byte pointer. It is valid UTF-8
    of exactly its declared size, so it passes every check `blobs` makes while
    being entirely the wrong content -- a silent corruption, not an error."""

    def test_probe_routes_lfs_to_the_media_path(self):
        from csvscrape import probe

        self.assertFalse(probe._via_graphql(
            {"is_lfs": True, "size": 133}), "an LFS row was offered to GraphQL")
        self.assertTrue(probe._via_graphql({"is_lfs": False, "size": 133}),
                        "a plain row of the same size was held back")


    def test_oversized_blobs_are_not_offered_to_graphql(self):
        """Above GRAPHQL_BLOB_MAX GraphQL returns an empty `text`, so such files
        go straight to the range path."""
        from csvscrape import config, probe

        self.assertFalse(probe._via_graphql(
            {"is_lfs": False, "size": config.GRAPHQL_BLOB_MAX + 1}),
            "a blob GraphQL will not serve was still batched")
        self.assertLess(config.GRAPHQL_BLOB_MAX, 11_416_275,
                        "above the size measured to come back empty")
        self.assertGreater(config.GRAPHQL_BLOB_MAX, 5_000_000,
                           "needlessly excluding blobs that do work")


    def test_a_prefetched_body_skips_the_range_request(self):
        from csvscrape import probe

        called = []

        class FakeGH:
            def head_bytes(self, *a, **k):
                called.append(a)
                return b"should not be used", 206

        row = {"content_key": "k", "bucket": 3, "size": 12, "is_lfs": False,
               "ref": "c" * 40,
               "full_name": "o/r", "path": "a.csv"}
        rec = probe.probe_one(FakeGH(), row, None, 512_000,
                              which=("custom",), body=b"a,b\n1,2\n")
        self.assertEqual(called, [], "fetched over the network anyway")
        self.assertEqual(rec["via"], "graphql")
        self.assertTrue(rec["fetched"])

    def test_the_fallback_records_which_path_served_it(self):
        from csvscrape import probe

        class FakeGH:
            def head_bytes(self, *a, **k):
                return b"a,b\n1,2\n", 206

        row = {"content_key": "k", "bucket": 3, "size": 12, "is_lfs": True,
               "ref": "c" * 40,
               "full_name": "o/r", "path": "a.csv"}
        rec = probe.probe_one(FakeGH(), row, None, 512_000, which=("custom",))
        self.assertEqual(rec["via"], "raw")


class RawPoolFindsTheLimit(unittest.TestCase):
    """raw/media state no budget in headers, so the rate has to be discovered:
    additive increase while requests succeed, a cut on a 429."""

    def pool(self, **kw):
        from csvscrape.client import RawPool
        return RawPool(**kw)

    def test_a_clean_second_raises_the_rate(self):
        from csvscrape import config

        pool = self.pool(rate=4.0)
        for _ in range(3):
            pool.reward()
        self.assertEqual(pool.rate, 4.0, "raised before a second's worth landed")
        pool.reward()
        self.assertEqual(pool.rate, 4.0 + config.RAW_RATE_STEP)

    def test_recovery_below_a_known_wall_is_not_paced_as_a_probe(self):
        """Climbing back to a rate that was just being served is not a probe and
        is not slowed down."""

        pool = self.pool(rate=10.0)
        pool._wall = 30.0                    # 10 is far below it: recover fast
        for _ in range(10):
            pool.reward()
        self.assertGreater(pool.rate, 10.0, "throttled its own recovery")

    def test_probing_above_a_known_wall_stays_cautious(self):

        pool = self.pool(rate=30.0)
        pool._wall = 30.0                    # at the wall: every step is a probe
        for _ in range(30):
            pool.reward()
        self.assertEqual(pool.rate, 30.0, "charged the wall without patience")

    def test_a_429_reduces_the_rate(self):
        from csvscrape import config

        pool = self.pool(rate=8.0)
        pool.penalise(None)
        self.assertEqual(pool.rate, 8.0 * config.RAW_RATE_BACKOFF)

    def test_a_burst_of_429s_is_one_reduction(self):
        """When one budget is blown every request in flight gets a 429; that is
        one congestion event and one reduction."""
        from csvscrape import config

        # Deliberately inside RAW_MAX_RATE: started above it, `_set_rate`
        # clamps and the assertion would read the clamp instead of the backoff.
        start = config.RAW_MAX_RATE / 2
        pool = self.pool(rate=start)
        for _ in range(14):
            pool.penalise(None)
        self.assertEqual(pool.rate, start * config.RAW_RATE_BACKOFF,
                         "compounded one event into many cuts")
        self.assertEqual(pool.stats["raw_throttled"], 14, "lost the waste count")

    def test_a_later_event_reduces_again(self):
        """The cooldown must not swallow a genuinely new event."""
        import time as _t

        from csvscrape import config

        start = config.RAW_MAX_RATE / 2
        pool = self.pool(rate=start)
        pool.penalise(None)
        pool._event_until = _t.monotonic()      # the event has passed
        pool.penalise(None)
        self.assertEqual(pool.rate, start * config.RAW_RATE_BACKOFF ** 2)

    def test_the_cut_is_from_the_achieved_rate_not_the_permitted_one(self):
        """The two differ when something other than the pacer limits throughput,
        and then cutting the permitted rate changes nothing on the
        wire."""
        pool = self.pool(rate=120.0)
        pool._achieved = 71.0
        pool.penalise(None)
        from csvscrape import config
        self.assertEqual(pool.rate, 71.0 * config.RAW_RATE_BACKOFF)
        self.assertEqual(pool.stats["raw_rate_wall"], 71.0,
                         "the wall must record what was sustained, not what was allowed")

    def test_achieved_rate_is_measured_from_real_traffic(self):
        from csvscrape import config

        pool = self.pool(rate=500.0)
        original = config.RAW_RATE_WINDOW
        self.addCleanup(setattr, config, "RAW_RATE_WINDOW", original)
        config.RAW_RATE_WINDOW = 0.05
        for _ in range(200):          # 200 / 500 per s = 0.4s, several windows
            with pool.slot():
                pass
        self.assertNotEqual(pool._achieved, 500.0, "never closed a window")
        # The pacer caps starts at 500/s, so the measured figure must sit at or
        # under it rather than reporting the loop's own speed.
        self.assertLessEqual(pool._achieved, 500.0 * 1.1)
        self.assertGreater(pool._achieved, 100.0)


    def test_backoff_stops_at_the_floor(self):
        from csvscrape import config

        import time as _t

        pool = self.pool(rate=1.0)
        for _ in range(20):
            # Each is a *separate* congestion event; one event is one cut, so
            # reaching the floor takes twenty of them rather than twenty 429s.
            pool._event_until = _t.monotonic()
            pool.penalise(None)
        self.assertEqual(pool.rate, config.RAW_MIN_RATE,
                         "backed the run off to a standstill")

    def test_increase_stops_at_the_ceiling(self):
        from csvscrape import config

        pool = self.pool(rate=config.RAW_MAX_RATE)
        for _ in range(200):
            pool.reward()
        self.assertEqual(pool.rate, config.RAW_MAX_RATE)

    def test_one_429_parks_every_worker(self):
        """One 429 parks every worker through the shared pacer, not just the one
        that got it."""
        pool = self.pool(rate=100.0)
        pool.penalise(retry_after="30")
        self.assertGreater(pool.pacer._next - time.monotonic(), 25.0)





    def test_a_throttled_retry_is_never_immediate(self):
        """A rejected request is charged against the budget, so a retry inside a
        throttle waits at least RAW_MIN_HOLD."""
        from csvscrape import config

        pool = self.pool(rate=30.0)
        before = time.monotonic()
        pool.penalise(None)
        self.assertGreaterEqual(pool.pacer._next - before,
                                config.RAW_MIN_HOLD - 0.05)




class WaitsDoNotEatTheProgressLine(unittest.TestCase):
    """Short waits are drawn over the status line rather than taking a durable
    line; long ones get both."""

    def setUp(self):
        import io

        from csvscrape.report import Reporter

        self.stream = type("Tty", (io.StringIO,), {"isatty": lambda self: True})()
        self.reporter = Reporter(self.stream)
        self.reporter("shard_start", index=7, tier="low", ext="csv",
                      query="extension:csv size:1024..1279")

    def durable(self) -> list[str]:
        return self.stream.getvalue().split("\n")[:-1]

    def test_a_short_wait_leaves_no_durable_line(self):
        for _ in range(12):
            self.reporter("wait", seconds=1.0, reason="retry-after", throttled=True)
        self.assertEqual(self.durable(), [],
                         "short waits are still accumulating in the scrollback")

    def test_the_status_line_survives_a_wait(self):
        self.reporter("wait", seconds=2.0, reason="retry-after", throttled=True)
        painted = self.stream.getvalue().split("\r\033[K")[-1]
        self.assertIn("1024..1279", painted, "the status line was not redrawn")
        self.assertIn("⏸", painted)



    def test_a_long_wait_still_gets_a_durable_line(self):
        """Long enough to look hung, so it also gets a durable line -- the only
        kind a piped log keeps."""
        self.reporter("wait", seconds=120.0, reason="secondary rate limit",
                      throttled=True)
        self.assertEqual(len(self.durable()), 1)
        self.assertIn("2.0m", self.durable()[0])








    def test_a_straggler_cannot_print_after_the_summary(self):
        """A pass leaves daemon workers running past the summary, and a fetch
        already on the wire still reports when it lands. That printed
        `raw throttle 1s` under "interrupted -- scanned repos are recorded"."""
        self.reporter("repo", done=1, of=2, csvs=1, lfs=0, name="a.csv")
        self.reporter.close()
        after = len(self.stream.getvalue())
        self.reporter("wait", seconds=1.0, reason="raw throttle", throttled=True)
        self.reporter("repo", done=2, of=2, csvs=2, lfs=0, name="b.csv")
        self.assertEqual(len(self.stream.getvalue()), after,
                         "a straggler wrote past close()")


    def test_a_piped_log_keeps_the_durable_lines_only(self):
        import io

        from csvscrape.report import Reporter

        plain = io.StringIO()
        rep = Reporter(plain)
        rep("shard_start", index=1, tier="low", ext="csv", query="extension:csv size:1..2")
        rep("wait", seconds=2.0, reason="retry-after", throttled=True)
        rep("shard_end", state="done", reason="exhausted", pages=1, total=5)
        rep.close()
        self.assertNotIn("\033[K", plain.getvalue())
        self.assertIn("exhausted", plain.getvalue())
        self.assertIn("backoffs", plain.getvalue())


class RateLimitHoldFloor(unittest.TestCase):
    """A reset already in the past must still hold for the floor, or the retry
    fires straight back into the limit."""

    def setUp(self):
        from csvscrape.client import Bucket

        self.Bucket = Bucket


    def test_floor_applies_when_the_reset_has_elapsed(self):
        held = self.Bucket(60.0, 10).hold_until(time.time() - 100, floor=4.0)
        self.assertEqual(held, 4.0)

    def test_a_real_future_reset_is_honoured_over_the_floor(self):
        held = self.Bucket(60.0, 10).hold_until(time.time() + 30, floor=4.0)
        self.assertAlmostEqual(held, 30.0, delta=1.0)

    def test_the_hold_is_actually_enforced(self):
        bucket = self.Bucket(60.0, 10, capacity=10)
        bucket.hold_until(time.time() - 100, floor=2.0)
        self.assertGreater(bucket._until, time.monotonic())

    def test_floor_grows_per_attempt(self):
        """A persistently stale reset must back off, not spin at one second."""
        floors = [min(60.0, 2.0 ** a) for a in range(6)]
        self.assertEqual(floors, [1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        self.assertTrue(all(b > a for a, b in zip(floors, floors[1:])))


class LowYieldRetirementIsOptIn(unittest.TestCase):
    """`low_yield` retirement is permanent and measures overlap with earlier
    shards rather than this shard's depth, so it is off by default."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, discover, log

        for mod in (config, log, discover):
            importlib.reload(mod)
        self.discover, self.log = discover, log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _reasons(self):
        return [r.get("reason") for r in self.log.read("shards")]

    def test_the_default_never_retires_on_yield(self):
        from csvscrape import cli

        args = cli.build_parser().parse_args(["discover"])
        self.assertEqual(args.min_new, 0)




    def test_genuine_exhaustion_still_retires(self):
        gh = StubGitHub(per_page=3)   # short page = the shard really is done
        self.discover.run(gh, max_shards=2, max_pages=5)
        self.assertEqual(set(self._reasons()), {"exhausted"})
        self.assertEqual(len(self.discover.completed()), 2)


class TreeRetryPolicy(unittest.TestCase):
    """Only success or a permanent failure (404, 451) retires a repo."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, log, trees

        for mod in (config, log, trees):
            importlib.reload(mod)
        self.trees, self.log = trees, log
        with self.log.Writer("discovery", rotate=1000) as w:
            for i in range(6):
                w.write({"repo_id": i, "full_name": f"o/r{i}", "path": "a.csv",
                         "blob_sha": "s", "tier": "hi", "shard": "q",
                         "size_hint": 1000, "fragment": None, "ts": 1})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    class GH:
        def __init__(self, outcome): self.outcome = outcome
        def tree(self, full_name, ref="HEAD"):
            return self.outcome(int(full_name.split("r")[1]))
        def raw(self, *a, **k): return None

    # `sha` is the commit the call resolved to, and every real response carries
    # one -- a scan that finds none is an error now, so a fixture without it
    # would exercise that path instead of the retry policy this class is about.
    OK = {"sha": "a" * 40,
          "tree": [{"path": "a.csv", "sha": "s", "size": 900, "type": "blob"}]}

    def test_a_rate_limited_repo_stays_queued(self):
        self.trees.run(self.GH(lambda n: {"error": 403}), workers=1)
        self.assertEqual(len(self.trees.pending()), 6, "403 retired the repo")


    def test_a_thrown_exception_stays_queued(self):
        def boom(n):
            raise RuntimeError("connection reset")
        self.trees.run(self.GH(boom), workers=1)
        self.assertEqual(len(self.trees.pending()), 6)

    def test_a_deleted_repo_retires(self):
        """404 will never succeed; retrying it forever is waste."""
        self.trees.run(self.GH(lambda n: {"error": 404}), workers=1)
        self.assertEqual(self.trees.pending(), [])


    def test_success_retires(self):
        self.trees.run(self.GH(lambda n: self.OK), workers=1)
        self.assertEqual(self.trees.pending(), [])


    def test_a_later_success_beats_an_earlier_failure(self):
        """Both records exist in the log; the newest decides."""
        self.trees.run(self.GH(lambda n: {"error": 500}), workers=1)
        self.trees.run(self.GH(lambda n: self.OK), workers=1)
        self.assertEqual(self.trees.scanned(), set(range(6)))


class ShardSchedule(unittest.TestCase):
    """The shard schedule's structural invariants."""

    SIZE_TIERS = {"lfs", "hi", "mid", "low", "tiny", "micro"}
    EXTENSIONS = {"csv", "tsv", "psv"}

    def setUp(self):
        from csvscrape import discover

        self.discover = discover
        self.shards = list(discover.shards())


    def test_every_extension_reaches_every_size_tier(self):
        """A missing (ext, tier) cell is a hole in the size spectrum."""
        cells = {(tier, ext) for tier, ext, _, _ in self.shards}
        for ext in self.EXTENSIONS:
            for tier in self.SIZE_TIERS:
                self.assertIn((tier, ext), cells, f"{ext} has no {tier} shard")

    def test_query_strings_are_unique(self):
        """A query string is a shard's identity in the log; duplicates would
        make `completed()` retire two shards on one shard's evidence."""
        queries = [q for _, _, q, _ in self.shards]
        self.assertEqual(len(set(queries)), len(queries))

    def test_tiers_tile_without_gaps_or_overlaps(self):
        for ext in self.EXTENSIONS:
            tiers = (self.discover.TIERS if ext == "csv"
                     else list(self.discover._scaled_tiers(ext)))
            for tier, lo, hi, step in tiers:
                ranges = sorted(
                    tuple(int(n) for n in q.split("size:")[1].split(".."))
                    for _, _, q, _ in self.discover._tier_shards(tier, lo, hi, step, ext)
                )
                self.assertEqual(ranges[0][0], lo, f"{ext}/{tier} starts late")
                self.assertEqual(ranges[-1][1], hi - 1, f"{ext}/{tier} ends early")
                for left, right in zip(ranges, ranges[1:]):
                    self.assertEqual(left[1] + 1, right[0],
                                     f"{ext}/{tier} gap or overlap at {left}")

    def test_no_shard_exceeds_the_result_cap_it_can_reach(self):
        """A tier is never given more shards than its population can fill."""
        from csvscrape import config

        for ext in ("tsv", "psv"):
            for tier, lo, hi, step in self.discover._scaled_tiers(ext):
                n = sum(1 for _ in self.discover._tier_shards(tier, lo, hi, step, ext))
                total = self.discover.SEARCH_TOTALS[ext][tier]
                self.assertLessEqual(
                    n, max(1, -(-total // config.CODE_SEARCH_RESULT_CAP)),
                    f"{ext}/{tier}: {n} shards for {total:,} files",
                )


class ConcurrentWriters(unittest.TestCase):
    """Several processes may write one log: none may seal, truncate or share
    another's live shard."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        import importlib

        from csvscrape import config, log

        importlib.reload(config)
        importlib.reload(log)
        self.log = log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def test_second_writer_leaves_a_live_shard_alone(self):
        first = self.log.Writer("probe", rotate=1000, flush_records=2)
        for i in range(5):
            first.write({"i": i})

        second = self.log.Writer("probe", rotate=1000, flush_records=2)
        try:
            # The live shard is untouched, so the first writer can still finish.
            first.write({"i": 5})
            first.close()
        finally:
            second.close()

        seen = sorted(r["i"] for r in self.log.read("probe"))
        self.assertEqual(seen, [0, 1, 2, 3, 4, 5], "records lost to a reseal")

    def test_two_writers_do_not_share_a_shard_index(self):
        first = self.log.Writer("probe")
        second = self.log.Writer("probe")
        try:
            self.assertNotEqual(first._index, second._index)
        finally:
            first.close()
            second.close()

    def test_rolling_onto_a_live_shard_steps_past_it(self):
        """A roll picks the next index without rescanning the directory, so
        it can land on a shard another writer opened meanwhile. Opening that
        shard would truncate it."""
        first = self.log.Writer("probe", rotate=2)
        second = self.log.Writer("probe")
        try:
            second.write({"i": "second"})
            second.flush()
            first.write({"i": 0})
            first.write({"i": 1})          # rolls onto second's index
            first.write({"i": 2})
            self.assertNotEqual(first._index, second._index)
            second.write({"i": "second-2"})
        finally:
            first.close()
            second.close()

        seen = sorted(str(r["i"]) for r in self.log.read("probe"))
        self.assertEqual(seen, ["0", "1", "2", "second", "second-2"])

    def test_a_filesystem_without_locks_does_not_read_as_live(self):
        """Only contention means a writer holds the shard; ENOLCK (a network
        filesystem without lock support) must not leave it unsealed forever,
        nor stop a writer from opening a shard at all."""
        import errno

        def no_locks(fd, op):
            raise OSError(errno.ENOLCK, "No locks available")

        with unittest.mock.patch.object(self.log.fcntl, "flock", no_locks):
            writer = self.log.Writer("probe")
            try:
                writer.write({"i": 0})
                self.assertFalse(self.log._is_live(writer._path))
            finally:
                writer.close()
        self.assertEqual([r["i"] for r in self.log.read("probe")], [0])

    def test_closing_twice_is_a_no_op(self):
        writer = self.log.Writer("probe")
        writer.write({"i": 0})
        writer.close()
        writer.close()  # must not raise from a rename of an already-moved file
        self.assertEqual([r["i"] for r in self.log.read("probe")], [0])

    def test_a_crashed_writer_is_still_sealed(self):
        """The lock must not make an abandoned shard unreadable forever."""
        abandoned = self.log.Writer("probe", flush_records=1)
        abandoned.write({"i": 0})
        abandoned._out.flush(__import__("zstandard").FLUSH_FRAME)
        abandoned._raw.flush()
        abandoned._raw.close()  # process died: kernel drops the flock

        reviver = self.log.Writer("probe")
        try:
            self.assertEqual([r["i"] for r in self.log.read("probe")], [0])
        finally:
            reviver.close()


class TheMeteredPoolsCountTheirOwnRejections(unittest.TestCase):
    """`Bucket` counts the requests the server refused for a rate limit, and only
    those."""

    def run_one(self, status, remaining="0", content=b"", retry_after=None,
                limiter=None):
        from csvscrape.client import GitHub

        headers = {"x-ratelimit-remaining": remaining,
                   "x-ratelimit-reset": str(time.time() + 30)}
        if retry_after is not None:
            headers["retry-after"] = retry_after

        class Resp:
            status_code = status
            content = b""
            @staticmethod
            def json():
                return {}
        Resp.headers = headers
        Resp.content = content

        gh = GitHub.__new__(GitHub)
        gh.on_event = lambda e, **kw: None
        gh.session = type("S", (), {
            "request": staticmethod(lambda *a, **k: Resp())})()
        limiter.take = lambda on_wait=None: None
        limiter.hold_for = lambda seconds: None
        limiter.hold_until = lambda epoch, floor=0.0: 0.0
        gh._request("GET", "https://example.invalid", limiter)
        return limiter

    def bucket(self):
        from csvscrape.client import Bucket

        return Bucket(3600.0, 4750.0, capacity=10, name="core")

    def test_a_spent_primary_limit_is_counted(self):
        got = self.run_one(429, remaining="0", limiter=self.bucket())
        self.assertEqual(got.stats["core_throttled"], 6,
                         "a refusal on every attempt should be six rejections")

    def test_a_retry_after_is_counted(self):
        got = self.run_one(403, remaining="5", retry_after="1",
                           limiter=self.bucket())
        self.assertEqual(got.stats["core_throttled"], 6)

    def test_a_secondary_limit_named_only_in_the_body_is_counted(self):
        got = self.run_one(403, remaining="5",
                           content=b"You have exceeded a secondary rate limit",
                           limiter=self.bucket())
        self.assertEqual(got.stats["core_throttled"], 6)

    def test_an_ordinary_403_is_not_counted(self):
        """A 403 for a private or blocked repo is not a rate limit."""
        got = self.run_one(403, remaining="4999", content=b"Must have push access",
                           limiter=self.bucket())
        self.assertEqual(got.stats["core_throttled"], 0)

    def test_a_served_request_that_spends_the_window_is_not_counted(self):
        got = self.run_one(200, remaining="0", limiter=self.bucket())
        self.assertEqual(got.stats["core_throttled"], 0,
                         "spending the last slot is the pacing working")


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class TheDrawHasTwoShapes(unittest.TestCase):
    """The benchmarking corpus and the correctness corpus want opposite things
    from the same pool, and the difference is the whole reason `--sampling`
    exists. Log-uniform flattens the size axis so a throughput benchmark sees
    every octave; uniform leaves it alone so the corpus generalises to "a CSV
    file on GitHub", which is the only frame a correctness comparison can be
    read in."""

    # A pool shaped like the real one: an order of magnitude fewer files with
    # every octave climbed.
    # Big enough at the bottom that the hash-residue thinning actually
    # engages, which is the one thing a small fixture cannot show.
    SHAPE = ((10, 20_000), (15, 100), (20, 10))

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, index, log, probe, sample, selectors

        for mod in (config, log, index, probe, sample, selectors):
            importlib.reload(mod)
        self.sample, self.index = sample, index

        (self.tmp / "index").mkdir(parents=True)
        rows, i = [], 0
        for bucket, count in self.SHAPE:
            for j in range(count):
                rows.append({
                    "repo_id": i, "owner": f"o{i % 400}", "full_name": f"o{i % 400}/r{i}",
                    "path": f"f{i}.csv", "blob_sha": f"s{i}", "content_key": f"{i:08d}",
                    "size": 2 ** bucket + j, "raw_size": 2 ** bucket, "bucket": bucket,
                    "is_lfs": False, "lfs_suspect": False, "tier": "hi",
                    "truncated": False, "ref": "e" * 40,
                })
                i += 1
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _draw(self, mode, n=60, seed=1):
        pool = self.sample.candidates(select="main")
        available = self.sample.supply(pool, mode, sizes=(mode == self.sample.UNIFORM))
        quota = self.sample.allocate(
            {s: rows for s, (rows, _) in available.items()}, n)
        picked, short = self.sample.draw(quota, seed=seed, per_repo=5, per_owner=5,
                                         select="main", pool=pool, mode=mode)
        counts = dict(picked.group_by("bucket").len().iter_rows())
        return quota, counts, short, picked, available

    def test_log_uniform_gives_every_octave_the_same_count(self):
        quota, counts, _, picked, _ = self._draw(self.sample.LOG_UNIFORM)
        self.assertEqual(picked.height, 60)
        # Bucket 20 holds ten files and takes all ten; the water-filling hands
        # the fifty it cannot use back to the two buckets that can.
        self.assertEqual(counts[20], 10)
        self.assertEqual(counts[10], counts[15])
        self.assertEqual(sorted(quota), [10, 15, 20])

    def test_uniform_reproduces_the_pool_it_was_drawn_from(self):
        """Not equal counts per octave -- proportional ones. The 1000:100:10
        supply has to come back out at roughly 100:10:1."""
        quota, counts, _, picked, _ = self._draw(self.sample.UNIFORM)
        self.assertEqual(list(quota), [self.sample.WHOLE_POOL])
        self.assertEqual(picked.height, 60)
        self.assertGreater(counts[10], 45, f"small files under-drawn: {counts}")
        self.assertLess(counts.get(20, 0), 3, f"large files over-drawn: {counts}")
        self.assertGreater(counts[10], 5 * counts.get(15, 0))

    def test_uniform_does_not_favour_the_small_end_within_a_size(self):
        """Log-uniform weights by 1/size *inside* an octave, which is exactly
        the tilt a uniform draw must not have."""
        import statistics

        pool = self.sample.candidates(select="main").filter(
            __import__("polars").col("bucket") == 10)
        # One octave only, so the per-octave allocation cannot be what shows.
        sizes = {}
        for mode in (self.sample.LOG_UNIFORM, self.sample.UNIFORM):
            picked, _ = self.sample.draw({10: 300} if mode == self.sample.LOG_UNIFORM
                                         else {self.sample.WHOLE_POOL: 300},
                                         seed=3, per_repo=5, per_owner=5,
                                         select="main", pool=pool, mode=mode)
            sizes[mode] = statistics.mean(picked["size"])
        self.assertLess(sizes[self.sample.LOG_UNIFORM], sizes[self.sample.UNIFORM],
                        "the 1/size weighting did not distinguish the modes")



    def test_both_modes_are_reproducible_from_the_seed(self):
        for mode in (self.sample.LOG_UNIFORM, self.sample.UNIFORM):
            first = self._draw(mode)[3]["content_key"].to_list()
            second = self._draw(mode)[3]["content_key"].to_list()
            self.assertEqual(first, second, mode)

    def test_the_default_seed_does_not_fall_off_the_bottom_of_the_hash(self):
        """`WHOLE_POOL` is -1 and Polars' `hash` seed is unsigned, so the per-
        stratum seed must stay non-negative, including for the default
        seed."""
        for mode in (self.sample.UNIFORM, self.sample.LOG_UNIFORM):
            self.assertEqual(self._draw(mode, n=20, seed=0)[3].height, 20, mode)

    def test_a_column_null_across_the_inference_window_does_not_fail_the_draw(self):
        """A metadata column that is null in the first rows (unprobed files) and
        set later must not fail building the drawn frame."""
        import polars as pl

        from csvscrape import index

        frame = pl.read_parquet(index.INDEX_PATH)
        frame.with_columns(
            comment=pl.when(pl.col("bucket") == 10).then(pl.lit("#"))
            .otherwise(pl.lit(None, dtype=pl.String))
        ).write_parquet(index.INDEX_PATH)
        # Descending stratum order puts the two all-null buckets (110 rows
        # between them) ahead of the one that carries a value.
        _, _, _, picked, _ = self._draw(self.sample.LOG_UNIFORM, n=400)
        self.assertEqual(picked.height, 400)
        self.assertEqual(picked.schema["comment"], pl.String)
        self.assertEqual(picked["comment"].null_count(), 110)


    def test_the_cli_offers_exactly_the_modes_sample_implements(self):
        """`cli` spells them out so building the parser does not import Polars;
        this is the link that keeps the two lists equal."""
        from csvscrape import cli

        self.assertEqual(tuple(cli.SAMPLING_MODES), tuple(self.sample.MODES))


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class SymlinksLeaveThePool(unittest.TestCase):
    """Symlinks (mode 120000) leave the pool at compaction: their bytes are a
    target path, and what they point at is either indexed itself or outside
    the tree."""

    def _build(self, modes):
        import importlib

        from csvscrape import config, index, log

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        self.addCleanup(os.environ.pop, "CSVSCRAPE_DATA", None)
        os.environ["CSVSCRAPE_DATA"] = str(tmp)
        for mod in (config, log, index):
            importlib.reload(mod)
        (tmp / "trees").mkdir(parents=True)
        record = {"repo_id": 1, "full_name": "o/a", "ref": "a" * 40, "state": "ok",
                  "truncated": False, "capped": False, "n_blobs": 2, "n_kept": 2,
                  "tier": "hi", "ts": 1,
                  "paths": ["real.csv", "link.csv"], "shas": ["s1", "s2"],
                  "sizes": [4096, 20]}
        if modes is not None:
            record["modes"] = modes
        with (tmp / "trees" / "part-00000.jsonl").open("w") as fh:
            fh.write(json.dumps(record) + "\n")
        index.build()
        return set(index.files().collect()["path"])

    def test_a_symlink_is_dropped_and_the_real_file_is_not(self):
        self.assertEqual(self._build(["100644", "120000"]), {"real.csv"})

    def test_an_executable_blob_is_not_a_symlink(self):
        self.assertEqual(self._build(["100644", "100755"]),
                         {"real.csv", "link.csv"})

    def test_a_log_written_before_modes_existed_indexes_as_it_always_did(self):
        """Not knowing is not the same as knowing it is a link, and dropping a
        whole vintage of the log would be the larger error."""
        self.assertEqual(self._build(None), {"real.csv", "link.csv"})


class MinimalFilteringIsTheDefault(unittest.TestCase):
    """The main corpora are ruined far more by a filter that excludes real
    CSVs than by a stray non-CSV that survives: one is a silent bias across the
    population, the other is a file you notice and delete. `not_binary` is
    therefore the narrowest useful gate, and this pins what it does *not*
    reject as firmly as what it does."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _file(self, data: bytes) -> Path:
        p = self.tmp / "f.csv"
        p.write_bytes(data)
        return p

    def _why(self, data: bytes):
        from csvscrape.selectors import not_binary

        return not_binary(self._file(data), {})

    def test_magic_prefixed_content_is_rejected(self):
        for prefix, kind in ((b"\x1f\x8b", "gzip"), (b"PK\x03\x04", "zip/xlsx"),
                             (b"PAR1", "parquet"), (b"%PDF", "pdf")):
            self.assertEqual(self._why(prefix + b"a,b\n1,2\n" * 100), f"magic/{kind}")

    def test_control_byte_soup_is_rejected(self):
        why = self._why(bytes(range(1, 20)) * 500)
        self.assertTrue(why.startswith("control_bytes"), why)

    def test_a_plain_csv_is_kept(self):
        self.assertIsNone(self._why(b"a,b\n1,2\n" * 500))

    def test_latin1_and_cjk_are_kept(self):
        """Not UTF-8 is not "not CSV" -- those files are the point."""
        self.assertIsNone(self._why("café;ä\n1;2\n".encode("latin-1") * 500))
        self.assertIsNone(self._why("名前,値\n1,2\n".encode("shift_jis") * 500))

    def test_utf16_is_kept_despite_being_half_nul(self):
        self.assertIsNone(self._why("a,b\n1,2\n".encode("utf-16-le") * 500))

    def test_html_is_kept_by_not_binary_and_dropped_by_text_only(self):
        """The one deliberate gap. HTML is text, so `main` keeps it and you
        delete the handful that turn up; `text-only` is for anyone who would
        rather not see them."""
        from csvscrape.selectors import not_markup

        page = b"<!DOCTYPE html>\n<html><body>nope</body></html>\n"
        self.assertIsNone(self._why(page))
        self.assertTrue(not_markup(self._file(page), {}).startswith("markup/"))

    def test_a_degenerate_but_real_file_is_kept(self):
        for data in (b"# only comments\n" * 100, b"\n" * 400, b"a\n", b","):
            self.assertIsNone(self._why(data), data[:20])


class TheUtf8FlagComposesWithEverySelector(unittest.TestCase):
    """`--utf8` is orthogonal to what a selector is *for*, so it composes onto
    one rather than doubling the registry."""

    def test_it_chains_onto_the_selector_s_own_verifier(self):
        from csvscrape import selectors

        base = selectors.get("main")
        composed = selectors.resolve("main", utf8=True)
        self.assertEqual(composed.name, "main+utf8")
        self.assertIsNot(composed.verify, base.verify)
        self.assertEqual(composed.needs_probe, base.needs_probe)

    def test_the_selector_s_own_verdict_comes_first(self):
        """`whole_file_utf8` reads every byte, so it must not run before a
        check that can reject on the first page."""
        from csvscrape import selectors

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "f.csv"
        # Binary *and* not UTF-8: whichever verifier answers first names it.
        path.write_bytes(b"PK\x03\x04" + bytes(range(128, 256)) * 100)
        why = selectors.resolve("main", utf8=True).verify(path, {})
        self.assertEqual(why, "magic/zip/xlsx")


    def test_every_selector_survives_being_composed(self):
        from csvscrape import selectors

        for name in selectors.REGISTRY:
            composed = selectors.resolve(name, utf8=True)
            self.assertTrue(callable(composed.filter), name)
            self.assertTrue(callable(composed.verify), name)


@unittest.skipUnless(HAVE_POLARS, "polars not installed")
class DatasetsAreDrawnSideBySide(unittest.TestCase):
    """Two corpora over one pool. Their bookkeeping has to be separate -- a
    shared log would have the second run read the first's files as its own
    progress and stop early -- while the bytes underneath stay shared, which is
    what makes the overlap free."""

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import config, download, index, log, probe, sample, selectors

        for mod in (config, log, index, probe, sample, selectors, download):
            importlib.reload(mod)
        self.download, self.log = download, log
        (self.tmp / "index").mkdir(parents=True)
        pl.DataFrame([{
            "repo_id": i, "owner": "o", "full_name": f"o/r{i}", "path": f"f{i}.csv",
            "blob_sha": f"s{i}", "content_key": f"k{i}", "size": 4096,
            "raw_size": 4096, "bucket": 12, "is_lfs": False, "lfs_suspect": False,
            "tier": "hi", "truncated": False, "ref": "e" * 40,
        } for i in range(10)]).write_parquet(index.INDEX_PATH)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("CSVSCRAPE_DATA", None)

    def _record(self, dataset, key):
        with self.log.Writer(self.download.log_name(dataset), rotate=100) as w:
            w.write({"content_key": key, "repo_id": 1, "full_name": "o/r",
                     "path": "a.csv", "bucket": 12, "stratum": 12, "size": 4096,
                     "is_lfs": False, "state": "ok", "bytes": 4096, "ts": 1})

    def test_one_dataset_s_outcomes_are_not_another_s(self):
        self._record("bench", "k1")
        self._record("correctness", "k2")
        self.assertEqual(self.download.outcomes("bench")[0], {"k1"})
        self.assertEqual(self.download.outcomes("correctness")[0], {"k2"})

    def test_each_dataset_gets_its_own_manifest(self):
        self._record("bench", "k1")
        self._record("correctness", "k2")
        for name, key in (("bench", "k1"), ("correctness", "k2")):
            path = self.download._write_manifest(name)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([r["content_key"] for r in rows], [key], name)

    def test_datasets_lists_what_has_been_drawn(self):
        self._record("bench", "k1")
        self._record("main-uniform", "k2")
        self.assertEqual(self.download.datasets(), ["bench", "main-uniform"])

    def test_the_draw_does_not_exclude_what_another_dataset_kept(self):
        """Overlap between the corpora is fine and excluding it would bias
        both; only *this* dataset's own history narrows its draw."""
        from csvscrape import sample

        self._record("bench", "k1")
        kept, _ = self.download.outcomes("correctness")
        self.assertEqual(kept, set())
        drawn = sample.candidates(exclude=kept, select="main").collect()
        self.assertIn("k1", drawn["content_key"].to_list())

    def test_the_cli_names_a_dataset_per_combination_by_default(self):
        """A shared default name would have the correctness run read the
        benchmark run's log as its own progress."""
        from csvscrape import cli

        parser = cli.build_parser()
        bench = parser.parse_args(["download", "-n", "5"])
        correctness = parser.parse_args(["download", "-n", "5", "--sampling", "uniform"])
        self.assertIsNone(bench.dataset)
        names = set()
        for args in (bench, correctness):
            names.add(args.dataset
                      or f"{args.select}-{args.sampling}" + ("-utf8" if args.utf8 else ""))
        self.assertEqual(len(names), 2, names)


class NormalisingSnifferVerdicts(unittest.TestCase):
    """Each tool spells "there is no such character" differently, and every one
    of those spellings is a plausible-looking one-or-seven-character string
    that would land in the manifest as a real delimiter if it were stored
    unexamined."""

    def test_clevercsv_empty_string_is_an_explicit_none(self):
        """`""` is CleverCSV saying it looked and found no quote, which is an
        answer -- not the silence of a sniffer that was never asked."""
        from csvscrape.dialects import NONE, from_clevercsv

        view = from_clevercsv({"delimiter": ";", "quotechar": "", "escapechar": ""})
        self.assertEqual(view["delimiter"], ";")
        self.assertEqual(view["quote"], NONE)
        self.assertEqual(view["escape"], NONE)
        self.assertIsNone(view["comment"], "CleverCSV has no comment notion")
        self.assertIsNone(view["terminator"])

    def test_duckdb_says_none_with_the_word_empty(self):
        """DuckDB reports Quote/Escape/Comment as the literal string "(empty)",
        not an empty one."""
        from csvscrape.dialects import from_duckdb

        from csvscrape.dialects import NONE

        view = from_duckdb({"Delimiter": ";", "Quote": "(empty)",
                            "Escape": "(empty)", "Comment": "(empty)",
                            "NewLineDelimiter": "\\n"})
        self.assertEqual(view["quote"], NONE)
        self.assertEqual(view["escape"], NONE)
        self.assertEqual(view["comment"], NONE)
        self.assertEqual(view["terminator"], "\n")

    def test_duckdb_newline_is_escape_text_not_a_byte(self):
        from csvscrape.dialects import from_duckdb

        self.assertEqual(from_duckdb({"NewLineDelimiter": "\\r\\n"})["terminator"],
                         "crlf")

    def test_duckdb_escape_equal_to_quote_is_rfc_doubling(self):
        """DuckDB reports `"` as the escape for an ordinary doubled-quote file.
        Recorded as `escape` it would configure the parser to treat a closing
        quote as an escape character; the struct has a flag for this case."""
        from csvscrape.dialects import from_duckdb

        from csvscrape.dialects import NONE

        view = from_duckdb({"Delimiter": ",", "Quote": '"', "Escape": '"'})
        self.assertEqual(view["escape"], NONE)
        self.assertTrue(view["double_quote_escape"])

    def test_custom_reports_a_missing_character_as_null_and_means_none(self):
        """`Shape` looks for every character it has a field for, so its None
        is an answer, unlike a None from a tool that has no such field."""
        from csvscrape.dialects import NONE, from_custom

        view = from_custom({"delimiter": None, "quote": '"', "quote_style": "strict",
                            "escape": None, "comment": None, "terminator": "lf"})
        self.assertEqual(view["delimiter"], NONE)
        self.assertEqual(view["quote"], '"')
        self.assertEqual(view["escape"], NONE)
        self.assertEqual(view["comment"], NONE)

    def test_a_multi_byte_delimiter_is_no_opinion(self):
        from csvscrape.dialects import byte

        self.assertIsNone(byte("::"))
        self.assertIsNone(byte("€"))          # U+20AC, above 0xFF
        self.assertEqual(byte("¤"), "¤")   # latin-1, a real byte

    def test_an_errored_sniffer_votes_on_nothing(self):
        from csvscrape.dialects import FIELDS, from_clevercsv, from_duckdb

        for view in (from_clevercsv({"error": "NoDetectionResult"}),
                     from_duckdb({"error": "IOException"})):
            self.assertEqual(set(view), set(FIELDS))
            self.assertTrue(all(v is None for v in view.values()))

    def test_errors_are_collected_beside_the_views(self):
        """An errored view is indistinguishable from abstention, so the failure
        itself has to be recorded somewhere a review can see it."""
        from csvscrape.dialects import errors

        raw = {"custom": {"delimiter": ","},
               "clevercsv": {"error": "no_dialect"},
               "duckdb": {"error": "OutOfMemoryException"},
               "chardet": {}}
        self.assertEqual(errors(raw), {"clevercsv": "no_dialect",
                                       "duckdb": "OutOfMemoryException",
                                       "chardet": "no_output"})
        self.assertEqual(errors({"custom": {"delimiter": ","}}), {})


class TheDialectIsOursTheOthersAreACheck(unittest.TestCase):
    def _views(self, **votes):
        from csvscrape.dialects import FIELDS

        return {name: dict.fromkeys(FIELDS) | view for name, view in votes.items()}

    def test_the_dialect_is_customs_whatever_the_others_say(self):
        from csvscrape.dialects import NONE, settle

        out = settle(self._views(custom={"delimiter": ";", "quote": NONE},
                                 clevercsv={"delimiter": ","},
                                 duckdb={"delimiter": ","}), b"a;b,c\n")
        self.assertEqual(out["dialect"]["delimiter"], ";")
        self.assertIsNone(out["dialect"]["quote"], "none is null in a dialect")
        self.assertEqual(out["agreement"], 1)
        self.assertEqual(out["contested"], ["delimiter"])

    def test_no_opinion_is_not_a_disagreement(self):
        from csvscrape.dialects import settle

        out = settle(self._views(custom={"delimiter": ",", "comment": "#"},
                                 clevercsv={"delimiter": ","}), b"#x\na,b\n")
        self.assertEqual(out["contested"], [])
        self.assertEqual(out["agreement"], 2)

    def test_a_character_that_is_not_in_the_file_is_a_default(self):
        """DuckDB reports `"` for a file without a single quote."""
        from csvscrape.dialects import NONE, settle

        views = self._views(custom={"delimiter": ",", "quote": NONE},
                            duckdb={"delimiter": ",", "quote": '"'})
        self.assertEqual(settle(views, b"a,b\n1,2\n")["contested"], [])
        self.assertEqual(settle(views, b'a,5" pipe\n')["contested"], ["quote"])

    def test_none_against_a_character_is_a_disagreement(self):
        from csvscrape.dialects import NONE, settle

        views = self._views(custom={"delimiter": ",", "quote": '"'},
                            clevercsv={"delimiter": ",", "quote": NONE})
        self.assertEqual(settle(views, b'"a",b\n')["contested"], ["quote"])


class TheDatasetDefinitionIsTheArtifact(unittest.TestCase):
    """`datasets/<name>.json` is what a parser benchmark consumes: sources,
    hashes and a dialect in the parser's own Config vocabulary. The bytes are
    not in it (that is what makes it distributable) and neither are the manual
    overrides (that is what keeps them from being clobbered)."""

    FILES = {
        "plain.csv":   b'a,b,c\n1,"x,y",3\n' * 300,
        "semi.csv":    b"a;b;c\n1;2;3\n" * 300,
        "comment.csv": b"# note\na,b\n1,2\n" * 300,
        "drift.csv":   b"a,b\n" * 300 + b"p;q;r\n" * 300,
    }

    def setUp(self):
        import importlib

        import polars as pl

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(os.environ.pop, "CSVSCRAPE_DATA", None)
        os.environ["CSVSCRAPE_DATA"] = str(self.tmp)
        from csvscrape import (config, dataset, download, index, log, probe,
                               sample, selectors, validate)
        for mod in (config, log, index, probe, sample, selectors, download, dataset):
            importlib.reload(mod)
        self.dataset, self.download = dataset, download
        # The artifact directory is next to the package; a test must not write
        # into the checkout.
        dataset.ROOT = self.tmp / "datasets"
        dataset.OVERRIDES = dataset.ROOT / "overrides.toml"

        (self.tmp / "index").mkdir(parents=True)
        self.bodies, rows = {}, []
        for i, (name, body) in enumerate(self.FILES.items()):
            key = validate.git_blob_sha1(len(body), iter([body]))
            self.bodies[key] = body
            rows.append({"repo_id": i, "owner": f"o{i}", "full_name": f"o{i}/r{i}",
                         "path": name, "blob_sha": key, "content_key": key,
                         "size": len(body), "raw_size": len(body),
                         "bucket": index.size_bucket(len(body)), "is_lfs": False,
                         "lfs_suspect": False, "tier": "hi", "truncated": False,
                         "ref": "e" * 40})
        self.rows = rows
        pl.DataFrame(rows).write_parquet(index.INDEX_PATH)
        self.fetches = []

    def _gh(self):
        import hashlib

        bodies, rows, fetches = self.bodies, self.rows, self.fetches

        class GH:
            def stream(self, url, sink, chunk=1 << 20):
                fetches.append(url)
                row = next(r for r in rows if url.endswith(r["path"]))
                body = bodies[row["content_key"]]
                sink.write(body)
                return len(body), hashlib.sha256(body).hexdigest()

        return GH()

    def _run(self, **kw):
        import json

        self.download.run(self._gh(), n=len(self.FILES), dataset="demo",
                          mode="uniform", sniff_procs=1, **kw)
        return json.loads(self.dataset.manifest_path("demo").read_text())

    def _by_path(self, doc):
        return {e["path"]: e for e in doc["files"].values()}

    def test_size_bounds_narrow_the_draw_inclusively(self):
        """semi.csv is 3,600 B and comment.csv 4,500 B: both bounds are
        inclusive, so exactly those two land and nothing else is fetched."""
        self.download.run(self._gh(), n=len(self.FILES), dataset="demo",
                          mode="uniform", sniff=False,
                          min_size=3600, max_size=4500)
        kept = {r["path"] for r in self.download.members("demo")}
        self.assertEqual(kept, {"semi.csv", "comment.csv"})
        self.assertEqual(len(self.fetches), 2)

    def test_a_dropped_file_is_replaced_and_never_redrawn(self):
        """Uniform draw of 3 of the 4 files; dropping one leaves a deficit the
        next run fills with the only file left, not the dropped one."""
        run = lambda: self.download.run(self._gh(), n=3, dataset="demo",
                                        mode="uniform", sniff=False)
        run()
        before = {r["content_key"] for r in self.download.members("demo")}
        self.assertEqual(len(before), 3)
        gone = sorted(before)[0]
        self.download.drop("demo", [gone], "not_csv")
        self.assertNotIn(gone, {r["content_key"]
                                for r in self.download.members("demo")})

        stats = run()
        after = {r["content_key"] for r in self.download.members("demo")}
        self.assertEqual(after, set(self.bodies) - {gone})
        self.assertEqual(stats["kept"], 3)
        kept, rejected = self.download.outcomes("demo")
        self.assertIn(gone, rejected)
        self.assertNotIn(gone, kept)

    def test_dropping_a_file_outside_the_dataset_is_an_error(self):
        with self.assertRaises(SystemExit):
            self.download.drop("demo", ["f" * 40], "nope")

    def test_crossed_size_bounds_are_an_error_not_an_empty_draw(self):
        stats = self.download.run(self._gh(), n=1, dataset="demo",
                                  min_size=10, max_size=5)
        self.assertIn("--min-size", stats["error"])
        self.assertEqual(self.fetches, [])

    def test_it_records_how_the_dataset_was_drawn(self):
        doc = self._run()
        block = doc["dataset"]
        self.assertEqual(block["selector"], "main")
        self.assertEqual(block["sampling"], "uniform")
        self.assertEqual(block["files"], len(self.FILES))
        self.assertEqual(block["sniff_cap"], self.dataset.SNIFF_CAP)

    def test_every_file_carries_what_restore_needs(self):
        """Source, pin and hash -- the definition has to be self-contained, or
        a re-fetch addresses a branch tip instead of the scanned commit."""
        doc = self._run()
        for key, entry in doc["files"].items():
            self.assertRegex(key, r"^[0-9a-f]{40}$")
            self.assertEqual(entry["ref"], "e" * 40)
            self.assertTrue(entry["repo"] and entry["path"])
            self.assertIn("size", entry)

    def test_the_dialect_is_in_the_parsers_vocabulary(self):
        from csvscrape.dialects import FIELDS

        doc = self._run()
        entry = self._by_path(doc)["semi.csv"]
        self.assertEqual(set(entry["dialect"]), set(FIELDS))
        self.assertEqual(entry["dialect"]["delimiter"], ";")
        self.assertEqual(entry["dialect"]["terminator"], "\n")

    def test_each_sniffers_own_view_survives_next_to_the_consensus(self):
        doc = self._run()
        from csvscrape.dialects import FIELDS

        entry = self._by_path(doc)["comment.csv"]
        # Each view is the same nine keys, so a consumer reads every sniffer
        # the same way and a missing opinion is a null rather than an absent
        # key. What they actually voted stays visible beside the consensus.
        for name in ("custom", "clevercsv", "duckdb"):
            self.assertEqual(set(entry["sniffers"][name]), set(FIELDS), name)
        self.assertEqual(entry["dialect"]["comment"], "#",
                         "only DuckDB and custom report comments; the "
                         "abstainer must not have outvoted them")
        self.assertEqual(entry["agreement"], 3)

    def test_drift_a_head_sample_cannot_see_is_recorded(self):
        doc = self._run()
        entry = self._by_path(doc)["drift.csv"]
        self.assertEqual(entry["stats"]["common_field_count_share"], 0.5)
        self.assertIn("ragged", entry["doubts"])

    def test_a_quoted_file_is_not_dragged_into_the_review_queue(self):
        doc = self._run()
        entry = self._by_path(doc)["plain.csv"]
        self.assertEqual(entry["stats"]["common_field_count_share"], 1.0)
        self.assertEqual(entry["doubts"], [])
        self.assertNotIn(
            [k for k, e in doc["files"].items() if e["path"] == "plain.csv"][0],
            self.dataset.needs_review(doc["files"]))

    def test_overrides_are_not_merged_into_the_definition(self):
        """The whole point of keeping them apart: the sniffed answer stays
        legible underneath, and editing an override regenerates nothing."""
        doc = self._run()
        key = [k for k, e in doc["files"].items() if e["path"] == "semi.csv"][0]
        self.dataset.ROOT.mkdir(parents=True, exist_ok=True)
        self.dataset.OVERRIDES.write_text(
            f'[files."{key}"]\ndelimiter = "|"\n', encoding="utf-8")
        doc = self._run()
        self.assertEqual(doc["files"][key]["dialect"]["delimiter"], ";",
                         "an override leaked into the sniffed dialect")
        self.assertEqual(self.dataset.overrides()[key]["delimiter"], "|")

    def test_an_overridden_file_leaves_the_review_queue(self):
        doc = self._run()
        contested = self.dataset.needs_review(doc["files"])
        self.assertTrue(contested)
        key = sorted(contested)[0]
        self.dataset.OVERRIDES.write_text(
            f'[files."{key}"]\ndelimiter = ","\n', encoding="utf-8")
        self.assertNotIn(key, self.dataset.needs_review(doc["files"]))

    def test_the_review_file_is_not_the_overrides_file(self):
        """Generated and hand-written never share a file, so regenerating
        cannot overwrite a decision."""
        self._run()
        review = self.dataset.review_path("demo")
        self.assertTrue(review.exists())
        self.assertNotEqual(review, self.dataset.OVERRIDES)
        # Per dataset: a second corpus must not rewrite this one's queue.
        self.assertNotEqual(review, self.dataset.review_path("other"))
        self.assertFalse(self.dataset.OVERRIDES.exists(),
                         "the tool wrote the file it must never write")


    def test_rerunning_download_resniffs_without_refetching(self):
        """The reason there is no separate sniff command: the per-stratum
        deficit is empty on a complete dataset, so the fetch loop never runs
        and the pass reduces to sniff-and-rewrite."""
        self._run()
        first = len(self.fetches)
        self.assertEqual(first, len(self.FILES))
        before = self.dataset.manifest_path("demo").read_text()
        doc = self._run()
        self.assertEqual(len(self.fetches), first, "it re-downloaded the corpus")
        self.assertEqual(doc["dataset"]["files"], len(self.FILES))
        self.assertTrue(before, "nothing was written the first time")

    def test_no_sniff_leaves_the_definition_alone(self):
        self._run()
        stamp = self.dataset.manifest_path("demo").read_text()
        self.download.run(self._gh(), n=len(self.FILES), dataset="demo",
                          mode="uniform", sniff=False)
        self.assertEqual(self.dataset.manifest_path("demo").read_text(), stamp)


class ReviewQueueIsPerDataset(unittest.TestCase):
    """Each dataset has its own review queue; decisions in `overrides.toml` are
    shared."""

    def setUp(self):
        from csvscrape import dataset

        self.dataset = dataset
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(setattr, dataset, "ROOT", dataset.ROOT)
        self.addCleanup(setattr, dataset, "OVERRIDES", dataset.OVERRIDES)
        dataset.ROOT = self.tmp
        dataset.OVERRIDES = self.tmp / "overrides.toml"

    def _unsettled(self, key: str) -> dict:
        """One file the sniffers could not settle: no delimiter at all."""
        from csvscrape import dialects

        return {key: {
            "repo": "o/r", "path": f"{key}.csv", "size": 10,
            "dialect": {field: None for field in dialects.FIELDS},
            "sniffers": {}, "contested": False,
        }}

    def test_building_one_dataset_leaves_the_others_queue_alone(self):
        first = self.dataset.write_review(self._unsettled("aaa"), "bench")
        second = self.dataset.write_review(self._unsettled("bbb"), "kaggle-main")
        self.assertNotEqual(first, second)
        self.assertTrue(first.exists())
        self.assertTrue(second.exists())
        self.assertIn("aaa", first.read_text())
        self.assertIn("bbb", second.read_text())
        # The point: the first is still the first after the second was built.
        self.assertNotIn("bbb", first.read_text())

    def test_the_queue_sits_beside_its_definition(self):
        self.assertEqual(self.dataset.review_path("bench").parent,
                         self.dataset.manifest_path("bench").parent)
        self.assertTrue(
            self.dataset.review_path("bench").name.startswith("bench"))

    def test_the_path_follows_a_repointed_root(self):
        """Derived at call time, not bound at import. A path bound beside ROOT
        keeps writing to the original directory when a caller moves the root --
        which is exactly how the suite came to overwrite the real datasets."""
        moved = self.tmp / "elsewhere"
        self.dataset.ROOT = moved
        self.assertEqual(self.dataset.review_path("bench").parent, moved)

    def test_overrides_stay_shared_across_datasets(self):
        """A hash is a content address, so the same bytes have the same dialect
        in every corpus that drew them. Splitting these per dataset would make
        one decision need taking twice."""
        self.dataset.OVERRIDES.write_text(
            '[files."k"]\ndelimiter = "|"\n', encoding="utf-8")
        self.assertEqual(self.dataset.overrides()["k"]["delimiter"], "|")
        queue = self.dataset.needs_review(self._unsettled("k"))
        self.assertEqual(queue, {})

    def _settled(self, key: str, **extra) -> dict:
        from csvscrape import dialects

        return {key: {
            "repo": "o/r", "path": f"{key}.csv", "size": 10,
            "dialect": dict.fromkeys(dialects.FIELDS) | {"delimiter": ","},
            "sniffers": {}, "contested": [], "sniffer_errors": {}, "doubts": [],
        } | extra}

    def test_a_settled_file_stays_out(self):
        self.assertEqual(self.dataset.needs_review(self._settled("k")), {})

    def test_errors_and_doubts_only_queue_a_file_when_asked_to(self):
        for extra in ({"sniffer_errors": {"duckdb": "OutOfMemoryException"}},
                      {"doubts": ["ragged"]}):
            files = self._settled("k", **extra)
            self.assertEqual(self.dataset.needs_review(files), {})
            self.assertIn("k", self.dataset.needs_review(files, "all"))
        text = self.dataset.write_review(files, "bench", "all").read_text()
        self.assertIn("DOUBTS      ragged", text)

    def test_a_file_without_a_dialect_is_always_queued(self):
        files = self._settled("k")
        files["k"]["dialect"]["delimiter"] = None
        self.assertIn("k", self.dataset.needs_review(files))

    def test_the_review_tells_none_from_no_opinion(self):
        from csvscrape import dialects

        view = dict.fromkeys(dialects.FIELDS) | {"delimiter": "\t",
                                                 "quote": dialects.NONE}
        files = self._settled("k", contested=["quote"],
                              sniffers={"clevercsv": view})
        text = self.dataset.write_review(files, "bench").read_text()
        self.assertIn("clevercsv  delimiter='\\t' quote=none escape=? comment=?",
                      text)
        self.assertIn("CONTESTED   quote", text)


class ThreadCountsMustBePositive(unittest.TestCase):
    """Each stage of a pass waits for the stage before it to drain, so a stage
    with no threads never finishes and the pass hangs."""

    def test_zero_workers_are_rejected(self):
        import io

        from csvscrape import cli

        for flag in ("--workers", "--raw-workers"):
            with self.subTest(flag=flag), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["probe", flag, "0"])
        args = cli.build_parser().parse_args(["probe", "--workers", "1"])
        self.assertEqual(args.workers, 1)


class SizeFlags(unittest.TestCase):
    """`download --min-size/--max-size` take bytes or binary suffixes, the same
    units the buckets and every printed size use."""

    def test_suffixes_are_binary(self):
        from csvscrape import cli

        for text, want in [("4096", 4096), ("64K", 65536), ("64kib", 65536),
                           ("1.5M", 3 << 19), ("2GiB", 2 << 30), ("10b", 10)]:
            with self.subTest(text=text):
                self.assertEqual(cli._size(text), want)

    def test_nonsense_is_rejected(self):
        import io

        from csvscrape import cli

        for bad in ("", "K", "-1", "12X", "inf", "nan"):
            with self.subTest(bad=bad), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["download", "--min-size", bad])

    def test_both_default_to_unbounded(self):
        from csvscrape import cli

        args = cli.build_parser().parse_args(["download", "--max-size", "1M"])
        self.assertIsNone(args.min_size)
        self.assertEqual(args.max_size, 1 << 20)



class FetchFromADefinition(unittest.TestCase):
    """`fetch` rebuilds the files of a dataset from `datasets/<name>.json`."""

    def test_missing_files_are_fetched_and_lost_ones_reported(self):
        import hashlib

        from csvscrape import config, dataset, download
        from csvscrape.validate import git_blob_sha1

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        for module, name in [(config, "DATA_ROOT"), (download, "CORPUS"), (dataset, "ROOT")]:
            self.addCleanup(setattr, module, name, getattr(module, name))
        config.DATA_ROOT = root
        download.CORPUS = root / "corpus"
        dataset.ROOT = root / "datasets"
        dataset.ROOT.mkdir()

        here = b"a,b\n1,2\n" * 20
        gone = b"x;y\n3;4\n" * 20
        key = {data: git_blob_sha1(len(data), iter([data])) for data in (here, gone)}
        entry = {"repo": "o/r", "ref": "e" * 40, "lfs": False, "bucket": 7}
        (dataset.ROOT / "tiny.json").write_text(json.dumps({
            "dataset": {"name": "tiny", "source": "github"},
            "files": {
                key[here]: entry | {"path": "here.csv", "size": len(here)},
                key[gone]: entry | {"path": "gone.csv", "size": len(gone)},
            },
        }))

        class GH:
            def stream(self, url, sink, chunk=1 << 20):
                if url.endswith("gone.csv"):
                    return None
                sink.write(here)
                return len(here), hashlib.sha256(here).hexdigest()

        first = download.fetch(GH(), "tiny", workers=1)
        self.assertEqual((first["fetched"], first["reused"]), (1, 0))
        self.assertEqual([rec["content_key"] for rec in first["failed"]], [key[gone]])
        self.assertEqual(download._target(key[here]).read_bytes(), here)

        again = download.fetch(GH(), "tiny", workers=1)
        self.assertEqual((again["fetched"], again["reused"]), (0, 1))
