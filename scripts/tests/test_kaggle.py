"""Offline tests for the Kaggle pipeline
"""
from __future__ import annotations

import io
import json
import random
import shutil
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path


def build_zip(members: dict[str, bytes], force_zip64: bool = False,
              stored: tuple[str, ...] = ()) -> bytes:
    """A real archive, so the parser is tested against a real writer."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name, body in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = (zipfile.ZIP_STORED if name in stored
                                  else zipfile.ZIP_DEFLATED)
            if force_zip64:
                info.file_size = len(body)
            archive.writestr(info, body)
    return buf.getvalue()


class ZipDirectory(unittest.TestCase):
    """Locating and parsing a central directory out of raw bytes."""

    BODIES = {
        "a.csv": b"x,y\n1,2\n" * 40,
        # Incompressible on purpose: any structured body deflates to a few
        # hundred bytes, and then a "read less than the member" test is not
        # testing anything. Seeded, so the archive is identical every run.
        "nested/dir/b.tsv": random.Random(7).randbytes(40_000),
        "plain.csv": b"a,b\n" * 500,
        "notes.md": b"# ignore me\n",
    }

    def setUp(self):
        self.raw = build_zip(self.BODIES, stored=("plain.csv",))

    def _members(self):
        from csvscrape import zipindex

        directory = zipindex.find_directory(
            self.raw[-zipindex.EOCD_SEARCH:], len(self.raw))
        return directory, zipindex.parse_directory(
            self.raw[directory.offset:directory.offset + directory.size])

    def test_every_entry_is_found(self):
        directory, members = self._members()
        self.assertEqual(directory.entries, len(self.BODIES))
        self.assertEqual(len(members), len(self.BODIES))
        self.assertEqual({m.name for m in members}, set(self.BODIES))

    def test_crc_and_sizes_match_the_real_content(self):
        from csvscrape import zipindex

        _, members = self._members()
        for member in members:
            body = self.BODIES[member.name]
            self.assertEqual(member.usize, len(body), member.name)
            self.assertEqual(member.crc, zipindex.crc_of(body), member.name)

    def test_members_round_trip_through_their_offsets(self):
        """The whole download path in miniature: locate, fetch, inflate, verify."""
        from csvscrape import zipindex

        _, members = self._members()
        for member in members:
            start = zipindex.data_offset(
                self.raw[member.header_offset:member.header_offset + 1024],
                member.header_offset)
            blob = self.raw[start:start + member.csize]
            data = zipindex.inflate_all(blob, member.method, member.usize)
            self.assertEqual(data, self.BODIES[member.name], member.name)
            self.assertEqual(zipindex.crc_of(data), member.crc)

    def test_a_prefix_costs_only_a_prefix(self):
        """A head must not require the member's whole compressed stream; the
        probe's cost rests on that."""
        from csvscrape import zipindex

        _, members = self._members()
        member = next(m for m in members if m.name == "nested/dir/b.tsv")
        start = zipindex.data_offset(
            self.raw[member.header_offset:member.header_offset + 1024],
            member.header_offset)
        want = 512
        partial = self.raw[start:start + 1024]
        self.assertLess(len(partial), member.csize)
        head = zipindex.inflate_prefix(partial, member.method, want)
        self.assertTrue(head)
        self.assertTrue(self.BODIES[member.name].startswith(head))

    def test_a_stream_cut_short_returns_what_it_decoded(self):
        """A truncated range is expected, not exceptional -- the caller asks
        for less than the member on purpose, so this must not raise."""
        from csvscrape import zipindex

        _, members = self._members()
        member = next(m for m in members if m.name == "a.csv")
        start = zipindex.data_offset(
            self.raw[member.header_offset:member.header_offset + 1024],
            member.header_offset)
        head = zipindex.inflate_prefix(self.raw[start:start + 4],
                                       member.method, 4096)
        self.assertIsInstance(head, bytes)

    def test_stored_members_need_no_inflation(self):
        from csvscrape import zipindex

        _, members = self._members()
        member = next(m for m in members if m.name == "plain.csv")
        self.assertEqual(member.method, zipindex.STORED)
        start = zipindex.data_offset(
            self.raw[member.header_offset:member.header_offset + 1024],
            member.header_offset)
        head = zipindex.inflate_prefix(self.raw[start:start + 64],
                                       member.method, 16)
        self.assertEqual(head, self.BODIES["plain.csv"][:16])

    def test_local_extra_is_read_rather_than_assumed(self):
        """The local and central extra fields may differ in length.

        Reusing the central directory's lengths puts the read a few bytes off,
        which deflate reports as a corrupt stream rather than as a wrong
        answer -- so it would show up as a mass of unreadable members, not as
        an obviously wrong offset.
        """
        from csvscrape import zipindex

        _, members = self._members()
        member = next(m for m in members if m.name == "a.csv")
        header = self.raw[member.header_offset:member.header_offset + 30]
        local_name, local_extra = struct.unpack("<HH", header[26:30])
        computed = zipindex.data_offset(
            self.raw[member.header_offset:member.header_offset + 1024],
            member.header_offset)
        self.assertEqual(computed,
                         member.header_offset + 30 + local_name + local_extra)

    def test_not_a_zip_is_reported_rather_than_guessed(self):
        from csvscrape import zipindex

        with self.assertRaises(zipindex.BadZip):
            zipindex.find_directory(b"this is a plain csv\n" * 100, 2000)

    def test_inflate_all_enforces_the_declared_length(self):
        from csvscrape import zipindex

        _, members = self._members()
        member = next(m for m in members if m.name == "a.csv")
        start = zipindex.data_offset(
            self.raw[member.header_offset:member.header_offset + 1024],
            member.header_offset)
        with self.assertRaises(zipindex.BadZip):
            zipindex.inflate_all(self.raw[start:start + 8], member.method,
                                 member.usize)

    def test_unsupported_compression_is_refused(self):
        from csvscrape import zipindex

        with self.assertRaises(zipindex.BadZip):
            zipindex.inflate_all(b"\x00" * 32, 12, 32)


class Zip64(unittest.TestCase):
    """Kaggle bundles saturate the 32-bit fields routinely, so the zip64 path is
    the normal path."""

    def test_zip64_records_are_preferred_over_the_32_bit_ones(self):
        from csvscrape import zipindex

        raw = build_zip({"big.csv": b"a,b\n" * 1000}, force_zip64=True)
        directory = zipindex.find_directory(raw[-zipindex.EOCD_SEARCH:], len(raw))
        members = zipindex.parse_directory(
            raw[directory.offset:directory.offset + directory.size])
        self.assertEqual(len(members), 1)
        self.assertEqual(members[0].usize, 4000)
        self.assertEqual(members[0].crc, zipindex.crc_of(b"a,b\n" * 1000))

    def test_zip64_extra_fields_are_read_positionally(self):
        """Only the *saturated* fields are present in the extra block, in a
        fixed order. Taking the first value as the uncompressed size
        unconditionally is the classic way to get this wrong, and it yields a
        member read from the wrong offset in a large archive."""
        from csvscrape.zipindex import U32_MAX, _zip64_extra

        # Only the offset saturated: the block holds one value, and it is the
        # offset -- not the size.
        extra = struct.pack("<HHQ", 0x0001, 8, 1 << 40)
        usize, csize, offset = _zip64_extra(extra, 10, 20, U32_MAX)
        self.assertEqual((usize, csize, offset), (10, 20, 1 << 40))

        # Size and offset saturated: two values, in order.
        extra = struct.pack("<HHQQ", 0x0001, 16, 1 << 35, 1 << 40)
        usize, csize, offset = _zip64_extra(extra, U32_MAX, 20, U32_MAX)
        self.assertEqual((usize, csize, offset), (1 << 35, 20, 1 << 40))

    def test_saturated_fields_without_a_zip64_record_are_an_error(self):
        """Silently keeping 0xFFFFFFFF would index every large member at 4 GB."""
        from csvscrape import zipindex

        eocd = zipindex.EOCD + struct.pack("<HHHHIIH", 0, 0, 1, 1,
                                           0xFFFFFFFF, 0xFFFFFFFF, 0)
        with self.assertRaises(zipindex.BadZip):
            zipindex.find_directory(eocd, 1 << 40)


class SubdivisionTree(unittest.TestCase):
    """Discovery reaches the whole pool only if the tree walk is right."""

    def test_the_split_is_geometric(self):
        from csvscrape import kaggle

        # Sizes span nine orders of magnitude, so an arithmetic midpoint puts
        # essentially the whole population on one side and makes no progress.
        mid = kaggle.split_at(1_000, 1_000_000_000)
        self.assertLess(mid, 10_000_000)
        self.assertGreater(mid, 100_000)

    def test_the_split_always_makes_progress(self):
        from csvscrape import kaggle

        # hi - lo >= 2 is the precondition: below that there is no integer
        # strictly between the bounds, which is why both callers test for
        # an atomic shard before asking for a split.
        for lo, hi in ((1, 3), (5, 7), (998, 1_000), (1, 10 ** 13),
                       (1_000, 1_000_000), (10 ** 9, 10 ** 13)):
            mid = kaggle.split_at(lo, hi)
            self.assertGreater(mid, lo, (lo, hi))
            self.assertLess(mid, hi, (lo, hi))

    def test_an_unvisited_tree_offers_the_roots(self):
        from csvscrape import kaggle

        pending = list(kaggle.shards({}))
        self.assertEqual([(lo, hi) for lo, hi, _ in pending],
                         list(kaggle.ROOT_SHARDS))

    def test_a_capped_shard_is_replaced_by_its_halves_and_never_enumerated(self):
        """The saving the whole discovery budget rests on: an interior node
        costs one call to identify, not the 500 that enumerating it would."""
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[0]
        mid = kaggle.split_at(lo, hi)
        states = {kaggle._shard_key(lo, hi): {"state": "capped"}}
        pending = {(a, b) for a, b, _ in kaggle.shards(states)}
        self.assertIn((lo, mid), pending)
        self.assertIn((mid, hi), pending)
        self.assertNotIn((lo, hi), pending)

    def test_a_done_shard_disappears(self):
        from csvscrape import kaggle

        states = {kaggle._shard_key(lo, hi): {"state": "done"}
                  for lo, hi in kaggle.ROOT_SHARDS}
        self.assertEqual(list(kaggle.shards(states)), [])

    def test_an_atomic_shard_is_not_split_forever(self):
        """A one-byte-wide shard that is still capped cannot be subdivided.
        Without the guard the walk recurses on it indefinitely."""
        from csvscrape import kaggle

        states = {kaggle._shard_key(lo, hi): {"state": "done"}
                  for lo, hi in kaggle.ROOT_SHARDS}
        states[kaggle._shard_key(1, 2)] = {"state": "atomic"}
        self.assertEqual(list(kaggle.shards(states)), [])

    def test_a_partial_shard_resumes_at_its_page(self):
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[0]
        states = {kaggle._shard_key(lo, hi): {"state": "partial", "pages": 37}}
        resume = {(a, b): page for a, b, page in kaggle.shards(states)}
        self.assertEqual(resume[(lo, hi)], 37)

    def test_an_errored_shard_is_retried_from_where_it_stopped(self):
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[0]
        states = {kaggle._shard_key(lo, hi): {"state": "error", "pages": 4}}
        resume = {(a, b): page for a, b, page in kaggle.shards(states)}
        self.assertEqual(resume[(lo, hi)], 4)

    def test_the_tree_walk_terminates_on_a_fully_capped_spine(self):
        """Every node capped down one side must still bottom out at `atomic`
        rather than running the walk out of stack."""
        from csvscrape import kaggle

        lo, hi = 1, 1_000
        states = {}
        for _ in range(64):
            states[kaggle._shard_key(lo, hi)] = {"state": "capped"}
            if hi - lo <= 1:
                states[kaggle._shard_key(lo, hi)] = {"state": "atomic"}
                break
            hi = kaggle.split_at(lo, hi)
        pending = list(kaggle.shards(states))
        self.assertTrue(all(b - a >= 1 for a, b, _ in pending))

    def test_no_shard_ever_asks_for_min_size_zero(self):
        """`minSize=0` silently disables the size filter, so no shard may ask for
        it."""
        from csvscrape import kaggle

        self.assertTrue(all(lo >= 1 for lo, _ in kaggle.ROOT_SHARDS))
        self.assertEqual(kaggle.ROOT_SHARDS[0][0], 1)

    def test_the_roots_tile_the_size_axis_without_gaps(self):
        from csvscrape import kaggle

        for (_, hi), (lo, _) in zip(kaggle.ROOT_SHARDS, kaggle.ROOT_SHARDS[1:]):
            self.assertEqual(hi, lo)


class DiscoveryLoop(unittest.TestCase):
    """The shard loop terminates. It recomputes its frontier from the log on
    every iteration -- which is what lets a split enqueue its children at once
    -- and a shard that ends `error` or `partial` is still on that frontier
    afterwards, so termination is a property of the loop rather than of the
    data."""

    def _run(self, responses, **kwargs):
        """Drive `discover` against a scripted API."""
        from csvscrape import config, kaggle

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = root

        calls = []

        class Client:
            bucket = None

            def get(self, path, _redirect=False, **params):
                calls.append(params)
                if len(calls) > 4000:
                    raise AssertionError("discover did not terminate")
                return responses(params)

        stats = kaggle.discover(Client(), **kwargs)
        return stats, calls

    def test_a_failing_shard_does_not_spin_the_pass_forever(self):
        """An error leaves the shard on the frontier, so a loop that only
        consults the log picks it again immediately -- for ever, spending the
        whole rate budget on one shard that is never going to answer."""
        stats, calls = self._run(lambda p: {"error": 500, "message": "boom"})
        self.assertGreater(stats["failed"], 0)
        self.assertLess(len(calls), 4000)

    def test_a_page_budget_already_spent_does_not_spin(self):
        """A `partial` shard is also still on the frontier, and resuming it
        with no budget left cannot even advance a page -- the tightest possible
        infinite loop."""
        stats, calls = self._run(
            lambda p: [{"ref": f"o/d{p['page']}-{i}"} for i in range(20)],
            max_pages=2)
        self.assertLess(len(calls), 4000)
        self.assertGreater(stats["new"], 0)

    def test_a_short_page_budget_spends_no_call_on_the_capped_test(self):
        """Below the API's own cap a full last page means the budget ran out,
        not that the shard is too big -- so asking costs a call and buys
        nothing."""
        _, calls = self._run(lambda p: [] if p["page"] > 1 else
                             [{"ref": "o/d"}], max_pages=3)
        self.assertNotIn(3, [c["page"] for c in calls])

    def test_a_capped_shard_costs_one_call_not_five_hundred(self):
        """The whole discovery budget rests on this: an interior node is
        identified by a single probe at the last page and never enumerated."""
        from csvscrape import kaggle

        def responses(params):
            # Every shard wider than a hair looks capped.
            if params["maxSize"] - params["minSize"] > 4:
                return [{"ref": f"o/x{params['page']}"} for _ in range(20)]
            return []

        stats, calls = self._run(responses, max_shards=3)
        self.assertGreater(stats["split"], 0)
        # An interior node is identified and abandoned; it is never walked.
        # Any request for a middle page would mean one was being enumerated,
        # which at 500 pages a shard is the whole cost this avoids.
        walked = [c["page"] for c in calls if 1 < c["page"] < kaggle.MAX_PAGE]
        self.assertEqual(walked, [])
        # One probe per shard considered, and nothing else beyond the pages an
        # uncapped shard legitimately reads.
        probes = [c for c in calls if c["page"] == kaggle.MAX_PAGE]
        self.assertGreaterEqual(len(probes), stats["split"])
        self.assertLess(len(calls), 10)


class MemberSelection(unittest.TestCase):
    """The per-dataset cap, and what it is allowed to let through."""

    def _member(self, name, crc=1, usize=10):
        from csvscrape.zipindex import Member

        return Member(name=name, method=8, crc=crc, csize=5, usize=usize,
                      header_offset=0)

    def test_directories_and_junk_are_dropped(self):
        from csvscrape import kaggle

        members = [self._member("data/"), self._member("__MACOSX/a.csv"),
                   self._member("real.csv")]
        kept, n_probe, total = kaggle.select_members(members)
        self.assertEqual([m.name for m in kept], ["real.csv"])
        self.assertEqual((n_probe, total), (1, 1))

    def test_only_kept_extensions_survive(self):
        from csvscrape import kaggle

        members = [self._member("a.csv"), self._member("b.png"),
                   self._member("c.parquet"), self._member("d.tsv")]
        kept, _, _ = kaggle.select_members(members)
        self.assertEqual({m.name for m in kept}, {"a.csv", "d.tsv"})

    def test_unreadable_compression_is_skipped(self):
        """A member this process cannot inflate must not enter the index: it
        would be drawn, fetched and rejected, wasting the draw's slot."""
        from csvscrape import kaggle
        from csvscrape.zipindex import Member

        members = [Member("a.csv", 12, 1, 5, 10, 0),
                   Member("b.csv", 8, 1, 5, 10, 0)]
        kept, _, _ = kaggle.select_members(members)
        self.assertEqual([m.name for m in kept], ["b.csv"])

    def test_the_cap_reports_what_it_discarded(self):
        from csvscrape import kaggle

        members = [self._member(f"f{i}.csv") for i in range(250)]
        kept, n_probe, total = kaggle.select_members(members, cap=100)
        self.assertEqual(len(kept), 250)      # all recorded
        self.assertEqual(n_probe, 100)        # only these probed
        self.assertEqual(total, 250)

    def test_selection_is_deterministic_across_runs(self):
        from csvscrape import kaggle

        members = [self._member(f"f{i}.csv") for i in range(250)]
        first, _, _ = kaggle.select_members(members, cap=10, record=10)
        second, _, _ = kaggle.select_members(list(reversed(members)), cap=10,
                                             record=10)
        self.assertEqual([m.name for m in first], [m.name for m in second])

    def test_selection_is_not_the_directory_order(self):
        """Central directory entries come out grouped by directory, so the
        first N of a large dataset would be one subtree -- one year, one
        sensor, one language -- and every file in it shares a producer."""
        from csvscrape import kaggle

        members = ([self._member(f"aaa/{i}.csv") for i in range(200)]
                   + [self._member(f"zzz/{i}.csv") for i in range(200)])
        kept, n_probe, _ = kaggle.select_members(members, cap=100, record=100)
        prefixes = {m.name.split("/")[0] for m in kept[:n_probe]}
        self.assertEqual(prefixes, {"aaa", "zzz"})

    def test_a_small_dataset_is_taken_whole(self):
        from csvscrape import kaggle

        members = [self._member("b.csv"), self._member("a.csv")]
        kept, n_probe, total = kaggle.select_members(members, cap=100)
        self.assertEqual({m.name for m in kept}, {"b.csv", "a.csv"})
        self.assertEqual((n_probe, total), (2, 2))

    def test_the_probed_set_is_a_stable_prefix_of_the_recorded_one(self):
        """What makes the probe cap revisitable: raising it extends the prefix
        rather than reshuffling it, so nothing already described is re-fetched
        and no verdict is orphaned."""
        from csvscrape import kaggle

        members = [self._member(f"f{i}.csv") for i in range(900)]
        recorded, probe_100, _ = kaggle.select_members(members, cap=100,
                                                       record=500)
        again, probe_250, _ = kaggle.select_members(members, cap=250,
                                                    record=500)
        self.assertEqual([m.name for m in recorded], [m.name for m in again])
        self.assertEqual((probe_100, probe_250), (100, 250))
        # The wider probe set contains the narrower one, in the same order.
        self.assertEqual([m.name for m in recorded[:100]],
                         [m.name for m in again[:probe_250]][:100])

    def test_recording_goes_wider_than_probing(self):
        """Raising --per-dataset later costs head fetches, not re-reading every
        central directory."""
        from csvscrape import kaggle

        members = [self._member(f"f{i}.csv") for i in range(9_000)]
        recorded, n_probe, n_kept = kaggle.select_members(members)
        self.assertEqual(len(recorded), kaggle.RECORD_MEMBERS)
        self.assertEqual(n_probe, kaggle.MAX_FILES_PER_DATASET)
        self.assertEqual(n_kept, 9_000)
        self.assertLess(n_probe, len(recorded))


class DirectoryReads(unittest.TestCase):
    """How many requests a dataset's file list costs.

    This is the unmetered half of the pipeline, so a wasted read does not show
    up as a rate-limit error -- it shows up only as the pass taking longer,
    which is indistinguishable from the pass being big. Counting the reads is
    the only way the cost stays honest.
    """

    class Counting:
        """A client over an in-memory archive that counts what it fetches."""

        def __init__(self, raw):
            self.raw = raw
            self.reads = []

        def range_get(self, url, start, end=None, attempts=4):
            self.reads.append((start, end))
            if start is None:
                return self.raw[-end:], len(self.raw)
            return (self.raw[start:(None if end is None else end + 1)],
                    len(self.raw))

        def read_range(self, url, start, end=None, attempts=4):
            got = self.range_get(url, start, end, attempts)
            return None if got is None else got[0]

    def test_a_small_archive_costs_one_read(self):
        """Most datasets are small, so the central directory is usually inside the
        tail already fetched."""
        from csvscrape import kaggle

        client = self.Counting(build_zip({"a.csv": b"x,y\n1,2\n"}))
        members, declared, total = kaggle.read_directory(client, "url")
        self.assertEqual(len(client.reads), 1)
        self.assertEqual(declared, 1)
        self.assertEqual([m.name for m in members], ["a.csv"])


    def test_a_directory_past_the_tail_costs_a_second_read(self):
        """A large archive's directory does not fit the tail window and costs a
        second read."""
        from csvscrape import kaggle, zipindex

        # Enough entries that the central directory exceeds the tail window.
        bodies = {f"d{i:05d}/file{i:05d}.csv": b"a,b\n" for i in range(4000)}
        client = self.Counting(build_zip(bodies))
        members, declared, _ = kaggle.read_directory(client, "url")
        directory_size = sum(46 + len(n) for n in bodies)
        self.assertGreater(directory_size, zipindex.EOCD_SEARCH)
        self.assertEqual(len(client.reads), 2)
        self.assertEqual(declared, 4000)
        self.assertEqual(len(members), 4000)

    def test_the_sliced_directory_equals_the_separately_fetched_one(self):
        """The in-tail shortcut must return exactly what the extra read would.
        An off-by-one here silently truncates the last member of every small
        dataset -- the majority of the pool."""
        from csvscrape import kaggle, zipindex

        raw = build_zip({f"f{i}.csv": b"a,b\n1,2\n" for i in range(12)})
        sliced, _, _ = kaggle.read_directory(self.Counting(raw), "url")
        directory = zipindex.find_directory(raw[-zipindex.EOCD_SEARCH:], len(raw))
        whole = zipindex.parse_directory(
            raw[directory.offset:directory.offset + directory.size])
        self.assertEqual(sliced, whole)

    def test_an_archive_shorter_than_the_tail_window_still_works(self):
        """A suffix range longer than the object returns the whole object, so
        the tail's start is 0 and the offsets must still resolve."""
        from csvscrape import kaggle, zipindex

        raw = build_zip({"a.csv": b"x\n"})
        self.assertLess(len(raw), zipindex.EOCD_SEARCH)
        members, _, total = kaggle.read_directory(self.Counting(raw), "url")
        self.assertEqual(total, len(raw))
        self.assertEqual([m.name for m in members], ["a.csv"])


class ContentIdentity(unittest.TestCase):
    """CRC32 plus length is the only identity Kaggle exposes before a fetch."""

    def test_identical_bytes_always_agree(self):
        """The property that makes a second hash pass pointless: a checksum
        cannot miss a duplicate, only invent one."""
        from csvscrape import kaggle, zipindex

        body = b"a,b\n1,2\n" * 77
        first = kaggle.content_key(zipindex.crc_of(body), len(body))
        second = kaggle.content_key(zipindex.crc_of(bytes(body)), len(body))
        self.assertEqual(first, second)

    def test_the_key_is_a_flat_filename(self):
        """It becomes a path in the content-addressed store, so it must not
        carry a separator or the store grows a directory per dataset."""
        from csvscrape import kaggle

        key = kaggle.content_key(0xdeadbeef, 123456)
        self.assertNotIn("/", key)
        self.assertNotIn("\\", key)
        self.assertTrue(key.startswith("deadbeef"))

    def test_the_key_cannot_be_confused_with_a_git_blob_sha(self):
        """Both sources share one store, so the two key spaces must not
        overlap -- a collision would have one source serve the other's bytes."""
        from csvscrape import kaggle

        key = kaggle.content_key(0xdeadbeef, 123456)
        self.assertIn("-", key)
        self.assertNotEqual(len(key), 40)


    def test_download_recomputes_the_same_key_off_the_disk(self):
        """`_already_here` reads a file back by its key, so the identity
        computed from bytes on disk has to equal the one the index recorded."""
        from csvscrape import download, kaggle, zipindex

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        body = b"a,b\n1,2\n" * 500
        path = root / "f"
        path.write_bytes(body)
        expected = kaggle.content_key(zipindex.crc_of(body), len(body))
        self.assertEqual(
            download.content_id(path, len(body), False, "kaggle"), expected)

    def test_a_multi_chunk_file_hashes_the_same_as_a_single_read(self):
        """CRC is computed incrementally over 1 MiB chunks; a running seed
        passed wrong gives a different answer only for files above the chunk
        size, which is exactly where nothing small would catch it."""
        from csvscrape import download, kaggle, zipindex

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        body = bytes(range(256)) * 20_000        # > 1 MiB
        path = root / "f"
        path.write_bytes(body)
        self.assertEqual(
            download.content_id(path, len(body), False, "kaggle"),
            kaggle.content_key(zipindex.crc_of(body), len(body)))


class ThrottleDetection(unittest.TestCase):
    """A 404 from `/datasets/download` is often the edge's penalty box (HTML)
    rather than the API's answer (JSON); only the latter may retire a
    dataset."""

    class Resp:
        def __init__(self, body=b"", ctype=""):
            self.content = body
            self.headers = {"content-type": ctype} if ctype else {}

    def test_an_html_404_is_recognised_as_a_throttle(self):
        from csvscrape import kaggle

        page = b"<!doctype html><meta charset=utf-8><title>404</title>"
        self.assertTrue(kaggle._is_html(self.Resp(page)))
        self.assertTrue(kaggle._is_html(self.Resp(b"", "text/html; charset=utf-8")))

    def test_a_json_404_is_a_real_answer(self):
        from csvscrape import kaggle

        self.assertFalse(kaggle._is_html(
            self.Resp(b'{"message":"not found"}', "application/json")))
        self.assertFalse(kaggle._is_html(self.Resp(b"")))


class MeteredPacing(unittest.TestCase):
    """The metered pool: congestion events, refusals, and one budget per signing
    route."""

    class Resp:
        def __init__(self, status=200, headers=None, body=b"{}"):
            self.status_code = status
            self.headers = headers or {}
            self.content = body

        def json(self):
            return json.loads(self.content)

    def _client(self, script, rate=1000.0, capacity=None):
        """A `Kaggle` whose session is scripted and whose buckets are fast.

        The pacing rate under test is asserted from the constant rather than
        by sitting still for it; a test that waits out the real interval is
        testing `Bucket`, which has its own. Every call the session sees is
        kept in `kg.sent`, as (method, url, params or body).
        """
        from csvscrape import kaggle
        from csvscrape.client import Bucket

        events: list[tuple[str, dict]] = []
        kg = kaggle.Kaggle(tok="test",
                           on_event=lambda kind, **kw: events.append((kind, kw)))
        kg.buckets = {name: Bucket(1.0, rate, capacity=capacity or rate,
                                   on_event=kg.on_event, name=f"kaggle_{name}")
                      for name in kg.buckets}
        sent = kg.sent = []

        class Session:
            headers: dict = {}

            def get(self, url, params=None, timeout=None, allow_redirects=True):
                sent.append(("get", url, params))
                return script()

            def post(self, url, json=None, timeout=None, allow_redirects=True):
                sent.append(("post", url, json))
                return script()

        kg.session = Session()
        return kg, events

    def _holds(self, events):
        return [kw for kind, kw in events
                if kind == "wait" and kw.get("throttled")]

    def test_one_congestion_event_is_announced_once_however_many_it_refuses(self):
        kg, events = self._client(
            lambda: self.Resp(429, {"retry-after": "0.01"}, b""))
        kg.get("/datasets/list")
        self.assertEqual(len(self._holds(events)), 1)

    def test_every_refused_call_is_still_counted(self):
        """The announcement is deduplicated; the cost is not. One event that
        refuses six calls is one line and six rejected requests, and the
        summary needs both numbers to say what the event cost."""
        kg, events = self._client(
            lambda: self.Resp(429, {"retry-after": "0.01"}, b""))
        kg.get("/datasets/list")
        self.assertEqual(kg.stats["kaggle_list_throttled"], 6)

    def test_a_retry_after_is_floored_at_the_pacing_interval(self):
        """A retry inside the throttle cannot succeed and is still charged to
        the budget being waited for -- the floor the metered GitHub path
        already applies."""
        answers = [self.Resp(429, {"retry-after": "0.001"}, b""), self.Resp()]
        kg, events = self._client(lambda: answers.pop(0), rate=4.0)
        kg.get("/datasets/list")
        self.assertAlmostEqual(self._holds(events)[0]["seconds"],
                               kg.buckets["list"].min_interval, places=6)







    def _redirect(self):
        return self.Resp(302, {"location": "https://storage.example/a.zip"},
                         b"")

    def test_each_route_is_charged_to_its_own_budget(self):
        """The REST and RPC downloads are metered separately; a shared bucket
        would pace the pair at the rate of one."""
        kg, _ = self._client(self._redirect)
        kg.signed_url("owner/slug", route="rest")
        kg.signed_url("owner/slug", route="rpc")
        kg.signed_url("owner/slug", route="rpc")
        self.assertEqual(kg.stats["kaggle_rest_requests"], 1)
        self.assertEqual(kg.stats["kaggle_rpc_requests"], 2)
        self.assertEqual(kg.stats["kaggle_list_requests"], 0)

    def test_both_routes_sign_the_same_request(self):
        """Same dataset, same pin, whichever route asks."""
        from csvscrape import kaggle

        kg, _ = self._client(self._redirect)
        self.assertEqual(kg.signed_url("owner/slug", 3, route="rest"),
                         ("https://storage.example/a.zip", 302))
        self.assertEqual(kg.signed_url("owner/slug", 3, route="rpc"),
                         ("https://storage.example/a.zip", 302))
        rest, rpc = kg.sent
        self.assertEqual(rest, ("get", kaggle.API + "/datasets/download/owner/slug",
                                {"datasetVersionNumber": 3}))
        self.assertEqual(rpc, ("post", kaggle.RPC + "/DownloadDataset",
                               {"ownerSlug": "owner", "datasetSlug": "slug",
                                "datasetVersionNumber": 3}))

    def test_a_refusal_holds_only_the_route_it_came_from(self):
        """A 429 on one route says the budget *it* is charged to is spent,
        and nothing about the other. Holding both would throw away half the
        pass's throughput every time either one was refused."""
        kg, _ = self._client(self._redirect)
        kg.buckets["rest"].hold_for(60.0)
        import time

        started = time.monotonic()
        kg.signed_url("owner/slug", route="rpc")
        self.assertLess(time.monotonic() - started, 5.0)


    def test_the_rpc_routes_missing_dataset_is_terminal(self):
        """Both routes answer a missing or private dataset with a JSON 403 --
        the API's own answer, not a throttle, so it is not retried."""
        kg, _ = self._client(lambda: self.Resp(
            403, {"content-type": "application/json"},
            b'{"error":{"code":403,"status":"PERMISSION_DENIED"}}'))
        self.assertEqual(kg.signed_url("owner/gone", route="rpc"), (None, 403))
        self.assertEqual(kg.stats["kaggle_rpc_requests"], 1)


    def test_an_unknown_route_is_refused_before_anything_runs(self):
        from csvscrape import kaggle

        with self.assertRaises(ValueError):
            kaggle.bundle(None, routes=("rest", "www"))
        with self.assertRaises(ValueError):
            kaggle.bundle(None, routes=())


class IndexBuild(unittest.TestCase):
    """The Kaggle index must carry the GitHub index's columns, or every
    selector and both draw shapes stop working over it."""

    def _root(self):
        from csvscrape import config, index

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        self.addCleanup(setattr, index, "KAGGLE_INDEX", index.KAGGLE_INDEX)
        self.addCleanup(setattr, index, "KAGGLE_DATASETS", index.KAGGLE_DATASETS)
        config.DATA_ROOT = root
        index.KAGGLE_INDEX = root / "index" / "kaggle_files.parquet"
        index.KAGGLE_DATASETS = root / "index" / "kaggle_datasets.parquet"
        return root

    def _write_bundles(self, root, records):
        from csvscrape import kaggle, log

        with log.Writer(kaggle.BUNDLES_LOG) as writer:
            for record in records:
                writer.write(record)

    def _record(self, ref, files, version=3):
        """One bundle record; `files` is [(name, body)]."""
        from csvscrape import kaggle, zipindex

        names, keys, sizes, crcs = [], [], [], []
        for name, body in files:
            crc = zipindex.crc_of(body)
            names.append(name)
            keys.append(kaggle.content_key(crc, len(body)))
            sizes.append(len(body))
            crcs.append(crc)
        return {
            "ref": ref, "owner": ref.split("/")[0], "state": "ok",
            "version": version, "license": "CC0", "total_bytes": 999,
            "zip_bytes": 500, "n_entries": len(files), "n_kept": len(files),
            "n_recorded": len(files), "n_probed": len(files),
            "capped": False,
            "signed_at": 1, "ts": 1,
            "names": names, "keys": keys, "sizes": sizes, "crcs": crcs,
            "csizes": [10] * len(files), "methods": [8] * len(files),
            "offsets": list(range(len(files))),
        }

    def test_the_index_carries_the_columns_a_draw_needs(self):
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [self._record("o/d", [("a.csv", b"x,y\n1,2\n")])])
        index.build_kaggle()
        names = set(index.kaggle_files().collect_schema().names())
        for column in ("content_key", "full_name", "owner", "path", "size",
                       "bucket", "is_lfs", "lfs_suspect", "ext", "occurrences",
                       "n_repos", "repo_id", "ref", "truncated", "tier"):
            self.assertIn(column, names, column)

    def test_every_selector_runs_over_an_unprobed_kaggle_pool(self):
        """An index built before anything was probed has no probe columns at
        all -- the same as the GitHub side -- so a selector naming one would
        fail outright rather than selecting nothing. `sample.candidates`
        supplies them as typed nulls, and this is the contract that says so.
        """
        from csvscrape import index, sample, selectors

        root = self._root()
        self._write_bundles(root, [self._record("o/d", [("a.csv", b"x,y\n1,2\n")])])
        index.build_kaggle()
        for name in sorted(selectors.REGISTRY):
            pool = sample.candidates(select=name, source="kaggle")
            # Collected, not just planned: a missing column raises here.
            pool.collect()

    def test_probe_verdicts_fold_into_the_kaggle_index(self):
        """The heads log the bundle pass writes has to reach the index, or
        every dialect selector draws from an empty pool forever."""
        from csvscrape import index, kaggle, log

        root = self._root()
        body = b"a,b\n1,2\n"
        record = self._record("o/d", [("a.csv", body)])
        self._write_bundles(root, [record])
        with log.Writer(kaggle.HEADS_LOG) as writer:
            writer.write({
                "content_key": record["keys"][0], "bucket": 3,
                "size": len(body), "ts": 2, "fetched": True, "http": 206,
                "via": "kaggle", "head_len": len(body),
                "is_lfs_pointer": False, "looks_like_html": False,
                "agreement": 3,
                "custom": {"delimiter": ",", "encoding": "utf-8"},
            })
        index.build_kaggle()
        row = index.kaggle_files().collect().to_dicts()[0]
        self.assertTrue(row["fetched"])
        self.assertEqual(row["delimiter"], ",")
        self.assertEqual(row["via"], "kaggle")
        # The Kaggle index records its own watermark, not the GitHub one.
        self.assertEqual(index.heads_watermark("kaggle"), "00000:00000000")

    def test_identical_content_in_two_datasets_collapses_to_one_row(self):
        """Republication is the Kaggle pool's defining property -- iris and
        titanic appear thousands of times -- and a pool that did not collapse
        them would fill a log-uniform draw's small buckets with copies."""
        from csvscrape import index

        body = b"a,b\n1,2\n"
        root = self._root()
        self._write_bundles(root, [
            self._record("one/d", [("iris.csv", body)]),
            self._record("two/d", [("iris_copy.csv", body)]),
        ])
        index.build_kaggle()
        rows = index.kaggle_files().collect().to_dicts()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["occurrences"], 2)
        self.assertEqual(rows[0]["n_repos"], 2)

    def test_distinct_content_stays_distinct(self):
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [
            self._record("one/d", [("a.csv", b"a,b\n1,2\n"),
                                   ("b.csv", b"c,d\n3,4\n")]),
        ])
        index.build_kaggle()
        self.assertEqual(index.kaggle_files().collect().height, 2)

    def test_only_index_extensions_reach_the_index(self):
        """Same content as a .txt and a .csv keeps the .csv row."""
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [
            self._record("one/d", [("labels.txt", b"0 0.5 0.5 0.1 0.1\n"),
                                   ("a.dat", b"1 2\n"), ("same.txt", b"a,b\n")]),
            self._record("two/d", [("same.csv", b"a,b\n")]),
        ])
        index.build_kaggle()
        self.assertEqual(index.kaggle_files().select("path").collect()["path"].to_list(),
                         ["same.csv"])

    def test_the_archive_coordinates_survive_into_the_index(self):
        """Without them a drawn row cannot be fetched at all."""
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [self._record("o/d", [("a.csv", b"x,y\n1,2\n")])])
        index.build_kaggle()
        row = index.kaggle_files().collect().to_dicts()[0]
        for column in ("crc", "csize", "method", "header_offset", "version"):
            self.assertIsNotNone(row[column], column)

    def test_the_version_pin_is_recorded(self):
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [
            self._record("o/d", [("a.csv", b"x,y\n1,2\n")], version=7)])
        index.build_kaggle()
        row = index.kaggle_files().collect().to_dicts()[0]
        self.assertEqual(row["version"], 7)
        self.assertEqual(row["ref"], "7")

    def test_buckets_agree_with_the_github_definition(self):
        """`download` compares an as-downloaded size against the indexed
        bucket, so two definitions that drifted would mis-stratify the corpus."""
        from csvscrape import index

        root = self._root()
        bodies = [("f%d.csv" % i, b"x" * (2 ** i + 1)) for i in range(4, 12)]
        self._write_bundles(root, [self._record("o/d", bodies)])
        index.build_kaggle()
        for row in index.kaggle_files().collect().to_dicts():
            self.assertEqual(row["bucket"], index.size_bucket(row["size"]),
                             row["path"])

    def test_an_empty_file_does_not_take_the_whole_build_down(self):
        """log2(0) is -inf and the cast to Int32 raises, failing the entire
        sink rather than mis-bucketing one row. `INDEX_MIN_BYTES` is 0 on
        purpose, so this is reachable on any real pool -- and it was, on the
        first live compact."""
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [
            self._record("o/d", [("empty.csv", b""),
                                 ("one.csv", b"\n"),
                                 ("real.csv", b"a,b\n1,2\n")])])
        index.build_kaggle()
        rows = {r["path"]: r for r in index.kaggle_files().collect().to_dicts()}
        self.assertEqual(rows["empty.csv"]["bucket"], 0)
        self.assertEqual(rows["empty.csv"]["size"], 0)
        self.assertEqual(rows["one.csv"]["bucket"], 0)

    def test_pool_for_routes_by_name(self):
        from csvscrape import index

        root = self._root()
        self._write_bundles(root, [self._record("o/d", [("a.csv", b"x,y\n1,2\n")])])
        index.build_kaggle()
        self.assertEqual(index.pool_for("kaggle").collect().height, 1)
        with self.assertRaises(SystemExit):
            index.pool_for("nope")

    def test_failed_bundles_never_reach_the_index(self):
        from csvscrape import index, kaggle, log

        self._root()
        good = self._record("o/good", [("a.csv", b"x,y\n1,2\n")])
        with log.Writer(kaggle.BUNDLES_LOG) as writer:
            writer.write(good)
            writer.write({"ref": "o/gone", "state": "gone", "http": 404, "ts": 1})
            writer.write({"ref": "o/raw", "state": "no_bundle", "ts": 1})
        index.build_kaggle()
        refs = set(index.kaggle_files().collect()["full_name"])
        self.assertEqual(refs, {"o/good"})

    def test_a_file_is_in_the_probe_cap_if_any_copy_of_it_is(self):
        """The head joins on content, so one copy inside some dataset's probe
        prefix is enough for every copy to have been queued."""
        from csvscrape import index

        root = self._root()
        shared = b"a,b\n1,2\n"
        big = self._record("o/big", [("p.csv", b"p\n1\n"), ("q.csv", shared),
                                     ("r.csv", b"r\n1\n")])
        big["n_probed"] = 1
        self._write_bundles(root, [big,
                                   self._record("o/small", [("s.csv", shared)])])
        index.build_kaggle()
        rows = {r["path"]: r["in_probe_cap"]
                for r in index.kaggle_files().collect().to_dicts()}
        self.assertEqual(rows, {"p.csv": True, "q.csv": True, "r.csv": False})



class OnlyTheApiRetiresADataset(unittest.TestCase):
    """A recorded dataset is never asked about again, so only an answer from the
    API may be recorded. A throttle that outlasted every retry, or a failed
    range read, says nothing about the dataset."""

    def _root(self):
        from csvscrape import config

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = root
        return root

    class Client:
        """Signs from a script, and serves one in-memory archive."""

        ranges = range_bytes = 0

        def __init__(self, answers, raw):
            self.answers, self.raw = answers, raw

        def signed_url(self, ref, version=None, route="rest"):
            return self.answers[ref]

        def range_get(self, url, start, end=None, attempts=4):
            if url == "unreadable":
                return None
            if start is None:
                return self.raw[-end:], len(self.raw)
            return (self.raw[start:(None if end is None else end + 1)],
                    len(self.raw))

        def read_range(self, url, start, end=None, attempts=4):
            got = self.range_get(url, start, end, attempts)
            return None if got is None else got[0]

    def _pass(self, answers):
        from csvscrape import kaggle, log

        self._root()
        with log.Writer(kaggle.DATASETS_LOG) as writer:
            for ref in answers:
                writer.write({"ref": ref, "version": 1, "total_bytes": 10,
                              "size_hint": 10})
        client = self.Client(answers, build_zip({"a.csv": b"x,y\n1,2\n"}))
        return kaggle.bundle(client, routes=("rest",), head_workers=1,
                             sniff_procs=0)

    def test_only_an_answer_from_the_api_is_recorded(self):
        from csvscrape import kaggle

        stats = self._pass({
            "o/ok": ("good", 302),
            "o/private": (None, 403),
            "o/throttled": (None, -1),
            "o/flaky": ("unreadable", 302),
        })
        self.assertEqual(kaggle.bundled(), {"o/ok", "o/private"})
        self.assertEqual(stats["deferred"], 2)
        self.assertEqual({r["ref"] for r in kaggle.pending_datasets()},
                         {"o/throttled", "o/flaky"})

    def test_records_that_retired_a_healthy_dataset_are_passed_over(self):
        """Older logs hold records for non-answers (a throttle, a failed read);
        those must not keep a dataset retired."""
        from csvscrape import kaggle

        self.assertFalse(kaggle.settled({"state": "gone", "http": -1}))
        self.assertFalse(kaggle.settled({"state": "gone", "http": 0}))
        self.assertFalse(kaggle.settled(
            {"state": "no_bundle", "message": "could not read the archive tail"}))
        self.assertTrue(kaggle.settled({"state": "gone", "http": 403}))
        self.assertTrue(kaggle.settled({"state": "gone", "http": 404}))
        self.assertTrue(kaggle.settled(
            {"state": "no_bundle", "message": "no end of central directory"}))
        self.assertTrue(kaggle.settled({"state": "ok", "pinned": True}))

    def test_an_unpinned_read_of_another_archive_is_read_again(self):
        """An unpinned `ok` whose archive length differs from the listed
        `total_bytes` read a different version than it names, and is
        read again."""
        from csvscrape import kaggle

        self.assertTrue(kaggle.settled(
            {"state": "ok", "zip_bytes": 10, "total_bytes": 10}))
        self.assertFalse(kaggle.settled(
            {"state": "ok", "zip_bytes": 10, "total_bytes": 12}))
        # Once pinned, a differing length is no longer evidence of anything --
        # the listed figure disagrees on some very large archives -- so the
        # re-read cannot loop.
        self.assertTrue(kaggle.settled(
            {"state": "ok", "zip_bytes": 10, "total_bytes": 12, "pinned": True}))
        self.assertFalse(kaggle.settled({"state": "drifted", "http": 302}))

    def test_a_failed_read_is_not_a_verdict_on_the_archive(self):
        from csvscrape import kaggle, zipindex

        client = self.Client({}, b"")
        with self.assertRaises(kaggle.Unreadable):
            kaggle.read_directory(client, "unreadable")
        with self.assertRaises(zipindex.BadZip):
            kaggle.read_directory(self.Client({}, b"not a zip at all" * 10),
                                  "good")


class RevisitingBundledDatasets(unittest.TestCase):
    """A bundled dataset is never queued by `pending_datasets` again, so the
    heads it still owes -- past a cap since raised, or failed -- are reached
    only by `revisits`, at the coordinates its bundle record already holds."""

    FILES = {"a.csv": b"a\n1\n", "b.csv": b"b\n2\n", "c.csv": b"c\n3\n",
             "d.csv": b"d\n4\n"}

    def setUp(self):
        from csvscrape import config, index

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        for module, name in ((config, "DATA_ROOT"), (index, "KAGGLE_INDEX"),
                             (index, "KAGGLE_DATASETS")):
            self.addCleanup(setattr, module, name, getattr(module, name))
        config.DATA_ROOT = root
        index.KAGGLE_INDEX = root / "index" / "kaggle_files.parquet"
        index.KAGGLE_DATASETS = root / "index" / "kaggle_datasets.parquet"

    class Client(OnlyTheApiRetiresADataset.Client):
        """Serves one archive per ref, and remembers every signing call."""

        heads_fail = False

        def __init__(self, archives):
            self.archives, self.signed = archives, []

        def signed_url(self, ref, version=None, route="rest"):
            self.signed.append((ref, version))
            return (ref, 302) if ref in self.archives else (None, 404)

        def range_get(self, url, start, end=None, attempts=4):
            if start is not None and self.heads_fail:
                return None
            self.raw = self.archives[url]
            return super().range_get(url, start, end, attempts)

    def _datasets(self, *refs):
        from csvscrape import kaggle, log

        with log.Writer(kaggle.DATASETS_LOG) as writer:
            for ref in refs:
                writer.write({"ref": ref, "version": 7, "total_bytes": 10,
                              "size_hint": 10})

    def _pass(self, client, cap):
        from csvscrape import kaggle

        return kaggle.bundle(client, routes=("rest",), head_workers=1,
                             sniff_procs=0, cap=cap)

    def _described(self):
        from csvscrape import kaggle, log

        return {r["content_key"] for r in log.read(kaggle.HEADS_LOG)
                if r.get("fetched")}

    def test_raising_the_cap_reaches_the_recorded_members(self):
        from csvscrape import index, kaggle

        self._datasets("o/d")
        client = self.Client({"o/d": build_zip(self.FILES)})
        self._pass(client, cap=1)
        self.assertEqual(len(self._described()), 1)
        # The first read is pinned too, to discovery's version.
        self.assertEqual(client.signed, [("o/d", 7)])

        client.signed.clear()
        stats = self._pass(client, cap=3)
        self.assertEqual(len(self._described()), 3)
        self.assertEqual(stats["revisited"], 1)
        # Pinned, because the offsets belong to the archive of that version.
        self.assertEqual(client.signed, [("o/d", 7)])

        index.build_kaggle()
        rows = index.kaggle_files().collect()
        self.assertEqual(int(rows["in_probe_cap"].sum()), 3)
        sets = index.pl.read_parquet(index.KAGGLE_DATASETS)
        self.assertEqual(sets["n_probed"].to_list(), [3])

        client.signed.clear()
        self._pass(client, cap=3)
        self.assertEqual(client.signed, [])
        self.assertEqual(kaggle.revisits(3), [])

    def test_a_failed_head_is_fetched_again(self):
        self._datasets("o/d")
        client = self.Client({"o/d": build_zip(self.FILES)})
        client.heads_fail = True
        self._pass(client, cap=2)
        self.assertEqual(self._described(), set())

        client.heads_fail = False
        stats = self._pass(client, cap=2)
        self.assertEqual(len(self._described()), 2)
        self.assertEqual(stats["probing"], 2)

    def test_an_archive_that_changed_is_not_read_at_the_old_offsets(self):
        from csvscrape import kaggle

        self._datasets("o/d")
        client = self.Client({"o/d": build_zip(self.FILES)})
        self._pass(client, cap=1)
        client.archives["o/d"] = build_zip(self.FILES | {"e.csv": b"e\n5\n"})
        stats = self._pass(client, cap=4)
        self.assertEqual(stats["drifted"], 1)
        self.assertEqual(len(self._described()), 1)
        self.assertEqual(kaggle.revisits(4), [])

    def test_a_record_read_before_pinning_is_read_again_once(self):
        from csvscrape import index, kaggle, log

        self._datasets("o/d")
        archive = build_zip(self.FILES)
        with log.Writer(kaggle.BUNDLES_LOG) as writer:
            writer.write({"ref": "o/d", "owner": "o", "license": "CC0",
                          "state": "ok", "version": 7, "ts": 1,
                          "zip_bytes": len(archive) + 1, "total_bytes": 10,
                          "n_entries": 1, "n_kept": 1, "n_recorded": 1,
                          "n_probed": 1, "capped": False, "names": ["z.csv"],
                          "keys": ["0-1"], "sizes": [1], "crcs": [0],
                          "csizes": [1], "methods": [0], "offsets": [0]})
        self.assertEqual(kaggle.revisits(1), [])
        self.assertEqual([r["ref"] for r in kaggle.pending_datasets()], ["o/d"])

        client = self.Client({"o/d": archive})
        self._pass(client, cap=4)
        self.assertEqual(client.signed, [("o/d", 7)])
        index.build_kaggle()
        paths = set(index.kaggle_files().collect()["path"])
        self.assertEqual(paths, set(self.FILES))

        client.signed.clear()
        self._pass(client, cap=4)
        self.assertEqual(client.signed, [])

    def test_content_owed_by_two_datasets_is_fetched_once(self):
        from csvscrape import kaggle

        self._datasets("o/big", "o/small")
        shared = {"a.csv": self.FILES["a.csv"]}
        client = self.Client({"o/big": build_zip(self.FILES),
                              "o/small": build_zip(shared)})
        client.heads_fail = True
        self._pass(client, cap=4)
        owed = kaggle.revisits(4)
        self.assertEqual([item["ref"] for item in owed], ["o/big"])
        self.assertEqual(len(owed[0]["members"]), 4)


class KaggleDownload(unittest.TestCase):
    """Fetching a member, and refusing one whose dataset moved on."""

    def _bed(self):
        from csvscrape import config, download

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        self.addCleanup(setattr, download, "CORPUS", download.CORPUS)
        config.DATA_ROOT = root
        download.CORPUS = root / "corpus"
        (root / "corpus").mkdir(parents=True, exist_ok=True)
        return root

    def _archive_and_row(self, body=b"a,b\n1,2\n" * 200, name="a.csv"):
        from csvscrape import index, kaggle, zipindex

        raw = build_zip({name: body})
        directory = zipindex.find_directory(raw[-zipindex.EOCD_SEARCH:], len(raw))
        member = zipindex.parse_directory(
            raw[directory.offset:directory.offset + directory.size])[0]
        row = {
            "content_key": kaggle.content_key(member.crc, member.usize),
            "repo_id": "o/d", "full_name": "o/d", "path": name,
            "size": member.usize, "bucket": index.size_bucket(member.usize),
            "crc": member.crc, "csize": member.csize, "method": member.method,
            "header_offset": member.header_offset, "version": 3,
            "ref": "3", "is_lfs": False,
        }
        return raw, row

    class Client:
        """A Kaggle client backed by bytes in memory."""

        def __init__(self, raw, url="https://example/x.zip", status=302):
            self.raw = raw
            self.url = url
            self.status = status
            self.signed = []

        def signed_url_cached(self, ref, version=None):
            self.signed.append((ref, version))
            return (self.url, self.status) if self.url else (None, self.status)

        def range_get(self, url, start, end=None, attempts=4):
            if start is None:                       # suffix range
                body = self.raw[-end:] if end else b""
                return body, len(self.raw)
            return (self.raw[start:(None if end is None else end + 1)],
                    len(self.raw))

        def read_range(self, url, start, end=None, attempts=4):
            got = self.range_get(url, start, end, attempts)
            return None if got is None else got[0]

    def test_a_member_is_fetched_and_stored_under_its_content_key(self):
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        record = download.fetch_one(self.Client(raw), row, None, "kaggle")
        self.assertEqual(record["state"], "ok", record.get("reason"))
        self.assertEqual(record["size"], row["size"])
        stored = download._target(row["content_key"])
        self.assertTrue(stored.exists())
        self.assertEqual(len(stored.read_bytes()), row["size"])

    def test_the_pinned_version_is_what_is_requested(self):
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        client = self.Client(raw)
        download.fetch_one(client, row, None, "kaggle")
        self.assertEqual(client.signed, [("o/d", 3)])

    def test_a_vanished_version_is_rejected_as_drift_not_refetched(self):
        """Kaggle does not retain superseded versions, so a failed pin is the
        drift itself. Admitting today's bytes instead would put a file in the
        corpus whose size, bucket, delimiter and encoding all describe a
        different version."""
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        client = self.Client(raw, url=None, status=404)
        record = download.fetch_one(client, row, None, "kaggle")
        self.assertEqual(record["state"], "rejected")
        self.assertEqual(record["reason"], "version_drift")

    def test_content_that_does_not_match_its_crc_is_refused(self):
        """The index's row describes the bytes it measured; anything else is
        mislabelled by whichever column the dataset selected on."""
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        row = row | {"crc": (row["crc"] + 1) % (1 << 32)}
        record = download.fetch_one(self.Client(raw), row, None, "kaggle")
        self.assertEqual(record["state"], "rejected")
        self.assertEqual(record["reason"], "fetch_failed")

    def test_a_file_already_in_the_store_is_reused_not_refetched(self):
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        download.fetch_one(self.Client(raw), row, None, "kaggle")
        client = self.Client(raw)
        record = download.fetch_one(client, row, None, "kaggle")
        self.assertEqual(record["state"], "ok")
        self.assertTrue(record["reused"])
        self.assertEqual(client.signed, [])

    def test_the_selector_gate_still_runs_on_a_kaggle_file(self):
        from csvscrape import download

        self._bed()
        raw, row = self._archive_and_row()
        record = download.fetch_one(
            self.Client(raw), row, lambda p, r: "nope", "kaggle")
        self.assertEqual(record["state"], "rejected")
        self.assertEqual(record["reason"], "verify/nope")


class DialectReport(unittest.TestCase):
    """`kaggle-dialects` is the same report over the Kaggle log, so both pools are
    counted in one vocabulary, each log alone."""

    def _root(self):
        from csvscrape import config

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = root
        return root

    def _head(self, key, delimiter, **extra):
        """One heads record of the shape `probe.describe` writes."""
        return {
            "content_key": key, "bucket": 3, "size": 8, "ts": 1,
            "fetched": True, "http": 206, "via": "kaggle", "head_len": 8,
            "is_lfs_pointer": False, "looks_like_html": False,
            "agreement": 3,
            "custom": {"delimiter": delimiter, "encoding": "utf-8",
                       "complete": True, "quote_style": "necessary"},
            "clevercsv": {"delimiter": delimiter},
            "duckdb": {"Delimiter": delimiter},
            "charset_normalizer": {"encoding": "utf-8"},
        } | extra

    def test_the_bundle_pass_log_is_reportable(self):
        from csvscrape import kaggle, log, probe

        self._root()
        with log.Writer(kaggle.HEADS_LOG) as writer:
            writer.write(self._head("a-1", ","))
            writer.write(self._head("b-2", ";"))
            writer.write(self._head("c-3", ";"))
        out = probe.report(kaggle.HEADS_LOG)
        self.assertIn("3 files probed", out)
        self.assertIn("delimiter/custom", out)
        self.assertIn("';':2", out)
        self.assertIn("3 of 3", out)

    def test_each_pool_is_counted_alone(self):
        """One report per pool, or the distribution describes neither."""
        from csvscrape import kaggle, log, probe

        self._root()
        with log.Writer(probe.HEADS_LOG) as writer:
            writer.write(self._head("gh-1", "\t"))
        with log.Writer(kaggle.HEADS_LOG) as writer:
            writer.write(self._head("kg-1", ";"))
        self.assertIn("1 files probed", probe.report(kaggle.HEADS_LOG))
        self.assertNotIn("'\\t'", probe.report(kaggle.HEADS_LOG))
        self.assertNotIn("';'", probe.report(probe.HEADS_LOG))




class CliContract(unittest.TestCase):
    """Constants the parser spells out rather than importing, so that building
    it stays cheap. They have to stay equal to their authorities."""

    def test_sources_match_the_index(self):
        from csvscrape import cli, index

        self.assertEqual(set(cli.SOURCES), set(index.SOURCES))

    def test_the_per_dataset_default_matches_the_module(self):
        from csvscrape import cli, kaggle

        self.assertEqual(cli.KAGGLE_PER_DATASET, kaggle.MAX_FILES_PER_DATASET)
        self.assertEqual(cli.KAGGLE_RECORD_MEMBERS, kaggle.RECORD_MEMBERS)
        self.assertLess(kaggle.MAX_FILES_PER_DATASET, kaggle.RECORD_MEMBERS)

    def test_the_routes_default_matches_the_module(self):
        from csvscrape import cli, kaggle

        args = cli.build_parser().parse_args(["kaggle-bundle"])
        self.assertEqual(tuple(args.routes.split(",")), kaggle.ROUTES)



    def test_the_dataset_name_carries_a_non_default_source(self):
        """Two draws under identical flags from different pools are two
        corpora; a shared default name would have the second read the first's
        log as its own progress and stop early."""
        from csvscrape import cli

        parser = cli.build_parser()
        github = parser.parse_args(["download"])
        kaggle = parser.parse_args(["download", "--source", "kaggle"])
        self.assertEqual(github.source, "github")
        self.assertEqual(kaggle.source, "kaggle")


if __name__ == "__main__":
    unittest.main()


class DiscoveryReportsThePoolNotTheDelta(unittest.TestCase):
    """Discovery reports the pool size (`total`) beside what this run added
    (`new`)."""

    def setUp(self):
        from csvscrape import config

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = self.tmp

    def _seed(self, refs):
        from csvscrape import kaggle, log

        with log.Writer(kaggle.DATASETS_LOG) as writer:
            for ref in refs:
                writer.write({"ref": ref, "owner": "o", "ts": 1})

    def _run(self, responses):
        from csvscrape import kaggle

        class Client:
            bucket = None

            def get(self, path, _redirect=False, **params):
                return responses(params)

        return kaggle.discover(Client())

    def test_the_total_counts_the_pool_and_new_counts_the_run(self):
        self._seed([f"o/old{i}" for i in range(7)])
        seen = {"n": 0}

        def responses(params):
            # One page of fresh datasets per shard, then exhausted.
            if params["page"] != 1:
                return []
            seen["n"] += 1
            return [{"ref": f"o/fresh{seen['n']}-{i}"} for i in range(3)]

        stats = self._run(responses)
        self.assertEqual(stats["known"], 7)
        self.assertGreater(stats["new"], 0)
        self.assertEqual(stats["total"], stats["known"] + stats["new"])


class AShortPageIsNotExhaustion(unittest.TestCase):
    """`fileType=csv` is applied after a page is cut, so a page can come back
    short with more behind it; only an empty page ends a shard."""

    def setUp(self):
        from csvscrape import config

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = self.tmp

    def _run(self, page_sizes):
        """`page_sizes` maps page number -> how many items it returns."""
        from csvscrape import kaggle

        asked = []

        class Client:
            bucket = None

            def get(self, path, _redirect=False, **params):
                p = params["page"]
                if p == kaggle.MAX_PAGE:
                    return []                      # never capped
                asked.append(p)
                return [{"ref": f"o/d{p}-{i}"} for i in range(page_sizes.get(p, 0))]

        return kaggle.discover(Client()), asked

    def test_a_thin_page_does_not_end_the_shard(self):
        sizes = {1: 20, 2: 20, 3: 19, 4: 20, 5: 20, 6: 7}
        stats, asked = self._run(sizes)
        deepest = max(asked)
        self.assertGreaterEqual(deepest, 7,
                                "stopped at the short page instead of reading on")
        # Everything behind the thin page is still collected.
        self.assertGreaterEqual(stats["total"], sum(sizes.values()))

    def test_an_empty_page_still_ends_the_shard(self):
        """The only honest end-of-results signal; without it every shard would
        walk to MAX_PAGE."""
        stats, asked = self._run({1: 20, 2: 20, 3: 0})
        self.assertLessEqual(max(asked), 4)


class RecheckingExhaustedShards(unittest.TestCase):
    """`--recheck` makes each finished shard prove it is finished by resuming at
    the page after its last."""

    def setUp(self):
        from csvscrape import config

        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(setattr, config, "DATA_ROOT", config.DATA_ROOT)
        config.DATA_ROOT = self.tmp

    def _seed(self, records):
        from csvscrape import kaggle, log

        with log.Writer(kaggle.SHARDS_LOG) as writer:
            for rec in records:
                writer.write(rec)

    def test_a_finished_shard_is_reopened_at_the_page_it_reached(self):
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[0]
        self._seed([{"shard": kaggle._shard_key(lo, hi), "lo": lo, "hi": hi,
                     "state": "done", "pages": 281, "ts": 1}])
        self.assertEqual(kaggle.reopen_exhausted(), 1)
        resume = {(a, b): page for a, b, page in kaggle.shards()}
        self.assertEqual(resume[(lo, hi)], 281,
                         "reopening lost the shard's progress")

    def test_interior_nodes_are_left_alone(self):
        """A `capped` record says the shard was split, not that it finished.
        Reopening one re-enumerates a range its halves already cover."""
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[1]
        mid = kaggle.split_at(lo, hi)
        self._seed([{"shard": kaggle._shard_key(lo, hi), "lo": lo, "hi": hi,
                     "state": "capped", "split_at": mid, "pages": 0, "ts": 1}])
        self.assertEqual(kaggle.reopen_exhausted(), 0)
        pending = {(a, b) for a, b, _ in kaggle.shards()}
        self.assertNotIn((lo, hi), pending)
        self.assertIn((lo, mid), pending)


    def test_a_genuinely_exhausted_shard_costs_one_request(self):
        from csvscrape import kaggle

        lo, hi = kaggle.ROOT_SHARDS[0]
        self._seed([{"shard": kaggle._shard_key(lo, hi), "lo": lo, "hi": hi,
                     "state": "done", "pages": 4, "ts": 1}])
        asked = []

        class Client:
            bucket = None

            def get(self, path, _redirect=False, **params):
                asked.append(params["page"])
                return []

        stats = kaggle.discover(Client(), recheck=True)
        self.assertEqual(stats["reopened"], 1)   # only the one seeded record
        self.assertIn(5, asked, "did not resume at the page after the last")
        self.assertNotIn(4, asked, "re-read a page it already had")

    def test_a_truncated_shard_picks_up_where_it_stopped(self):
        from csvscrape import kaggle

        # Every root seeded, so the count below is not confounded by the three
        # that would otherwise enumerate from page 1.
        self._seed([{"shard": kaggle._shard_key(lo, hi), "lo": lo, "hi": hi,
                     "state": "done", "pages": 3, "ts": 1}
                    for lo, hi in kaggle.ROOT_SHARDS])

        class Client:
            bucket = None

            def get(self, path, _redirect=False, **params):
                p = params["page"]
                if p == kaggle.MAX_PAGE or p > 6:
                    return []
                # Refs keyed by the shard too, so the four do not collide.
                return [{"ref": f"o/d{params['minSize']}-{p}-{i}"}
                        for i in range(20)]

        stats = kaggle.discover(Client(), recheck=True)
        self.assertEqual(stats["reopened"], len(kaggle.ROOT_SHARDS))
        # Pages 4, 5 and 6 of each root were behind its premature `done`.
        self.assertEqual(stats["new"], len(kaggle.ROOT_SHARDS) * 3 * 20)
