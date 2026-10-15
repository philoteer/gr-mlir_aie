#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import tempfile
import time
import hashlib

import pmt
from gnuradio import blocks, gr, gr_unittest, mlir_aie


class qa_sigmf_tag_injector(gr_unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "test.sigmf-meta")

    def tearDown(self):
        self.directory.cleanup()

    def metadata(self, captures, annotations=None):
        with open(self.path, "w") as file:
            json.dump({"global": {"core:datatype": "ci16_le"},
                       "captures": captures, "annotations": annotations or []}, file)

    def run_stream(self, start_sample=0, tag_key="rx_freq", single_callback=False):
        samples = tuple(range(24))
        tag = gr.tag_t()
        tag.offset = 7
        tag.key = pmt.intern("existing")
        tag.value = pmt.from_long(42)
        source = blocks.vector_source_i(samples, False, 1, [tag])
        injector = mlir_aie.sigmf_tag_injector(
            self.path, gr.sizeof_int, tag_key, start_sample)
        injector.set_max_noutput_items(24 if single_callback else 4)
        if single_callback:
            injector.set_min_noutput_items(24)
        sink = blocks.vector_sink_i()
        tb = gr.top_block()
        tb.connect(source, injector, sink)
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        tb.run()
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.frame_tags = [t for t in sink.tags()
                           if pmt.symbol_to_string(t.key) == "frame_start"]
        self.check_frame_timestamps(self.frame_tags, before, after)
        self.end_tags = [t for t in sink.tags()
                         if pmt.symbol_to_string(t.key) == "frame_end"]
        self.check_frame_timestamps(self.end_tags, before, after)
        self.event_tags = sorted(self.frame_tags + self.end_tags, key=lambda t: t.offset)
        timestamps = [pmt.to_uint64(pmt.dict_ref(
            t.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL)) for t in self.event_tags]
        self.assertEqual(sorted(timestamps), timestamps)
        self.assertEqual(samples, tuple(sink.data()))
        existing = [t for t in sink.tags() if pmt.symbol_to_string(t.key) == "existing"]
        self.assertEqual([(7, 42)], [(t.offset, pmt.to_long(t.value)) for t in existing])
        return [(t.offset, pmt.to_double(t.value)) for t in sink.tags()
                if pmt.symbol_to_string(t.key) == tag_key]

    def check_frame_timestamps(self, tags, before, after):
        previous = before
        ids = []
        for tag in tags:
            self.assertTrue(pmt.is_dict(tag.value))
            timestamp = pmt.dict_ref(tag.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL)
            self.assertTrue(pmt.is_uint64(timestamp))
            timestamp = pmt.to_uint64(timestamp)
            self.assertGreaterEqual(timestamp, previous)
            self.assertLessEqual(timestamp, after)
            previous = timestamp
            ids.append(pmt.symbol_to_string(pmt.dict_ref(
                tag.value, pmt.intern("frame_id"), pmt.PMT_NIL)))
        self.assertEqual(len(ids), len(set(ids)))

    def test_frames_and_frequencies(self):
        # Deliberately unsorted, with unrelated annotations and an EOF frame.
        offsets = [23, 0, 4, 3, 8, 24]
        self.metadata([{"core:sample_start": 0, "core:frequency": 2.412e9},
                       {"core:sample_start": 4, "core:frequency": 2.417e9}],
                      [{"core:sample_start": n, "core:label": "wifi-frame-%08d" % n}
                       for n in offsets] + [{"core:label": "noise"}, {}])
        self.assertEqual([(0, 2.412e9), (4, 2.417e9)], self.run_stream())
        self.assertEqual([0, 3, 4, 8, 23], [t.offset for t in self.frame_tags])
        for tag in self.frame_tags:
            self.assertEqual("wifi-frame-%08d" % tag.offset,
                             pmt.symbol_to_string(pmt.dict_ref(
                                 tag.value, pmt.intern("frame_id"), pmt.PMT_NIL)))
            self.assertEqual(tag.offset, pmt.to_uint64(pmt.dict_ref(
                tag.value, pmt.intern("sample_start"), pmt.PMT_NIL)))

    def test_frame_seek_and_large_offsets(self):
        base = 2**33
        self.metadata([], [{"core:sample_start": base + n,
                            "core:sample_count": 8,
                            "core:label": "wifi-frame-%08d" % i}
                           for i, n in enumerate([-1, 0, 4, 23, 24])])
        self.assertEqual([], self.run_stream(base))
        self.assertEqual([0, 4, 23], [t.offset for t in self.frame_tags])
        self.assertEqual([base, base + 4, base + 23], [pmt.to_uint64(pmt.dict_ref(
            t.value, pmt.intern("sample_start"), pmt.PMT_NIL)) for t in self.frame_tags])
        # The frame starting before the seek still has an end event after it.
        self.assertEqual([6, 7, 11], [t.offset for t in self.end_tags])
        self.assertEqual([base - 1, base, base + 4], [pmt.to_uint64(pmt.dict_ref(
            t.value, pmt.intern("sample_start"), pmt.PMT_NIL)) for t in self.end_tags])
        self.assertEqual([base + 6, base + 7, base + 11], [pmt.to_uint64(pmt.dict_ref(
            t.value, pmt.intern("sample_offset"), pmt.PMT_NIL)) for t in self.end_tags])

    def test_frame_ends_and_callback_timestamp(self):
        self.metadata([], [{"core:sample_start": start, "core:sample_count": count,
                            "core:label": "wifi-frame-%08d" % start}
                           for start, count in [(23, 2), (2, 3), (0, 4), (8, 1), (22, 2)]])
        for single_callback in (False, True):
            self.assertEqual([], self.run_stream(single_callback=single_callback))
            self.assertEqual([0, 2, 8, 22, 23], [t.offset for t in self.frame_tags])
            self.assertEqual([3, 4, 8, 23], [t.offset for t in self.end_tags])
            for tag in self.event_tags:
                self.assertEqual(tag.offset, pmt.to_uint64(pmt.dict_ref(
                    tag.value, pmt.intern("sample_offset"), pmt.PMT_NIL)))
            if single_callback:
                timestamps = {pmt.to_uint64(pmt.dict_ref(
                    t.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL))
                              for t in self.event_tags}
                self.assertEqual(1, len(timestamps))

    def test_invalid_frame_counts_and_overflow(self):
        for start, count in [(0, 0), (0, -1), (0, 1.5), (0, "8"), (0, True),
                             (0, None), (2**64 - 1, 2), (2**64 - 4, 5)]:
            with self.subTest(start=start, count=count):
                self.metadata([], [{"core:sample_start": start, "core:sample_count": count,
                                    "core:label": "wifi-frame-0"}])
                with self.assertRaises(RuntimeError):
                    mlir_aie.sigmf_tag_injector(self.path)
        # The largest representable end is valid; do not reject it via an
        # overflowing intermediate start + count calculation.
        self.metadata([], [{"core:sample_start": 2**64 - 4, "core:sample_count": 4,
                            "core:label": "wifi-frame-0"}])
        self.run_stream(2**64 - 4)
        self.assertEqual([3], [t.offset for t in self.end_tags])
        self.assertEqual(2**64 - 1, pmt.to_uint64(pmt.dict_ref(
            self.end_tags[0].value, pmt.intern("sample_offset"), pmt.PMT_NIL)))

    def test_invalid_frame_annotations(self):
        for annotations in (
            [{"core:label": "wifi-frame-0"}],
            [{"core:label": "wifi-frame-0", "core:sample_start": -1}],
            [{"core:label": "wifi-frame-0", "core:sample_start": 1.5}],
            [{"core:label": "wifi-frame-0", "core:sample_start": n} for n in (0, 4)],
        ):
            self.metadata([], annotations)
            with self.assertRaises(RuntimeError):
                mlir_aie.sigmf_tag_injector(self.path)

    def test_synthetic_latency_recording(self):
        metadata_path = os.environ.get("SIGMF_LATENCY_METADATA")
        if not metadata_path:
            self.skipTest("set SIGMF_LATENCY_METADATA to test the generated I/Q recording")
        with open(metadata_path) as file:
            metadata = json.load(file)
        data_path = os.path.splitext(metadata_path)[0] + ".sigmf-data"
        output_path = os.path.join(self.directory.name, "output.cf32")
        self.assertEqual("cf32_le", metadata["global"]["core:datatype"])
        source = blocks.file_source(gr.sizeof_gr_complex, data_path, False)
        injector = mlir_aie.sigmf_tag_injector(metadata_path)
        injector.set_max_noutput_items(4096)
        samples = blocks.file_sink(gr.sizeof_gr_complex, output_path, False)
        tags = blocks.tag_debug(gr.sizeof_gr_complex, "latency QA", "")
        tags.set_display(False)
        tags.set_save_all(True)
        tb = gr.top_block()
        tb.connect(source, injector, samples)
        tb.connect(injector, tags)
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        tb.run()
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        samples.close()
        with open(data_path, "rb") as original, open(output_path, "rb") as output:
            self.assertEqual(hashlib.sha256(original.read()).digest(),
                             hashlib.sha256(output.read()).digest())
        frames = [t for t in tags.current_tags()
                  if pmt.symbol_to_string(t.key) == "frame_start"]
        expected = sorted((a["core:sample_start"], a["core:label"])
                          for a in metadata["annotations"]
                          if a.get("core:label", "").startswith("wifi-frame-"))
        self.assertTrue(expected)
        self.assertEqual(expected, [(t.offset, pmt.symbol_to_string(pmt.dict_ref(
            t.value, pmt.intern("frame_id"), pmt.PMT_NIL))) for t in frames])
        self.check_frame_timestamps(frames, before, after)
        ends = [t for t in tags.current_tags()
                if pmt.symbol_to_string(t.key) == "frame_end"]
        expected_ends = sorted((a["core:sample_start"] + a["core:sample_count"] - 1,
                                a["core:label"])
                               for a in metadata["annotations"]
                               if a.get("core:label", "").startswith("wifi-frame-"))
        self.assertEqual(expected_ends, [(t.offset, pmt.symbol_to_string(pmt.dict_ref(
            t.value, pmt.intern("frame_id"), pmt.PMT_NIL))) for t in ends])
        self.check_frame_timestamps(ends, before, after)
        for tag in frames + ends:
            self.assertEqual(tag.offset, pmt.to_uint64(pmt.dict_ref(
                tag.value, pmt.intern("sample_offset"), pmt.PMT_NIL)))
        start_times = {pmt.symbol_to_string(pmt.dict_ref(
            t.value, pmt.intern("frame_id"), pmt.PMT_NIL)): pmt.to_uint64(pmt.dict_ref(
                t.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL)) for t in frames}
        for tag in ends:
            frame_id = pmt.symbol_to_string(pmt.dict_ref(
                tag.value, pmt.intern("frame_id"), pmt.PMT_NIL))
            self.assertGreaterEqual(pmt.to_uint64(pmt.dict_ref(
                tag.value, pmt.intern("timestamp_ns"), pmt.PMT_NIL)), start_times[frame_id])
        frequencies = [(t.offset, pmt.to_double(t.value)) for t in tags.current_tags()
                       if pmt.symbol_to_string(t.key) == "rx_freq"]
        self.assertEqual([(0, metadata["captures"][0]["core:frequency"])], frequencies)

    def test_exact_offsets_and_frequency_changes(self):
        self.metadata([
            {"core:sample_start": 0, "core:frequency": 2412000000.001859},
            {"core:sample_start": 3, "core:frequency": 2412000000.001859},
            {"core:sample_start": 4, "core:frequency": 2417000000.001862},
            {"core:sample_start": 5},
            {"core:sample_start": 8, "core:frequency": 2417000000.001862},
            {"core:sample_start": 23, "core:frequency": 2412000000.001859},
            {"core:sample_start": 24, "core:frequency": 2422000000.0},
        ])
        self.assertEqual([(0, 2412000000.001859), (4, 2417000000.001862),
                          (23, 2412000000.001859)], self.run_stream())

    def test_large_offsets_and_initial_seek(self):
        base = 2**33
        self.metadata([
            {"core:sample_start": 0, "core:frequency": 1e9},
            {"core:sample_start": base - 2, "core:frequency": 2e9},
            {"core:sample_start": base + 4, "core:frequency": 3e9},
        ])
        self.assertEqual([(0, 2e9), (4, 3e9)], self.run_stream(base, "frequency"))

    def test_first_frequency_after_stream_start(self):
        self.metadata([{"core:sample_start": 0},
                       {"core:sample_start": 8, "core:frequency": 1e9}])
        self.assertEqual([(8, 1e9)], self.run_stream())
        self.metadata([])
        self.assertEqual([], self.run_stream())

    def test_invalid_metadata(self):
        for captures in (
            [{"core:sample_start": -1, "core:frequency": 1e9}],
            [{"core:sample_start": 1.5, "core:frequency": 1e9}],
            [{"core:sample_start": 0, "core:frequency": "1e9"}],
            [{"core:frequency": 1e9}],
            [{"core:sample_start": 2}, {"core:sample_start": 1}],
        ):
            self.metadata(captures)
            with self.assertRaises(RuntimeError):
                mlir_aie.sigmf_tag_injector(self.path)
        with open(self.path, "w") as file:
            file.write("{invalid json")
        with self.assertRaises(RuntimeError):
            mlir_aie.sigmf_tag_injector(self.path)
        with self.assertRaises(RuntimeError):
            mlir_aie.sigmf_tag_injector(self.path + ".missing")
        self.metadata([])
        with self.assertRaises(ValueError):
            mlir_aie.sigmf_tag_injector(self.path, 0)
        with self.assertRaises(ValueError):
            mlir_aie.sigmf_tag_injector(self.path, 4, "")


if __name__ == "__main__":
    gr_unittest.run(qa_sigmf_tag_injector)
