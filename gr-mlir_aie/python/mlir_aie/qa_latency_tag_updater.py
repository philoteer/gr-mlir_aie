#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later

import hashlib
import json
import os
import tempfile
import time

import pmt
from gnuradio import blocks, gr, gr_unittest, mlir_aie


def field(value, name):
    return pmt.dict_ref(value, pmt.intern(name), pmt.PMT_NIL)


def dictionary(**values):
    result = pmt.make_dict()
    for name, value in values.items():
        result = pmt.dict_add(result, pmt.intern(name), value)
    return result


def tag(offset, key, value):
    result = gr.tag_t()
    result.offset = offset
    result.key = pmt.intern(key)
    result.value = value
    result.srcid = pmt.intern("original-source")
    return result


class qa_latency_tag_updater(gr_unittest.TestCase):
    def check_point(self, original, updated, name, before, after):
        self.assertEqual(original.offset, updated.offset)
        self.assertTrue(pmt.equal(original.key, updated.key))
        self.assertTrue(pmt.equal(original.srcid, updated.srcid))
        keys = pmt.dict_keys(original.value)
        for index in range(pmt.length(keys)):
            key = pmt.nth(index, keys)
            if pmt.symbol_to_string(key) != "timestamps_ns":
                self.assertTrue(pmt.equal(pmt.dict_ref(original.value, key, pmt.PMT_NIL),
                                          pmt.dict_ref(updated.value, key, pmt.PMT_NIL)))
        timestamps = field(updated.value, "timestamps_ns")
        timestamp = field(timestamps, name)
        self.assertTrue(pmt.is_uint64(timestamp))
        self.assertGreaterEqual(pmt.to_uint64(timestamp), before)
        self.assertLessEqual(pmt.to_uint64(timestamp), after)
        self.assertGreaterEqual(pmt.to_uint64(timestamp),
                                pmt.to_uint64(field(updated.value, "timestamp_ns")))
        previous = field(original.value, "timestamps_ns")
        if pmt.is_dict(previous):
            keys = pmt.dict_keys(previous)
            for index in range(pmt.length(keys)):
                key = pmt.nth(index, keys)
                self.assertTrue(pmt.equal(pmt.dict_ref(previous, key, pmt.PMT_NIL),
                                          pmt.dict_ref(timestamps, key, pmt.PMT_NIL)))

    def test_injector_and_chained_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "test.sigmf-meta")
            with open(path, "w") as file:
                json.dump({"captures": [{"core:sample_start": 0, "core:frequency": 2.412e9}],
                           "annotations": [
                               {"core:sample_start": start, "core:sample_count": count,
                                "core:label": "wifi-frame-%08d" % start}
                               for start, count in [(0, 4), (3, 2), (8, 1), (23, 9)]]}, file)
            samples = tuple(range(32))
            existing = tag(7, "existing", pmt.from_long(42))
            source = blocks.vector_source_i(samples, False, 1, [existing])
            injector = mlir_aie.sigmf_tag_injector(path, gr.sizeof_int)
            points = [mlir_aie.latency_tag_updater(name, gr.sizeof_int)
                      for name in ["after_sync", "before_decode", "after_sync"]]
            tb = gr.top_block()
            tb.connect(source, injector, *points)
            sinks = []
            for block in [injector] + points:
                block.set_max_noutput_items(4)
                sink = blocks.vector_sink_i()
                tb.connect(block, sink)
                sinks.append(sink)
            before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            tb.run()
            after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            for sink in sinks:
                self.assertEqual(samples, tuple(sink.data()))
                self.assertEqual(10, len(sink.tags()))  # 8 events, frequency, existing
            for index, name in enumerate(["after_sync", "before_decode", "after_sync"]):
                originals = sorted(sinks[index].tags(), key=lambda t: (t.offset, str(t.key)))
                updated = sorted(sinks[index + 1].tags(), key=lambda t: (t.offset, str(t.key)))
                for old, new in zip(originals, updated):
                    if pmt.symbol_to_string(old.key) in ("frame_start", "frame_end"):
                        self.check_point(old, new, name, before, after)
                        if index == 0:
                            self.assertFalse(pmt.dict_has_key(old.value, pmt.intern("timestamps_ns")))
                        if index == 2:
                            self.assertTrue(pmt.equal(old.value, new.value))
                    else:
                        self.assertEqual(old.offset, new.offset)
                        self.assertTrue(pmt.equal(old.key, new.key))
                        self.assertTrue(pmt.equal(old.value, new.value))
                        self.assertTrue(pmt.equal(old.srcid, new.srcid))

    def test_existing_fields_and_one_timestamp_per_callback(self):
        original = dictionary(frame_id=pmt.intern("wifi-frame-0"),
                              timestamp_ns=pmt.from_uint64(1),
                              sample_start=pmt.from_uint64(0),
                              custom=pmt.intern("preserve me"),
                              timestamps_ns=dictionary(earlier=pmt.from_uint64(99)))
        tags = [tag(0, "frame_start", original), tag(3, "frame_end", original)]
        # Eight bytes per vector item exercises arbitrary item sizes/vectors.
        source = blocks.vector_source_i(tuple(range(64)), False, 2, tags)
        updater = mlir_aie.latency_tag_updater("checkpoint", 2 * gr.sizeof_int)
        updater.set_min_noutput_items(32)
        updater.set_max_noutput_items(32)
        sink = blocks.vector_sink_i(2)
        tb = gr.top_block()
        tb.connect(source, updater, sink)
        before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        tb.run()
        after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.assertEqual(tuple(range(64)), tuple(sink.data()))
        self.assertEqual(2, len(sink.tags()))
        for old, new in zip(tags, sink.tags()):
            self.check_point(old, new, "checkpoint", before, after)
        self.assertTrue(pmt.equal(field(sink.tags()[0].value, "timestamps_ns"),
                                  field(sink.tags()[1].value, "timestamps_ns")))

    def test_nonmatching_and_malformed_tags_pass_unchanged(self):
        valid = dictionary(timestamp_ns=pmt.from_uint64(1))
        tags = [tag(0, "unrelated", valid),
                tag(1, "frame_start", pmt.intern("not a dictionary")),
                tag(2, "frame_end", dictionary(frame_id=pmt.intern("missing time"))),
                tag(3, "frame_start", dictionary(timestamp_ns=pmt.from_double(1.0))),
                tag(4, "frame_end", dictionary(timestamp_ns=pmt.from_uint64(1),
                                               timestamps_ns=pmt.intern("invalid")))]
        source = blocks.vector_source_b(tuple(range(8)), False, 1, tags)
        updater = mlir_aie.latency_tag_updater("point", gr.sizeof_char)
        sink = blocks.vector_sink_b()
        tb = gr.top_block()
        tb.connect(source, updater, sink)
        tb.run()
        self.assertEqual(tuple(range(8)), tuple(sink.data()))
        self.assertEqual(len(tags), len(sink.tags()))
        for old, new in zip(tags, sink.tags()):
            self.assertEqual(old.offset, new.offset)
            self.assertTrue(pmt.equal(old.key, new.key))
            self.assertTrue(pmt.equal(old.value, new.value))
            self.assertTrue(pmt.equal(old.srcid, new.srcid))

    def test_invalid_arguments(self):
        for name, itemsize in [("", 8), ("point", 0), ("point", 2**31)]:
            with self.subTest(name=name, itemsize=itemsize):
                with self.assertRaises(ValueError):
                    mlir_aie.latency_tag_updater(name, itemsize)

    def test_synthetic_latency_recording(self):
        path = os.environ.get("SIGMF_LATENCY_METADATA")
        if not path:
            self.skipTest("set SIGMF_LATENCY_METADATA to test the generated I/Q recording")
        with open(path) as file:
            metadata = json.load(file)
        data_path = os.path.splitext(path)[0] + ".sigmf-data"
        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "output.cf32")
            source = blocks.file_source(gr.sizeof_gr_complex, data_path, False)
            injector = mlir_aie.sigmf_tag_injector(path)
            first = mlir_aie.latency_tag_updater("first")
            second = mlir_aie.latency_tag_updater("second")
            output = blocks.file_sink(gr.sizeof_gr_complex, output_path, False)
            tags = blocks.tag_debug(gr.sizeof_gr_complex, "latency QA", "")
            tags.set_display(False)
            tags.set_save_all(True)
            tb = gr.top_block()
            tb.connect(source, injector, first, second, output)
            tb.connect(second, tags)
            before = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            tb.run()
            after = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            output.close()
            with open(data_path, "rb") as original, open(output_path, "rb") as updated:
                self.assertEqual(hashlib.sha256(original.read()).digest(),
                                 hashlib.sha256(updated.read()).digest())
            expected = []
            for annotation in metadata["annotations"]:
                if annotation.get("core:label", "").startswith("wifi-frame-"):
                    start = annotation["core:sample_start"]
                    expected.extend([(start, "frame_start", annotation["core:label"]),
                                     (start + annotation["core:sample_count"] - 1,
                                      "frame_end", annotation["core:label"])])
            events = [t for t in tags.current_tags()
                      if pmt.symbol_to_string(t.key) in ("frame_start", "frame_end")]
            self.assertTrue(expected)
            self.assertEqual(sorted(expected), sorted(
                (t.offset, pmt.symbol_to_string(t.key),
                 pmt.symbol_to_string(field(t.value, "frame_id"))) for t in events))
            for event in events:
                first_ns = field(field(event.value, "timestamps_ns"), "first")
                second_ns = field(field(event.value, "timestamps_ns"), "second")
                self.assertTrue(pmt.is_uint64(first_ns))
                self.assertTrue(pmt.is_uint64(second_ns))
                self.assertLessEqual(before, pmt.to_uint64(field(event.value, "timestamp_ns")))
                self.assertLessEqual(pmt.to_uint64(field(event.value, "timestamp_ns")),
                                     pmt.to_uint64(first_ns))
                self.assertLessEqual(pmt.to_uint64(first_ns), pmt.to_uint64(second_ns))
                self.assertLessEqual(pmt.to_uint64(second_ns), after)


if __name__ == "__main__":
    gr_unittest.run(qa_latency_tag_updater)
