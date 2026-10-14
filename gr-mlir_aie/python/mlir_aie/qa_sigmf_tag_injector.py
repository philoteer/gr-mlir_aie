#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os
import tempfile

import pmt
from gnuradio import blocks, gr, gr_unittest, mlir_aie


class qa_sigmf_tag_injector(gr_unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "test.sigmf-meta")

    def tearDown(self):
        self.directory.cleanup()

    def metadata(self, captures):
        with open(self.path, "w") as file:
            json.dump({"global": {"core:datatype": "ci16_le"},
                       "captures": captures, "annotations": []}, file)

    def run_stream(self, start_sample=0, tag_key="rx_freq"):
        samples = tuple(range(24))
        tag = gr.tag_t()
        tag.offset = 7
        tag.key = pmt.intern("existing")
        tag.value = pmt.from_long(42)
        source = blocks.vector_source_i(samples, False, 1, [tag])
        injector = mlir_aie.sigmf_tag_injector(
            self.path, gr.sizeof_int, tag_key, start_sample)
        injector.set_max_noutput_items(4)
        sink = blocks.vector_sink_i()
        tb = gr.top_block()
        tb.connect(source, injector, sink)
        tb.run()
        self.assertEqual(samples, tuple(sink.data()))
        existing = [t for t in sink.tags() if pmt.symbol_to_string(t.key) == "existing"]
        self.assertEqual([(7, 42)], [(t.offset, pmt.to_long(t.value)) for t in existing])
        return [(t.offset, pmt.to_double(t.value)) for t in sink.tags()
                if pmt.symbol_to_string(t.key) == tag_key]

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
