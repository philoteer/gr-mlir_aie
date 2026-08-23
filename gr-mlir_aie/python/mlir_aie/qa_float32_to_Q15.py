#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


class qa_float32_to_Q15(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def run_converter(self, samples, safe=False):
        source = blocks.vector_source_f(samples, False)
        converter = mlir_aie.float32_to_Q15(safe)
        sink = blocks.vector_sink_s()
        self.tb.connect(source, converter, sink)
        self.tb.run()
        return tuple(sink.data())

    def test_unsafe_q15_conversion(self):
        samples = (0.0, 0.5, -0.5, 0.1, -0.1, 0.25)
        expected = (0, 16384, -16384, 3276, -3276, 8192)

        self.assertEqual(expected, self.run_converter(samples))

    def test_safe_q15_conversion(self):
        samples = (0.0, 0.5, -0.5, 0.1, -0.1, 1.0, -1.0, 2.0, -2.0, math.nan)
        expected = (0, 16384, -16384, 3277, -3277, 32767, -32768, 32767, -32768, 0)

        self.assertEqual(expected, self.run_converter(samples, safe=True))


if __name__ == "__main__":
    gr_unittest.run(qa_float32_to_Q15)
