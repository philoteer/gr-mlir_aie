#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


class qa_float32_to_int32_Q(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def run_converter(self, samples, safe=False):
        source = blocks.vector_source_f(samples, False)
        converter = mlir_aie.float32_to_int32_Q(4, 27, safe)
        sink = blocks.vector_sink_i()
        self.tb.connect(source, converter, sink)
        self.tb.run()
        return tuple(sink.data())

    def test_unsafe_q4_27_conversion(self):
        samples = (0.0, 1.5, -2.25, 0.1, -0.1)
        expected = (0, 201326592, -301989888, 13421773, -13421773)

        self.assertEqual(expected, self.run_converter(samples))

    def test_safe_q4_27_conversion(self):
        samples = (0.0, 0.1, -0.1, 100.0, -100.0, math.nan)
        expected = (0, 13421773, -13421773, 2147483647, -2147483648, 0)

        self.assertEqual(expected, self.run_converter(samples, safe=True))

    def test_invalid_format(self):
        self.assertRaises(ValueError, mlir_aie.float32_to_int32_Q, 4, 26)


if __name__ == "__main__":
    gr_unittest.run(qa_float32_to_int32_Q)
