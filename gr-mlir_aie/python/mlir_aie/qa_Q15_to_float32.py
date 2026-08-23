#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


class qa_Q15_to_float32(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_q15_conversion(self):
        samples = (0, 16384, -16384, 32767, -32768, 8192, -8192)
        expected = (0.0, 0.5, -0.5, 32767.0 / 32768.0, -1.0, 0.25, -0.25)

        source = blocks.vector_source_s(samples, False)
        converter = mlir_aie.Q15_to_float32()
        sink = blocks.vector_sink_f()
        self.tb.connect(source, converter, sink)
        self.tb.run()

        self.assertEqual(expected, tuple(sink.data()))


if __name__ == "__main__":
    gr_unittest.run(qa_Q15_to_float32)
