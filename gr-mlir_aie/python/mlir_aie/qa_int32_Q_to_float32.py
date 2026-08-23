#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


class qa_int32_Q_to_float32(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_q4_27_conversion(self):
        samples = (0, 201326592, -301989888, -67108864, 100663296)
        expected = (0.0, 1.5, -2.25, -0.5, 0.75)

        source = blocks.vector_source_i(samples, False)
        converter = mlir_aie.int32_Q_to_float32(4, 27)
        sink = blocks.vector_sink_f()
        self.tb.connect(source, converter, sink)
        self.tb.run()

        self.assertEqual(expected, tuple(sink.data()))

    def test_invalid_format(self):
        self.assertRaises(ValueError, mlir_aie.int32_Q_to_float32, 4, 26)


if __name__ == "__main__":
    gr_unittest.run(qa_int32_Q_to_float32)
