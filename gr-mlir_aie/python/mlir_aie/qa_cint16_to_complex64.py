#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

import struct

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


def packed_cint16(real, imag):
    packed = (imag & 0xFFFF) << 16 | (real & 0xFFFF)
    return struct.unpack("i", struct.pack("I", packed))[0]

class qa_cint16_to_complex64(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_q15_conversion_and_interleaving(self):
        samples = (
            packed_cint16(0, 0),
            packed_cint16(16384, -16384),
            packed_cint16(32767, -32768),
            packed_cint16(-8192, 24576),
        )
        expected = (
            0j,
            complex(0.5, -0.5),
            complex(32767.0 / 32768.0, -1.0),
            complex(-0.25, 0.75),
        )

        source = blocks.vector_source_i(samples, False)
        converter = mlir_aie.cint16_to_complex64()
        sink = blocks.vector_sink_c()
        self.tb.connect(source, converter, sink)
        self.tb.run()

        self.assertEqual(expected, tuple(sink.data()))


if __name__ == '__main__':
    gr_unittest.run(qa_cint16_to_complex64)
