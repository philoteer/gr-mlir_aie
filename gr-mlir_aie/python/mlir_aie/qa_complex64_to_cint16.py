#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import struct

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


def packed_cint16(real, imag):
    packed = (imag & 0xFFFF) << 16 | (real & 0xFFFF)
    return struct.unpack("i", struct.pack("I", packed))[0]


class qa_complex64_to_cint16(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def run_converter(self, samples, safe=False):
        source = blocks.vector_source_c(samples, False)
        converter = mlir_aie.complex64_to_cint16(safe)
        sink = blocks.vector_sink_i()
        self.tb.connect(source, converter, sink)
        self.tb.run()
        return tuple(sink.data())

    def test_unsafe_q15_conversion_and_interleaving(self):
        samples = (
            0j,
            complex(0.5, -0.5),
            complex(0.1, -0.1),
            complex(0.25, 0.75),
        )
        expected = (
            packed_cint16(0, 0),
            packed_cint16(16384, -16384),
            packed_cint16(3276, -3276),
            packed_cint16(8192, 24576),
        )

        self.assertEqual(expected, self.run_converter(samples))

    def test_safe_q15_conversion_and_interleaving(self):
        samples = (
            0j,
            complex(0.5, -0.5),
            complex(0.1, -0.1),
            complex(1.0, -1.0),
            complex(2.0, -2.0),
            complex(0.25, 0.75),
            complex(math.nan, math.nan),
        )
        expected = (
            packed_cint16(0, 0),
            packed_cint16(16384, -16384),
            packed_cint16(3277, -3277),
            packed_cint16(32767, -32768),
            packed_cint16(32767, -32768),
            packed_cint16(8192, 24576),
            packed_cint16(0, 0),
        )

        self.assertEqual(expected, self.run_converter(samples, safe=True))


if __name__ == "__main__":
    gr_unittest.run(qa_complex64_to_cint16)
