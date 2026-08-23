#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import struct

import numpy as np

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


def packed_cint32(real, imag):
    packed = (imag & 0xFFFFFFFF) << 32 | (real & 0xFFFFFFFF)
    return struct.unpack("q", struct.pack("Q", packed))[0]


class vector_sink_s64(gr.sync_block):

    def __init__(self):
        gr.sync_block.__init__(self, name="vector_sink_s64", in_sig=[np.int64], out_sig=None)
        self.data = []

    def work(self, input_items, output_items):
        self.data.extend(input_items[0].tolist())
        return len(input_items[0])


class qa_complex64_to_cint32(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def run_converter(self, samples, safe=False):
        source = blocks.vector_source_c(samples, False)
        converter = mlir_aie.complex64_to_cint32(4, 27, safe)
        sink = vector_sink_s64()
        self.tb.connect(source, converter, sink)
        self.tb.run()
        return tuple(sink.data)

    def test_unsafe_q4_27_conversion_and_interleaving(self):
        samples = (complex(1.5, -2.25), complex(0.1, -0.1))
        expected = (
            packed_cint32(201326592, -301989888),
            packed_cint32(13421773, -13421773),
        )

        self.assertEqual(expected, self.run_converter(samples))

    def test_safe_q4_27_conversion_and_saturation(self):
        samples = (complex(100.0, -100.0), complex(math.nan, math.nan))
        expected = (
            packed_cint32(2147483647, -2147483648),
            packed_cint32(0, 0),
        )

        self.assertEqual(expected, self.run_converter(samples, safe=True))

    def test_invalid_format(self):
        self.assertRaises(ValueError, mlir_aie.complex64_to_cint32, 4, 26)


if __name__ == "__main__":
    gr_unittest.run(qa_complex64_to_cint32)
