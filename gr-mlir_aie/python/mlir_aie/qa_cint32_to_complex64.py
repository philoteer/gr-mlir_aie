#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import struct

import numpy as np

from gnuradio import blocks, gr, gr_unittest
from gnuradio import mlir_aie


def packed_cint32(real, imag):
    packed = (imag & 0xFFFFFFFF) << 32 | (real & 0xFFFFFFFF)
    return struct.unpack("q", struct.pack("Q", packed))[0]


class vector_source_s64(gr.sync_block):

    def __init__(self, samples):
        gr.sync_block.__init__(self, name="vector_source_s64", in_sig=None, out_sig=[np.int64])
        self.samples = np.asarray(samples, dtype=np.int64)
        self.offset = 0

    def work(self, input_items, output_items):
        if self.offset == len(self.samples):
            return -1
        count = min(len(output_items[0]), len(self.samples) - self.offset)
        output_items[0][:count] = self.samples[self.offset:self.offset + count]
        self.offset += count
        return count


class qa_cint32_to_complex64(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_q4_27_conversion_and_interleaving(self):
        samples = (
            packed_cint32(0, 0),
            packed_cint32(201326592, -301989888),
            packed_cint32(-67108864, 100663296),
        )
        expected = (0j, complex(1.5, -2.25), complex(-0.5, 0.75))

        source = vector_source_s64(samples)
        converter = mlir_aie.cint32_to_complex64(4, 27)
        sink = blocks.vector_sink_c()
        self.tb.connect(source, converter, sink)
        self.tb.run()

        self.assertEqual(expected, tuple(sink.data()))

    def test_invalid_format(self):
        self.assertRaises(ValueError, mlir_aie.cint32_to_complex64, 4, 26)


if __name__ == "__main__":
    gr_unittest.run(qa_cint32_to_complex64)
