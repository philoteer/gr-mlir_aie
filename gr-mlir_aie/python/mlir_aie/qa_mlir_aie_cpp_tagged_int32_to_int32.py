#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

from gnuradio import gr_unittest
from gnuradio import mlir_aie


class qa_mlir_aie_cpp_tagged_int32_to_int32(gr_unittest.TestCase):

    def test_num_slots_must_be_positive(self):
        with self.assertRaises(ValueError):
            mlir_aie.mlir_aie_cpp_tagged_int32_to_int32(num_slots=0)


if __name__ == '__main__':
    gr_unittest.run(qa_mlir_aie_cpp_tagged_int32_to_int32)
