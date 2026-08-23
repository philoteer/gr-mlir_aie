#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr
import pmt

import aie.iron as iron
from aie.utils import DefaultNPURuntime, NPUKernel


INPUT_DTYPE = np.int32
OUTPUT_DTYPE = np.int32
METADATA_DTYPE = np.int32

MAX_TAGS_PER_TILE = 7
N_TILES = 4
METADATA_WORDS_PER_TILE = 2 + 2 * MAX_TAGS_PER_TILE


class mlir_aie_python_tagged_int32_to_int32(gr.basic_block):
    """
    Run an MLIR-AIE int32 kernel and recreate its output stream tags.
    """
    def __init__(self,
                 path_xclbin="aie-kernel-src/build/final.xclbin",
                 path_insts_bin="aie-kernel-src/build/insts.bin",
                 kernel_name="MLIR_AIE",
                 VECTOR_SIZE=4096):
        gr.basic_block.__init__(self,
            name="mlir_aie_python_tagged_int32_to_int32",
            in_sig=[INPUT_DTYPE],
            out_sig=[OUTPUT_DTYPE])

        npu_kernel = NPUKernel(
            path_xclbin,
            path_insts_bin,
            kernel_name=kernel_name,
        )
        self.kernel_handle = DefaultNPURuntime.load(npu_kernel)
        self.out_buf = iron.zeros(VECTOR_SIZE, dtype=OUTPUT_DTYPE)
        self.out_meta_buf = iron.zeros(
            N_TILES * METADATA_WORDS_PER_TILE, dtype=METADATA_DTYPE)
        self.VECTOR_SIZE = VECTOR_SIZE
        self.TILE_SIZE = VECTOR_SIZE // N_TILES

        self.set_tag_propagation_policy(gr.TPP_DONT)
        self.tag_key = pmt.intern("wifi_start")
        self.tag_srcid = pmt.intern("sync_short")

    def forecast(self, noutput_items, ninputs):
        ninput_items_required = [noutput_items] * ninputs
        return ninput_items_required

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        n_chunks = min(len(in0), len(out0)) // self.VECTOR_SIZE

        if n_chunks == 0:
            return 0

        output_abs_start = self.nitems_written(0)
        total_produced = 0

        for chunk_idx in range(n_chunks):
            input_start = chunk_idx * self.VECTOR_SIZE
            input_end = input_start + self.VECTOR_SIZE
            current_in_tensor = iron.tensor(
                in0[input_start:input_end], dtype=INPUT_DTYPE)

            DefaultNPURuntime.run(
                self.kernel_handle,
                [current_in_tensor, self.out_buf, self.out_meta_buf],
            )

            npu_output = self.out_buf.numpy()
            metadata = self.out_meta_buf.numpy()

            for tile_idx in range(N_TILES):
                meta_start = tile_idx * METADATA_WORDS_PER_TILE
                tile_meta = metadata[
                    meta_start:meta_start + METADATA_WORDS_PER_TILE]
                tile_len = max(0, min(int(tile_meta[0]), self.TILE_SIZE))
                tag_count = int(tile_meta[1])
                tile_start = tile_idx * self.TILE_SIZE

                out0[total_produced:total_produced + tile_len] = npu_output[
                    tile_start:tile_start + tile_len]

                tile_abs_start = output_abs_start + total_produced
                for tag_idx in range(tag_count):
                    tag_offset = int(tile_meta[2 + 2 * tag_idx])
                    tag_value = int(tile_meta[3 + 2 * tag_idx]) / float(1 << 29)
                    if 0 <= tag_offset < tile_len:
                        self.add_item_tag(
                            0,
                            tile_abs_start + tag_offset,
                            self.tag_key,
                            pmt.from_double(tag_value),
                            self.tag_srcid,
                        )

                total_produced += tile_len

        self.consume_each(n_chunks * self.VECTOR_SIZE)
        return total_produced
