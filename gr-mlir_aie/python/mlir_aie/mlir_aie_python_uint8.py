#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

#TODO0: generalize better
#TODO1: make vector size adjustable
#TODO2: implement forecast() (currently left as 1:1)

import numpy
from gnuradio import gr

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime, get_current_device

DEV = iron.get_current_device()
DTYPE=np.uint8

class mlir_aie_python_uint8(gr.basic_block):
    """
    You know the rules and so do I 
    """
    def __init__(self, path_xclbin="aie-kernel-src/build/final.xclbin",path_insts_bin="aie-kernel-src/build/insts.bin", VECTOR_SIZE = 16384):
        gr.basic_block.__init__(self,
            name="mlir_aie_python_uint8",
            in_sig=[DTYPE],
            out_sig=[DTYPE])
            
        npu_kernel = NPUKernel(
            path_xclbin,
            path_insts_bin,
            kernel_name="MLIR_AIE",
        )
         
        self.kernel_handle = DefaultNPURuntime.load(npu_kernel)
        self.out_buf = iron.zeros(VECTOR_SIZE, dtype=DTYPE)
        self.VECTOR_SIZE = VECTOR_SIZE

    #TODO implement forecast()
    def forecast(self, noutput_items, ninputs):
        #ninput_items_required = [min(self.VECTOR_SIZE, noutput_items)] * ninputs
        ninput_items_required = [noutput_items] * ninputs   
        return ninput_items_required

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        
        # Calculate how many full chunks fit in the available input and output buffers
        n_input_items = len(in0)
        n_output_items = len(out0)
        
        # Available items logic
        available_items = min(n_input_items, n_output_items)
        n_chunks = available_items // self.VECTOR_SIZE

        # If we don't have enough data for even a single chunk, return 0 to wait for more
        if n_chunks == 0:
            return 0

        # Process multiple chunks
        for i in range(n_chunks):
            start_idx = i * self.VECTOR_SIZE
            end_idx = start_idx + self.VECTOR_SIZE
            
            # Slice the current chunk from input
            # iron.tensor creation might involve data movement depending on backend implementation
            # Here we wrap the numpy slice into an iron tensor
            current_in_tensor = iron.tensor(in0[start_idx:end_idx], dtype=DTYPE)
            
            # Prepare buffers list: [Input, Output]
            # reusing self.out_buf to minimize allocation overhead
            buffers = [current_in_tensor, self.out_buf] 
            
            # Execute the kernel
            DefaultNPURuntime.run(self.kernel_handle, buffers)        
            
            # Copy the result from NPU buffer back to GNU Radio output buffer
            # We assume self.out_buf is updated in place or handled by the runtime
            out0[start_idx:end_idx] = self.out_buf.numpy()
        
        # Calculate total items processed
        total_processed = n_chunks * self.VECTOR_SIZE
        
        # Tell the scheduler how many items were consumed from input
        self.consume_each(total_processed)
        
        # Return the number of items produced
        return total_processed
