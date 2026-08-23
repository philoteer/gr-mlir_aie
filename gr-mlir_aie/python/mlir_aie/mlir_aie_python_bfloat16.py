#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 gr-mlir_aie author.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#


import numpy as np
from gnuradio import gr

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx
from aie.iron import Program, Runtime, Worker, ObjectFifo
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

import aie.iron as iron
from aie.utils import NPUKernel, DefaultNPURuntime, get_current_device

import ml_dtypes

DEV = iron.get_current_device()
DTYPE_GR=np.float32
DTYPE_NPU=ml_dtypes.bfloat16

class mlir_aie_python_bfloat16(gr.basic_block):
    """
    docstring for block mlir_aie_python_bfloat16
    """
    def __init__(self, path_xclbin="aie-kernel-src/build/final.xclbin",path_insts_bin="aie-kernel-src/build/insts.bin", VECTOR_SIZE = 8192):
        gr.basic_block.__init__(self,
            name="mlir_aie_python_bfloat16",
            in_sig=[DTYPE_GR, ],
            out_sig=[DTYPE_GR, ])

        npu_kernel = NPUKernel(
            path_xclbin,
            path_insts_bin,
            kernel_name="MLIR_AIE",
        )
         
        self.kernel_handle = DefaultNPURuntime.load(npu_kernel)
        self.out_buf = iron.zeros(VECTOR_SIZE, dtype=DTYPE_NPU)
        self.VECTOR_SIZE = VECTOR_SIZE

    #TODO implement forecast()
    def forecast(self, noutput_items, ninputs):
        #ninput_items_required = [min(self.VECTOR_SIZE, noutput_items)] * ninputs
        ninput_items_required = [noutput_items] * ninputs   
        return ninput_items_required

    def general_work(self, input_items, output_items):
        in0 = input_items[0]
        out0 = output_items[0]
        
        n_input_items = len(in0)
        n_output_items = len(out0)
        
        available_items = min(n_input_items, n_output_items)
        n_chunks = available_items // self.VECTOR_SIZE

        if n_chunks == 0:
            return 0

        for i in range(n_chunks):
            start_idx = i * self.VECTOR_SIZE
            end_idx = start_idx + self.VECTOR_SIZE
            
            current_in_tensor = iron.tensor(in0[start_idx:end_idx].astype(DTYPE_NPU), dtype=DTYPE_NPU)
            buffers = [current_in_tensor, self.out_buf] 
            DefaultNPURuntime.run(self.kernel_handle, buffers)                    
            out0[start_idx:end_idx] = self.out_buf.numpy().astype(DTYPE_GR)
        
        total_processed = n_chunks * self.VECTOR_SIZE        
        self.consume_each(total_processed)
        
        return total_processed
