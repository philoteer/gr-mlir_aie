# aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

# Multi-kernel chain example: three compute cores connected in series.
#
#   ShimDMA -> [core0: scale by 2+j] -> [core1: rotate 90deg] -> [core2: add 0.5+j0.25] -> ShimDMA
#
# Each core does real vector math on cbfloat16 data (no passthrough), so a
# mis-wired pipeline is easy to catch:
#
#   y = ((2 + j1)x)*j + (0.5 + j0.25) = (-1 + j2)x + (0.5 + j0.25)
#
# PIPELINE BUBBLE DEMONSTRATION
# -----------------------------
# The inter-core ObjectFifo depth is the "are we feeding the pipeline?"
# knob:
#
#   FIFO_DEPTH=1 : single-buffered handoff between cores. While core N is
#                  computing on a buffer it also blocks core N-1 from filling
#                  the next one, so all stages run serialized per tile and the
#                  idle gaps ("bubbles") propagate down the chain.
#
#   FIFO_DEPTH>=2: double-buffered handoff. Core N-1 fills buffer i+1 while
#                  core N still computes on buffer i, so stages overlap and
#                  bubbles disappear.
#
# Build both variants and compare run time of test.py:
#   make FIFO_DEPTH=1        # bubbled pipeline (default)
#   make FIFO_DEPTH=2        # properly fed pipeline
import os
import sys

import ml_dtypes
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
import aie.iron as iron

if len(sys.argv) > 1:
    if sys.argv[1] == "npu":
        dev = NPU1()
    elif sys.argv[1] == "npu2":
        dev = NPU2()
    else:
        raise ValueError(f"Unsupported device: {sys.argv[1]}")
else:
    dev = iron.get_current_device()

NUM_CORES = 3
TILE_ITERATIONS = 4
FIFO_DEPTH = int(os.environ.get("FIFO_DEPTH", "1"))
if FIFO_DEPTH < 1:
    raise ValueError("FIFO_DEPTH must be >= 1")

tensor_size = 4096 # complex samples per tensor
tile_size = tensor_size // TILE_ITERATIONS # complex samples per tile

# Define tensor types (complex data: real/imag interleaved bfloat16 => *2)
tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[ml_dtypes.bfloat16]]
tile_ty = np.ndarray[(tile_size * 2,), np.dtype[ml_dtypes.bfloat16]]

# External, binary kernel definitions (one .o per stage, one per core).
#
# Every kernel is stateful (see the individual stage_*.cc files): it keeps an
# internal phase counter that advances once per processed tile and modulates
# the computation.  The kernels are NOT memoryless - the output of a tile
# depends on how many tiles were fed to the core since it was (re)started.
stage_scale_mul_fn = Kernel(
    "stage_scale_mul",
    "stage_scale_mul.o",
    [tile_ty, tile_ty, np.int32],
)
stage_rotate90_fn = Kernel(
    "stage_rotate90",
    "stage_rotate90.o",
    [tile_ty, tile_ty, np.int32],
)
stage_add_const_fn = Kernel(
    "stage_add_const",
    "stage_add_const.o",
    [tile_ty, tile_ty, np.int32],
)

# Input data movement
of_in = ObjectFifo(tile_ty, name="in")

# Inter-stage data movement. Depth is deliberately exposed as the bubble knob;
# see header comment.
of_stage1 = ObjectFifo(tile_ty, name="stage1", depth=FIFO_DEPTH)
of_stage2 = ObjectFifo(tile_ty, name="stage2", depth=FIFO_DEPTH)

# Output data movement
of_out = ObjectFifo(tile_ty, name="out")


# Task performed by every core in the chain
def core_fn(of_in, of_out, stage_kernel):
    for _ in range_(TILE_ITERATIONS):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        stage_kernel(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


# One worker per compute core, chained via the inter-stage object fifos
worker0 = Worker(core_fn, [of_in.cons(), of_stage1.prod(), stage_scale_mul_fn])
worker1 = Worker(core_fn, [of_stage1.cons(), of_stage2.prod(), stage_rotate90_fn])
worker2 = Worker(core_fn, [of_stage2.cons(), of_out.prod(), stage_add_const_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_out, _):
    rt.start(worker0)
    rt.start(worker1)
    rt.start(worker2)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
