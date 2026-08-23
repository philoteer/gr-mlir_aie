# section-3/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Derived from mlir-aie example structure and adapted for this project.
import numpy as np
import ml_dtypes
import sys

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

tensor_size = 4096
tile_size = tensor_size // 4

# cbfloat16 is represented as interleaved bfloat16 real/imag values at the IRON boundary.
tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[ml_dtypes.bfloat16]]
tile_ty = np.ndarray[(tile_size * 2,), np.dtype[ml_dtypes.bfloat16]]

# External, binary kernel definition
freq_shift_fn = Kernel(
    "frequency_shift",
    "freq_shift.o",
    [tile_ty, tile_ty, np.int32],
)

# Input data movement
of_in = ObjectFifo(tile_ty, name="in")

# Output data movement
of_out = ObjectFifo(tile_ty, name="out")


# Task for the core to perform
def core_fn(of_in, of_out, freq_shift):
    for _ in range_(4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        freq_shift(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), freq_shift_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)


# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer())

# Print the generated MLIR
print(module)
