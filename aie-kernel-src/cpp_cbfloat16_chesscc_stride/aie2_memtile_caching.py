# section-3/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates

#example for memtile cacheing.
#see guide at: https://github.com/Xilinx/mlir-aie/blob/main/programming_guide/section-2/section-2b/03_Implicit_Copy/README.md
import numpy as np
import ml_dtypes
import sys
import inspect

from aie.iron import Kernel, ObjectFifo, Worker
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iron_compat import Program, Runtime, SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron.controlflow import range_
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

tensor_size_in = 4096
tensor_size_out = tensor_size_in
tile_size_in = tensor_size_in // 4
tile_size_out = tile_size_in
overlap_in = 0

stride_in= tile_size_in - overlap_in

# Define tensor types
# Input is complex (requires *2 for real/imag interleaved components)
in_tensor_ty = np.ndarray[(tensor_size_in*2,), np.dtype[ml_dtypes.bfloat16]]
in_tile_ty = np.ndarray[(tile_size_in*2,), np.dtype[ml_dtypes.bfloat16]]

out_tensor_ty = np.ndarray[(tensor_size_out*2,), np.dtype[ml_dtypes.bfloat16]]
out_tile_ty = np.ndarray[(tile_size_in*2,), np.dtype[ml_dtypes.bfloat16]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

# External, binary kernel definition
scale_fn = Kernel(
    "vector_scalar_mul_vector",
    "scale.o",
    [in_tile_ty, out_tile_ty, np.int32],
)

# Input data movement
# See https://github.com/Xilinx/mlir-aie/tree/main/programming_guide/section-2/section-2a for ObjectFIFO constructor
of_in = ObjectFifo(in_tile_ty, name="in")
in_dims =[(1, stride_in * 2), (4, 512), (512, 1)]

in_consumer = of_in.cons()
if "dims_to_stream" in inspect.signature(in_consumer.forward).parameters:
    of_in_mem = in_consumer.forward(name="in_mem", dims_to_stream=in_dims)
else:
    from aie.helpers.taplib import TensorAccessPattern
    in_pattern = TensorAccessPattern(
        [tile_size_in * 2], 0,
        sizes=[size for size, _ in in_dims],
        strides=[stride for _, stride in in_dims],
    )
    of_in_mem = in_consumer.forward(name="in_mem", to_stream=in_pattern)

# Output data movement
of_out = ObjectFifo(out_tile_ty, name="out")


# Task for the core to perform
def core_fn(of_in, of_out, scale_scalar):
    for _ in range_(4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        scale_scalar(elem_in, elem_out, tile_size_in)
        of_in.release(1)
        of_out.release(1)

# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in_mem.cons(), of_out.prod(), scale_fn])

# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty, in_tensor_ty) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

# Create the program from the device type and runtime
my_program = Program(dev, rt)

# Place components (assign them resources on the device) and generate an MLIR module
module = my_program.resolve_program(SequentialPlacer()) if SequentialPlacer else my_program.resolve_program()

# Print the generated MLIR
print(module)
