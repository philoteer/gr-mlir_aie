# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import aie.iron as iron
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
from aie.iron.placers import SequentialPlacer


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

# Complex values are interleaved real/imaginary components at the IRON boundary.
in_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int16]]
in_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int16]]
out_tensor_ty = np.ndarray[(tensor_size * 2,), np.dtype[np.int32]]
out_tile_ty = np.ndarray[(tile_size * 2,), np.dtype[np.int32]]

fft_fn = Kernel("fft64_cint16", "fft64.o", [in_tile_ty, out_tile_ty, np.int32])

of_in = ObjectFifo(in_tile_ty, name="in")
of_out = ObjectFifo(out_tile_ty, name="out")


def core_fn(of_in, of_out, fft):
    for _ in range_(tensor_size // tile_size):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        fft(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


worker = Worker(core_fn, [of_in.cons(), of_out.prod(), fft_fn])

runtime = Runtime()
with runtime.sequence(in_tensor_ty, out_tensor_ty, out_tensor_ty) as (
    input_tensor,
    output_tensor,
    _,
):
    runtime.start(worker)
    runtime.fill(of_in.prod(), input_tensor)
    runtime.drain(of_out.cons(), output_tensor, wait=True)

program = Program(dev, runtime)
print(program.resolve_program(SequentialPlacer()))
