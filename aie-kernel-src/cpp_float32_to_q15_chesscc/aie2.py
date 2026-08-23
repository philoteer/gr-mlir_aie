# section-3/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
from aie.iron.placers import SequentialPlacer
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

in_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.float32]]
in_tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
out_tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int16]]
out_tile_ty = np.ndarray[(tile_size,), np.dtype[np.int16]]

convert_fn = Kernel("float32_to_q15", "convert.o", [in_tile_ty, out_tile_ty, np.int32])

of_in = ObjectFifo(in_tile_ty, name="in")
of_out = ObjectFifo(out_tile_ty, name="out")


def core_fn(of_in, of_out, convert):
    for _ in range_(4):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        convert(elem_in, elem_out, tile_size)
        of_in.release(1)
        of_out.release(1)


my_worker = Worker(core_fn, [of_in.cons(), of_out.prod(), convert_fn])

rt = Runtime()
with rt.sequence(in_tensor_ty, out_tensor_ty, out_tensor_ty) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)

my_program = Program(dev, rt)
module = my_program.resolve_program(SequentialPlacer())
print(module)
