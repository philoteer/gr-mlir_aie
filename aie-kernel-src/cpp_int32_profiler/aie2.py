#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.placers import SequentialPlacer


PAYLOAD_WORDS = 1024
PROFILE_WORDS = 16


def build_program(device):
    input_type = np.ndarray[(PAYLOAD_WORDS,), np.dtype[np.int32]]
    output_type = np.ndarray[
        (PAYLOAD_WORDS + PROFILE_WORDS,), np.dtype[np.int32]
    ]

    input_fifo = ObjectFifo(input_type, name="input")
    output_fifo = ObjectFifo(output_type, name="output")
    kernel = Kernel(
        "profile_mul2",
        "profile_kernel.o",
        [input_type, output_type, np.int32],
    )

    def core(input_fifo, output_fifo, profile_mul2):
        input_tile = input_fifo.acquire(1)
        output_tile = output_fifo.acquire(1)
        profile_mul2(input_tile, output_tile, PAYLOAD_WORDS)
        input_fifo.release(1)
        output_fifo.release(1)

    worker = Worker(core, [input_fifo.cons(), output_fifo.prod(), kernel])
    runtime = Runtime()
    with runtime.sequence(input_type, output_type) as (host_input, host_output):
        runtime.start(worker)
        runtime.fill(input_fifo.prod(), host_input)
        runtime.drain(output_fifo.cons(), host_output, wait=True)

    return Program(device, runtime).resolve_program(SequentialPlacer())


def main():
    parser = argparse.ArgumentParser(description="Generate the profiler example MLIR")
    parser.add_argument("device", choices=("npu", "npu2"), nargs="?", default="npu")
    args = parser.parse_args()
    device = NPU1Col1() if args.device == "npu" else NPU2Col1()
    print(build_program(device))


if __name__ == "__main__":
    main()
