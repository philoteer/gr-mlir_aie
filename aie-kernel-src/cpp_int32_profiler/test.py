#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.utils import DefaultNPURuntime, NPUKernel

from profile_record import PAYLOAD_WORDS, PROFILE_WORDS, expected_checksum, parse_profile


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Test the AIE cycle-profiler example")
    parser.add_argument("--xclbin", type=Path, default=directory / "build/final.xclbin")
    parser.add_argument("--insts", type=Path, default=directory / "build/insts.bin")
    args = parser.parse_args()

    payload = np.arange(-PAYLOAD_WORDS // 2, PAYLOAD_WORDS // 2, dtype=np.int32)
    output = iron.zeros(PAYLOAD_WORDS + PROFILE_WORDS, dtype=np.int32)
    kernel = NPUKernel(str(args.xclbin), str(args.insts), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(kernel)
    DefaultNPURuntime.run(handle, [iron.tensor(payload, dtype=np.int32), output])

    result = output.numpy()
    np.testing.assert_array_equal(result[:PAYLOAD_WORDS], payload * 2)
    profile = parse_profile(result)
    if profile["checksum"] != expected_checksum(payload):
        raise AssertionError("kernel checksum does not match the output payload")
    if any(profile[name] == 0 for name in ("total_cycles", "vector_cycles", "scalar_cycles")):
        raise AssertionError("expected non-zero cycle counters")
    print("PASS: payload and profile record are valid")


if __name__ == "__main__":
    main()
