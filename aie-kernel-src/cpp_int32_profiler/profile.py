#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.utils import DefaultNPURuntime, NPUKernel

from profile_record import (
    CYCLE_FIELDS,
    PAYLOAD_WORDS,
    PROFILE_WORDS,
    expected_checksum,
    format_report,
    parse_profile,
)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Aggregate AIE kernel cycle counters")
    parser.add_argument("--xclbin", type=Path, default=directory / "build/final.xclbin")
    parser.add_argument("--insts", type=Path, default=directory / "build/insts.bin")
    parser.add_argument("--runs", type=int, default=100, help="number of kernel invocations")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    rng = np.random.default_rng(1)
    output = iron.zeros(PAYLOAD_WORDS + PROFILE_WORDS, dtype=np.int32)
    kernel = NPUKernel(str(args.xclbin), str(args.insts), kernel_name="MLIR_AIE")
    handle = DefaultNPURuntime.load(kernel)
    totals = {name: 0 for name in CYCLE_FIELDS}
    totals["run_count"] = 0

    for _ in range(args.runs):
        payload = rng.integers(-10000, 10001, PAYLOAD_WORDS, dtype=np.int32)
        DefaultNPURuntime.run(
            handle, [iron.tensor(payload, dtype=np.int32), output]
        )
        profile = parse_profile(output.numpy())
        if profile["checksum"] != expected_checksum(payload):
            raise AssertionError("profile checksum mismatch")
        for name in CYCLE_FIELDS:
            totals[name] += int(profile[name])
        totals["run_count"] += int(profile["run_count"])

    print(format_report(totals))


if __name__ == "__main__":
    main()
