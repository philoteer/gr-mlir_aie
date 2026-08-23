#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Host-side correctness + debugging test for the stateful multi-kernel chain.
#
# The three stage kernels are NO LONGER memoryless: each keeps an internal
# phase counter (a small state machine) that advances once per processed tile
# and modulates the computation (see stage_scale_mul.cc / stage_rotate90.cc /
# stage_add_const.cc).  Because the cores run forever (while_true in aie2.py),
# the internal phase persists across host invocations: the correct output of a
# tile depends on the GLOBAL tile index since the cores were last (re)loaded.
#
# For global tile g the chain computes:
#   y = ((x * (2+j1) * j^(g%4)) * j^((g%3)+1)) + (0.5+j0.25) * (-1)^(g%2)
#
# This makes the design useful for debugging host code:
#   * a correct host keeps feeding tiles and tracks the global tile index g ->
#     outputs match the reference below;
#   * a host bug that resets/reloads the NPU kernels mid-stream (while the
#     kernels are in execution) restarts all internal phases at 0; if the host
#     keeps using its own continuous g, the outputs no longer match -> the
#     bug becomes immediately noticeable.
#
# Modes:
#   python3 test.py [iters]             verify continuous host tracking + time
#   python3 test.py --break-state       also demonstrate that resetting the
#                                       kernels mid-stream is detectable

import os
import shutil
import sys
import tempfile
import time

import ml_dtypes
import numpy as np

from aie.utils import NPUKernel, tensor

TENSOR_SIZE = 4096 # complex samples per tensor (matches aie2.py)
VEC_SIZE = TENSOR_SIZE * 2 # bf16 elements, real/imag interleaved
TILE_ITERATIONS = 4 # tiles per tensor (matches aie2.py)
TILE_SIZE = TENSOR_SIZE // TILE_ITERATIONS

TOL = 0.02 # loose enough for 3 stages of bfloat16 rounding


def complex_to_cbfloat16(x):
    """Interleave re/im and round each component to bfloat16."""
    out = np.empty(VEC_SIZE, dtype=ml_dtypes.bfloat16)
    out[0::2] = np.real(x).astype(ml_dtypes.bfloat16)
    out[1::2] = np.imag(x).astype(ml_dtypes.bfloat16)
    return out


def cbfloat16_to_complex(b):
    f = np.asarray(b, dtype=np.float32)
    return f[0::2] + 1j * f[1::2]


def reference_tile(x, g):
    """Numpy model of the whole chain for ONE tile with global index g.

    g is the number of tiles fed to the cores since they were (re)loaded.
    """
    # stage_scale_mul : y = (2+j1) * j^(g%4) * x
    y = x * (2 + 1j) * (1j ** (g % 4))
    # stage_rotate90  : y = j^((g%3)+1) * y   (j, -1, -j cycling)
    y = y * (1j ** ((g % 3) + 1))
    # stage_add_const : y = y + (0.5+j0.25) * (-1)^(g%2)
    y = y + (0.5 + 0.25j) * ((-1) ** (g % 2))
    return y


def reference(x, g0):
    """Numpy model for a whole tensor; its first tile has global index g0."""
    out = np.empty(TENSOR_SIZE, dtype=complex)
    for t in range(TILE_ITERATIONS):
        sl = slice(t * TILE_SIZE, (t + 1) * TILE_SIZE)
        out[sl] = reference_tile(x[sl], g0 + t)
    return out


def run_one(k, in_t, out_t):
    k(in_t, out_t)
    return cbfloat16_to_complex(np.array(out_t))


def load_fresh(xclbin, insts):
    """Load the kernel from a private copy so the cores (and their internal
    phases) are guaranteed to start from a known state."""
    td = tempfile.mkdtemp(prefix="mlir_aie_chain_")
    xclbin2 = os.path.join(td, "final.xclbin")
    insts2 = os.path.join(td, "insts.bin")
    shutil.copy(xclbin, xclbin2)
    shutil.copy(insts, insts2)
    return NPUKernel(xclbin2, insts2, kernel_name="MLIR_AIE"), td


def main():
    here = os.path.dirname(os.path.realpath(__file__))
    xclbin = os.path.join(here, "build", "final.xclbin")
    insts = os.path.join(here, "build", "insts.bin")
    if not (os.path.exists(xclbin) and os.path.exists(insts)):
        sys.exit("build/final.xclbin or build/insts.bin missing - run make first")

    break_state = "--break-state" in sys.argv
    iters = 20
    for a in sys.argv[1:]:
        if a.isdigit():
            iters = int(a)

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, TENSOR_SIZE) + 1j * rng.uniform(-1.0, 1.0, TENSOR_SIZE)
    in_np = complex_to_cbfloat16(x)
    out_np = np.zeros(VEC_SIZE, dtype=ml_dtypes.bfloat16)

    temp_dirs = []

    # Load fresh so the internal phase counters start at 0 (g starts at 0).
    k, td = load_fresh(xclbin, insts)
    temp_dirs.append(td)
    in_t = tensor(in_np, dtype=ml_dtypes.bfloat16)
    out_t = tensor(out_np, dtype=ml_dtypes.bfloat16)

    # ------------------------------------------------------------------
    # 1. Correct host: feed a continuous stream and track the global tile
    #    index g; every invocation must match the state-tracking reference.
    # ------------------------------------------------------------------
    g = 0
    num_inv = 3
    worst = 0.0
    for _ in range(num_inv):
        got = run_one(k, in_t, out_t)
        exp = reference(cbfloat16_to_complex(in_np), g)
        err = np.max(np.abs(got - exp))
        worst = max(worst, err)
        g += TILE_ITERATIONS
        print(f"  invocation g={g - TILE_ITERATIONS:2d}..{g - 1:2d}: max err {err:.5f}")

    print(f"continuous host tracking: max err over {num_inv} invocations = "
          f"{worst:.5f} (tolerance {TOL})")
    if worst > TOL:
        sys.exit("FAILED: output does not match the state-tracking reference")
    print("PASS: stateful chain matches reference while host tracks g")

    # ------------------------------------------------------------------
    # 2. Debugging demo: a host bug that resets the kernels mid-stream is
    #    noticeable.  We keep the host's g running but reload the kernels
    #    (as a buggy host would), which restarts all internal phases at 0.
    #    The composite state period is lcm(4,3,2)=12 tiles, so we break at
    #    a g that is NOT a multiple of 12 to make the two states distinct.
    # ------------------------------------------------------------------
    if break_state:
        # Bump the (host-side) g until it lands on a distinct phase.
        while (g % 4, g % 3, g % 2) == (0, 0, 0):
            g += TILE_ITERATIONS

        print(f"\n--break-state: simulating a host that resets the kernels "
              f"mid-stream while keeping its own g running...")
        k2, td2 = load_fresh(xclbin, insts)  # fresh cores: phases back to 0
        temp_dirs.append(td2)
        got = run_one(k2, in_t, out_t)

        exp_continuous = reference(cbfloat16_to_complex(in_np), g)  # host's g
        exp_reset = reference(cbfloat16_to_complex(in_np), 0)       # actual state
        err_cont = np.max(np.abs(got - exp_continuous))
        err_reset = np.max(np.abs(got - exp_reset))
        print(f"  host expects g={g:2d} (continuous):    max err {err_cont:.5f}")
        print(f"  actual kernels at g=0 (reset):        max err {err_reset:.5f}")
        if err_reset <= TOL and err_cont > TOL:
            print("DETECTED: kernels were reset mid-stream (phases restarted at 0) - "
                  "a buggy host that kept counting g would see this mismatch.")
        else:
            print("NOTE: state reset not distinguishable with this data; "
                  "expected err_reset small and err_cont large.")

    # ------------------------------------------------------------------
    # 3. Timing (bubble comparison): FIFO_DEPTH=1 vs FIFO_DEPTH=2 builds.
    #    Uses the same already-loaded kernel; state keeps advancing, which
    #    does not affect wall-clock time.
    # ------------------------------------------------------------------
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        k(in_t, out_t)
        _ = np.array(out_t)
        times.append(time.perf_counter() - t0)
    med_ms = np.median(times) * 1e3
    print(f"median over {iters} runs: {med_ms:.3f} ms/run "
          f"({TENSOR_SIZE / (med_ms * 1e-3) / 1e6:.3f} Msps)")

    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()