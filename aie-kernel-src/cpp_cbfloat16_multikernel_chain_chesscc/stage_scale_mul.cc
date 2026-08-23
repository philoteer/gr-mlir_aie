//===- stage_scale_mul.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Stage 1: stateful complex scalar multiply.
//
// This kernel is NOT memoryless.  It maintains an internal phase counter, a
// 4-state state machine, that advances by one every time a tile is processed.
// The complex multiplier is rotated by 90 degrees as the state advances:
//
//   phase 0: c = (2 + j1)       * a
//   phase 1: c = (2 + j1) * j   * a = (-1 + j2) * a
//   phase 2: c = (2 + j1) * j^2 * a = -(2 + j1) * a
//   phase 3: c = (2 + j1) * j^3 * a = (1 - j2) * a
//
// The phase lives in a `static` variable, so it survives across kernel calls
// for as long as the core keeps running.  The core is started once and then
// loops forever (while_true in aie2.py), so the phase also survives across
// host invocations: the output of a tile depends on how many tiles have been
// fed to the core since it was (re)started.
//
// Debugging value for host code:
//   * a host that tracks the phase can verify the stream tile-by-tile;
//   * a host that resets/reconfigures the NPU while the pipeline is in
//     execution restarts the phase at 0, and the outputs stop matching the
//     continuous reference -> the mistake becomes visible.

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

extern "C" {

void stage_scale_mul(cbfloat16 *a, cbfloat16 *c, int32_t N) {
  constexpr int vec_factor = 16;
  cbfloat16 *__restrict pA1 = a;
  cbfloat16 *__restrict pC1 = c;
  const int F = N / vec_factor;

  static int32_t phase = 0; // internal state machine: phase counter (mod 4)

  cbfloat16 fac;
  switch (phase & 3) {
  case 0:
    fac.real = (bfloat16)2.0f;
    fac.imag = (bfloat16)1.0f;
    break;
  case 1:
    fac.real = (bfloat16)-1.0f;
    fac.imag = (bfloat16)2.0f;
    break;
  case 2:
    fac.real = (bfloat16)-2.0f;
    fac.imag = (bfloat16)-1.0f;
    break;
  default: // phase & 3 == 3
    fac.real = (bfloat16)1.0f;
    fac.imag = (bfloat16)-2.0f;
    break;
  }
  phase = (phase + 1) & 3;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<cbfloat16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::accum<caccfloat, vec_factor> cout = aie::mul(A0, fac);
    aie::store_v(pC1, cout.template to_vector<cbfloat16>(0));
    pC1 += vec_factor;
  }
}

} // extern "C"