//===- stage_rotate90.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Stage 2: stateful +90-degree rotation.
//
// Like the other stages this kernel is NOT memoryless: it keeps an internal
// phase counter, a 3-state state machine, that advances once per processed
// tile.  The rotation angle follows the phase:
//
//   phase 0: c = j^1 * a = j * a   (+90 deg)
//   phase 1: c = j^2 * a = -a      (+180 deg)
//   phase 2: c = j^3 * a = -j * a  (+270 deg)
//
// The phase is kept in a `static` variable, surviving across kernel calls for
// as long as the core keeps running (see stage_scale_mul.cc for the full
// rationale).  A host bug that resets/reconfigures the NPU while the kernels
// are in execution restarts this phase at 0, making the deviation from the
// continuous reference immediately visible.

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

extern "C" {

void stage_rotate90(cbfloat16 *a, cbfloat16 *c, int32_t N) {
  constexpr int vec_factor = 16;
  cbfloat16 *__restrict pA1 = a;
  cbfloat16 *__restrict pC1 = c;
  const int F = N / vec_factor;

  static int32_t phase = 0; // internal state machine: phase counter (mod 3)

  cbfloat16 fac;
  switch (phase % 3) {
  case 0:
    fac.real = (bfloat16)0.0f;
    fac.imag = (bfloat16)1.0f;
    break;
  case 1:
    fac.real = (bfloat16)-1.0f;
    fac.imag = (bfloat16)0.0f;
    break;
  default: // phase % 3 == 2
    fac.real = (bfloat16)0.0f;
    fac.imag = (bfloat16)-1.0f;
    break;
  }
  phase = (phase + 1) % 3;

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