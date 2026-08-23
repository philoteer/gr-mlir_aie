//===- stage_add_const.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Stage 3: stateful complex constant add.
//
// This kernel is NOT memoryless: it keeps an internal phase counter, a 2-state
// state machine, that advances once per processed tile.  The sign of the added
// constant follows the phase:
//
//   phase 0: c = a + (0.5 + j0.25)
//   phase 1: c = a - (0.5 + j0.25)
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

void stage_add_const(cbfloat16 *a, cbfloat16 *c, int32_t N) {
  constexpr int vec_factor = 16;
  cbfloat16 *__restrict pA1 = a;
  cbfloat16 *__restrict pC1 = c;
  const int F = N / vec_factor;

  static int32_t phase = 0; // internal state machine: phase counter (mod 2)

  cbfloat16 off;
  if ((phase & 1) == 0) {
    off.real = (bfloat16)0.5f;
    off.imag = (bfloat16)0.25f;
  } else {
    off.real = (bfloat16)-0.5f;
    off.imag = (bfloat16)-0.25f;
  }
  phase = (phase + 1) & 1;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<cbfloat16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::vector<cbfloat16, vec_factor> C0 = aie::add(A0, off);
    aie::store_v(pC1, C0);
    pC1 += vec_factor;
  }
}

} // extern "C"