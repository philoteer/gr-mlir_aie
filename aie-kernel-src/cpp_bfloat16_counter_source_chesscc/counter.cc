//===- counter.cc -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Derived from mlir-aie example structure and adapted for this project.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

static float counter_value = 0.0f;

extern "C" {

void counter_source(float *step, bfloat16 *out, int32_t N) {
  const float step_value = step[0];  

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int32_t i = 0; i < N; i++) {
    out[i] = (bfloat16)counter_value;
    counter_value += step_value;
  }
}

} // extern "C"
