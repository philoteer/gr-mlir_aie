//===- arg.cc ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Derived from mlir-aie example structure and adapted for this project.
//
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

static inline float atan2_estimate(float y, float x) {
    if (x == 0.0f && y == 0.0f) {
        return 0.0f;
    }

    const float pi_4 = 0.78539816339744830962f;
    const float three_pi_4 = 2.35619449019234492885f;
    float abs_y = y < 0.0f ? -y : y;
    float angle;

    if (x < 0.0f) {
        float r = (x + abs_y) / (abs_y - x);
        const float r2 = r * r;
        angle = three_pi_4 + ((-0.0775571848f * r2 + 0.2873136883f) * r2 - 0.9951546670f) * r;
    } else {
        float r = (x - abs_y) / (x + abs_y);
        const float r2 = r * r;
        angle = pi_4 + ((-0.0775571848f * r2 + 0.2873136883f) * r2 - 0.9951546670f) * r;
    }

    return y < 0.0f ? -angle : angle;
}

extern "C" {

void complex_to_arg(cbfloat16 *a, bfloat16 *c, int32_t N) {
  cbfloat16 *__restrict pA = a;
  bfloat16 *__restrict pC = c;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int32_t i = 0; i < N; i++) {
    const float real = (float)pA[i].real;
    const float imag = (float)pA[i].imag;
    //pC[i] = (bfloat16)atan2f(imag, real);
    pC[i] = (bfloat16)atan2_estimate(imag, real);
  }
}

} // extern "C"
