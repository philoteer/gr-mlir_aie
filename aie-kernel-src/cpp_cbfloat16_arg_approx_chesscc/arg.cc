//===- arg.cc ---------------------------------------------------*- C++ -*-===//
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

#define PI 3.14159265358979323846f
#define HALF_PI 1.57079632679489661923f
#define QUARTER_PI 0.78539816339744830962f
#define TAN_PI_8 0.41421356237309504880f

static inline float absf_approx(float x) { return x < 0.0f ? -x : x; }

static inline float atan_small_approx(float x) {
  const float x2 = x * x;
  return x *
         (1.0f +
          x2 * (-0.33333333333333333333f +
                x2 * (0.2f +
                      x2 * (-0.14285714285714285714f +
                            x2 * (0.11111111111111111111f +
                                  x2 * (-0.09090909090909090909f +
                                        x2 * (0.07692307692307692308f +
                                              x2 * (-0.06666666666666666667f +
                                                    x2 * 0.05882352941176470588f))))))));
}

static inline float atan_positive_approx(float x) {
  if (x <= TAN_PI_8)
    return atan_small_approx(x);

  const float reduced = (x - 1.0f) / (x + 1.0f);
  return QUARTER_PI + atan_small_approx(reduced);
}

static inline float atan2_approx(float y, float x) {
  if (x == 0.0f) {
    if (y > 0.0f)
      return HALF_PI;
    if (y < 0.0f)
      return -HALF_PI;
    return 0.0f;
  }

  const float ay = absf_approx(y);
  const float ax = absf_approx(x);
  float angle;

  if (ay <= ax)
    angle = atan_positive_approx(ay / ax);
  else
    angle = HALF_PI - atan_positive_approx(ax / ay);

  if (x < 0.0f)
    angle = PI - angle;
  if (y < 0.0f)
    angle = -angle;

  return angle;
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
    pC[i] = (bfloat16)atan2_approx(imag, real);
  }
}

} // extern "C"
