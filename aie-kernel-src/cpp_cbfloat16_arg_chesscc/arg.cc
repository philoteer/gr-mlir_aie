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

extern "C" {

void complex_to_arg(cbfloat16 *a, bfloat16 *c, int32_t N) {
  cbfloat16 *__restrict pA = a;
  bfloat16 *__restrict pC = c;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int32_t i = 0; i < N; i++) {
    const float real = (float)pA[i].real;
    const float imag = (float)pA[i].imag;
    pC[i] = (bfloat16)atan2f(imag, real);
  }
}

} // extern "C"
