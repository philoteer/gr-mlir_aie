//===- freq_source.cc -------------------------------------------*- C++ -*-===//
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

static float oscillator_real = 1.0f;
static float oscillator_imag = 0.0f;

extern "C" {

void frequency_source(float *frequency, cbfloat16 *out, int32_t N) {
  
  float phase_step = 6.28318530717958647692f * frequency[0];
  const float step_real = cosf(phase_step);
  const float step_imag = sinf(phase_step);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int32_t i = 0; i < N; i++) {
    out[i].real = (bfloat16)oscillator_real;
    out[i].imag = (bfloat16)oscillator_imag;

    const float next_real =
        oscillator_real * step_real - oscillator_imag * step_imag;
    const float next_imag =
        oscillator_real * step_imag + oscillator_imag * step_real;
    oscillator_real = next_real;
    oscillator_imag = next_imag;
  }
}

} // extern "C"
