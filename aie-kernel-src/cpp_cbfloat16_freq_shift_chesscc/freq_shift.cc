//===- freq_shift.cc --------------------------------------------*- C++ -*-===//
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

namespace {

constexpr int kVecFactor = 16;

// Example shift: exp(i * 2*pi*37*n/512), i.e. +37 bins over a 512-sample
// oscillator period. The 16 lane phases are advanced by kBlockStep each loop.
alignas(64) const cbfloat16 kInitialOscillator[kVecFactor] = {
    {(bfloat16)1.000000000f, (bfloat16)0.000000000f},
    {(bfloat16)0.898674466f, (bfloat16)0.438616239f},
    {(bfloat16)0.615231591f, (bfloat16)0.788346428f},
    {(bfloat16)0.207111376f, (bfloat16)0.978317371f},
    {(bfloat16)-0.242980180f, (bfloat16)0.970031253f},
    {(bfloat16)-0.643831543f, (bfloat16)0.765167266f},
    {(bfloat16)-0.914209756f, (bfloat16)0.405241314f},
    {(bfloat16)-0.999322385f, (bfloat16)-0.036807223f},
    {(bfloat16)-0.881921264f, (bfloat16)-0.471396737f},
    {(bfloat16)-0.585797857f, (bfloat16)-0.810457198f},
    {(bfloat16)-0.170961889f, (bfloat16)-0.985277642f},
    {(bfloat16)0.278519689f, (bfloat16)-0.960430519f},
    {(bfloat16)0.671558955f, (bfloat16)-0.740951125f},
    {(bfloat16)0.928506080f, (bfloat16)-0.371317194f},
    {(bfloat16)0.997290457f, (bfloat16)0.073564564f},
    {(bfloat16)0.863972856f, (bfloat16)0.503538384f},
};

const cbfloat16 kBlockStep = {(bfloat16)0.555570233f,
                              (bfloat16)0.831469612f};

} // namespace

extern "C" {

void frequency_shift(cbfloat16 *a, cbfloat16 *c, int32_t N) {
  cbfloat16 *__restrict pA = a;
  cbfloat16 *__restrict pC = c;
  const int F = N / kVecFactor;

  aie::vector<cbfloat16, kVecFactor> oscillator =
      aie::load_v<kVecFactor>(kInitialOscillator);

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<cbfloat16, kVecFactor> samples = aie::load_v<kVecFactor>(pA);
    pA += kVecFactor;

    aie::accum<caccfloat, kVecFactor> shifted = aie::mul(samples, oscillator);
    aie::store_v(pC, shifted.template to_vector<cbfloat16>(0));
    pC += kVecFactor;

    aie::accum<caccfloat, kVecFactor> next_osc =
        aie::mul(oscillator, kBlockStep);
    oscillator = next_osc.template to_vector<cbfloat16>(0);
  }
}

} // extern "C"
