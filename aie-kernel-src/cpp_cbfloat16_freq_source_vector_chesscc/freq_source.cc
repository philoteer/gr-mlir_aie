//===- freq_source.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <math.h>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#define kVecFactor 16
#define kQShift 15
#define kPhaseScale 4294967296.0f
#define kQuadrantPhase 0x40000000u
#define kHalfPiQ30 1686629713
#define kSinAQ30 1073495352
#define kSinBQ30 -177929150
#define kSinCQ30 8067080
#define OSCILLATOR_RESET_CNT 4

alignas(64) static aie::vector<cint16, kVecFactor> oscillator;
static float last_frequency = -99.0f;
static uint32_t phase_step_q;
alignas(64) static aie::vector<cint16, kVecFactor> rotate_factor;
// One full turn is 2^32, so unsigned overflow wraps phase modulo 2*pi.
static uint32_t current_phase = 0;
static int osc_cnt = OSCILLATOR_RESET_CNT+1;

static inline uint32_t frequency_to_phase_step_q(float frequency) {
  float cycles = frequency - floorf(frequency);
  return (uint32_t)(cycles * kPhaseScale);
}

static inline int16_t clamp_q15(int32_t x) {
  if (x > 32767)
    return 32767;
  if (x < -32768)
    return -32768;
  return (int16_t)x;
}

static inline int16_t sin_first_quadrant_q15(uint32_t phase_q30) {
  int64_t x = ((int64_t)phase_q30 * kHalfPiQ30 + (1ll << 29)) >> 30;
  int64_t x2 = (x * x + (1ll << 29)) >> 30;
  int64_t inner = kSinBQ30 + ((x2 * kSinCQ30 + (1ll << 29)) >> 30);
  int64_t poly = kSinAQ30 + ((x2 * inner + (1ll << 29)) >> 30);
  int64_t y = (x * poly + (1ll << 29)) >> 30;
  return clamp_q15((int32_t)((y + (1 << 14)) >> 15));
}

static inline int16_t sin_minimax_3(uint32_t phase) {
  uint32_t quadrant = phase >> 30;
  uint32_t offset = phase & (kQuadrantPhase - 1);
  int16_t magnitude = sin_first_quadrant_q15(
      (quadrant & 1) ? kQuadrantPhase - offset : offset);

  return (quadrant < 2) ? magnitude : (int16_t)-magnitude;
}

static inline int16_t cos_minimax_3(uint32_t phase) {
  return sin_minimax_3(phase + kQuadrantPhase);
}

static inline cint16 make_q15(int16_t real, int16_t imag) {
  cint16 value;
  value.real = real;
  value.imag = imag;
  return value;
}

extern "C" {

void frequency_source(float *frequency, cbfloat16 *out, int32_t N) {
  cbfloat16 *__restrict pOut = out;

  // 1. Update frequency-dependent parameters if the frequency changed
  if (last_frequency != frequency[0]) {
    last_frequency = frequency[0];
    phase_step_q = frequency_to_phase_step_q(frequency[0]);
    
    uint32_t block_phase_step_q = (uint32_t)kVecFactor * phase_step_q;
    cint16 next = make_q15(cos_minimax_3(block_phase_step_q),
                           sin_minimax_3(block_phase_step_q));
    rotate_factor = aie::broadcast<cint16, kVecFactor>(next);
  }
  
  // 2. Re-calculate the oscillator vector based on the current phase tracking
  if(osc_cnt >= OSCILLATOR_RESET_CNT)
  {
    alignas(32) cint16 oscillator_init[kVecFactor];
    for (int i = 0; i < kVecFactor; i++) {
      uint32_t element_phase = current_phase + (uint32_t)i * phase_step_q;
      oscillator_init[i] = make_q15(cos_minimax_3(element_phase),
                                    sin_minimax_3(element_phase));
    }
    oscillator = aie::load_v<kVecFactor>(oscillator_init);
    osc_cnt = 0;
  }
  
  // 3. Execute the streaming loop
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int32_t i = 0; i < N; i += kVecFactor) {
  // Step A: Vectorized conversion from cint16 to cbfloat16
    aie::vector<cbfloat16, kVecFactor> out_vec = aie::to_float<cbfloat16>(oscillator, kQShift);
    
    // Step B: Direct vector store to the output pointer
    aie::store_v(pOut, out_vec);
    pOut += kVecFactor;
          
    // Step C: Update oscillator for the next iteration
    aie::accum<cacc48, kVecFactor> next_osc = aie::mul(oscillator, rotate_factor);
    oscillator = next_osc.template to_vector<cint16>(kQShift);
  }

  // 4. Track the end-of-execution phase to ensure the next call starts seamlessly
  current_phase += (uint32_t)N * phase_step_q;
  osc_cnt++;
}
}
