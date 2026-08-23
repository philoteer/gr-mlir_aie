// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include <aie_api/aie.hpp>

namespace {

constexpr unsigned kFftSize = 64;
constexpr unsigned kVectorSize = 8;
constexpr unsigned kTwiddleShift = 15;
constexpr unsigned kStageShift = 15;

#include "fft64_twiddles.h"

alignas(aie::vector_decl_align) static cint32 scratch[kFftSize];
alignas(aie::vector_decl_align) static cint32 bins[kFftSize];

} // namespace

extern "C" {

void fft64_cint16(const cint16 *__restrict input,
                  cint32 *__restrict output,
                  int32_t count) {
  aie::set_rounding(aie::rounding_mode::positive_inf);
  aie::set_saturation(aie::saturation_mode::saturate);

  for (int32_t frame = 0; frame + kFftSize <= count; frame += kFftSize) {
    aie::fft_dit_r4_stage<16>(input + frame, kStage0Tw2, kStage0Tw1,
                              kStage0Tw3, kFftSize, kTwiddleShift,
                              kStageShift, false, bins);
    aie::fft_dit_r4_stage<4>(bins, kStage1Tw2, kStage1Tw1, kStage1Tw3,
                             kFftSize, kTwiddleShift, kStageShift, false,
                             scratch);
    aie::fft_dit_r4_stage<1>(scratch, kStage2Tw2, kStage2Tw1, kStage2Tw3,
                             kFftSize, kTwiddleShift, kStageShift, false,
                             bins);

    // fftshift: place negative frequencies before non-negative frequencies.
    for (unsigned i = 0; i < kFftSize / 2; i += kVectorSize) {
      aie::store_unaligned_v(output + frame + i,
                   aie::load_v<kVectorSize>(bins + i + kFftSize / 2));
      aie::store_unaligned_v(output + frame + i + kFftSize / 2,
                   aie::load_v<kVectorSize>(bins + i));
    }
  }
}

} // extern "C"
