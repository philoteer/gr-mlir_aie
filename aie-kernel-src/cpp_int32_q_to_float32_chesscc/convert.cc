//===- convert.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#ifndef FRACTIONAL_BITS
#error "FRACTIONAL_BITS must be defined"
#endif

static_assert(FRACTIONAL_BITS >= 0 && FRACTIONAL_BITS <= 31,
              "FRACTIONAL_BITS must be in the range [0, 31]");

extern "C" {

void int32_q_to_float32(int32_t* input, float* output, int32_t count)
{
    constexpr float fixed_point_scale = 1.0F / static_cast<float>(uint64_t{1} << FRACTIONAL_BITS);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]) * fixed_point_scale;
    }
}

} // extern "C"
