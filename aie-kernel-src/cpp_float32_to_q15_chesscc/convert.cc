//===- convert.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

extern "C" {

void float32_to_q15(float* input, int16_t* output, int32_t count)
{
    constexpr float q15_scale = 32768.0F;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        output[i] = static_cast<int16_t>(input[i] * q15_scale);
    }
}

} // extern "C"
