//===- convert.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

extern "C" {

void q15_to_bfloat16(int16_t* input, bfloat16* output, int32_t count)
{
    constexpr float q15_scale = 1.0F / 32768.0F;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        output[i] = static_cast<bfloat16>(static_cast<float>(input[i]) * q15_scale);
    }
}

} // extern "C"
