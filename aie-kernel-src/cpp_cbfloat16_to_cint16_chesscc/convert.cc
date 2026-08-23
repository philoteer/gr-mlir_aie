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

void cbfloat16_to_cint16(cbfloat16* input, cint16* output, int32_t count)
{
    constexpr float q15_scale = 32768.0F;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        cint16 value;
        value.real = static_cast<int16_t>(static_cast<float>(input[i].real) * q15_scale);
        value.imag = static_cast<int16_t>(static_cast<float>(input[i].imag) * q15_scale);
        output[i] = value;
    }
}

} // extern "C"
