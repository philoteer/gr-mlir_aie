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

void cint16_to_cbfloat16(cint16* input, cbfloat16* output, int32_t count)
{
    constexpr float q15_scale = 1.0F / 32768.0F;

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        cbfloat16 value;
        value.real = static_cast<bfloat16>(static_cast<float>(input[i].real) * q15_scale);
        value.imag = static_cast<bfloat16>(static_cast<float>(input[i].imag) * q15_scale);
        output[i] = value;
    }
}

} // extern "C"
