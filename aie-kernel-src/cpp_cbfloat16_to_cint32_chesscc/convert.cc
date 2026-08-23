//===- convert.cc -----------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

#ifndef FRACTIONAL_BITS
#error "FRACTIONAL_BITS must be defined"
#endif

static_assert(FRACTIONAL_BITS >= 0 && FRACTIONAL_BITS <= 31,
              "FRACTIONAL_BITS must be in the range [0, 31]");

extern "C" {

void cbfloat16_to_cint32(cbfloat16* input, cint32* output, int32_t count)
{
    constexpr float fixed_point_scale = static_cast<float>(uint64_t{1} << FRACTIONAL_BITS);

    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int32_t i = 0; i < count; ++i) {
        cint32 value;
        value.real = static_cast<int32_t>(static_cast<float>(input[i].real) * fixed_point_scale);
        value.imag = static_cast<int32_t>(static_cast<float>(input[i].imag) * fixed_point_scale);
        output[i] = value;
    }
}

} // extern "C"
