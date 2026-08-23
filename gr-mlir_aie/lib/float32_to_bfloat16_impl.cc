/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "float32_to_bfloat16_impl.h"
#include <gnuradio/io_signature.h>
#include <cstdint>

namespace gr {
namespace mlir_aie {

using input_type = float;
using output_type = std::int16_t;
float32_to_bfloat16::sptr float32_to_bfloat16::make()
{
    return gnuradio::make_block_sptr<float32_to_bfloat16_impl>();
}


/*
 * The private constructor
 */
float32_to_bfloat16_impl::float32_to_bfloat16_impl()
    : gr::sync_block("float32_to_bfloat16",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
}

/*
 * Our virtual destructor.
 */
float32_to_bfloat16_impl::~float32_to_bfloat16_impl() {}

int float32_to_bfloat16_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    int j = 0;
    
    const int vec_step = 32;
    if (noutput_items >= vec_step) {
        for (; j <= noutput_items - vec_step; j += vec_step) {
            // Load 32 floats: 16 into reg_b (low), 16 into reg_a (high)
            // Note: vcvtne2ps2bf16 expects 'a' to be the upper half source 
            // and 'b' to be the lower half source.
            __m512 input_low  = _mm512_loadu_ps(&in[j]);
            __m512 input_high = _mm512_loadu_ps(&in[j + 16]);

            // Perform conversion
            __m512bh bf16_vec = _mm512_cvtne2ps_pbh(input_high, input_low);

            // Store 32 bf16 values (512 bits)
            // Casting to __m512i for store intrinsic
            _mm512_storeu_si512((void*)&out[j], _mm512_castbh_si512(bf16_vec));
        }
    }

    // Scalar fallback for remaining elements (or if VECTOR_SIZE < 32)
    for (; j < noutput_items; j++) {
        out[j] = float_to_bfloat16(in[j]);
    }
    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
