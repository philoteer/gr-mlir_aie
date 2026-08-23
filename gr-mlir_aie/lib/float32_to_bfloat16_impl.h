/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_FLOAT32_TO_BFLOAT16_IMPL_H
#define INCLUDED_MLIR_AIE_FLOAT32_TO_BFLOAT16_IMPL_H

#include <gnuradio/mlir_aie/float32_to_bfloat16.h>
#include <immintrin.h> 

namespace gr {
namespace mlir_aie {

class float32_to_bfloat16_impl : public float32_to_bfloat16
{
private:
    static uint16_t float_to_bfloat16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(uint32_t));
        // Truncate the lower 16 bits (standard bfloat16 conversion)
        return static_cast<uint16_t>(x >> 16);
    }
    #ifndef _mm512_castbh_si512
    static inline __m512i _mm512_castbh_si512(__m512bh a) {
        return (__m512i)a;
    }
    #endif
    
public:
    float32_to_bfloat16_impl();
    ~float32_to_bfloat16_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_FLOAT32_TO_BFLOAT16_IMPL_H */
