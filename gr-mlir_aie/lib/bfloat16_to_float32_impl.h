/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_IMPL_H
#define INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_IMPL_H

#include <gnuradio/mlir_aie/bfloat16_to_float32.h>
#include <immintrin.h> 

namespace gr {
namespace mlir_aie {

class bfloat16_to_float32_impl : public bfloat16_to_float32
{
private:
    // Nothing to declare in this block.
    static float bfloat16_to_float(uint16_t bf) {
        uint32_t x = static_cast<uint32_t>(bf) << 16;
        float f;
        std::memcpy(&f, &x, sizeof(uint32_t));
        return f;
    }
    #ifndef _mm512_castbh_si512
    static inline __m512i _mm512_castbh_si512(__m512bh a) {
        return (__m512i)a;
    }
    #endif
    
public:
    bfloat16_to_float32_impl();
    ~bfloat16_to_float32_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_IMPL_H */
