/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_CBFLOAT_TO_COMPLEX64_IMPL_H
#define INCLUDED_MLIR_AIE_CBFLOAT_TO_COMPLEX64_IMPL_H

#include <gnuradio/mlir_aie/cbfloat_to_complex64.h>
#include <gnuradio/io_signature.h>
#include <immintrin.h> 

namespace gr {
namespace mlir_aie {

class cbfloat_to_complex64_impl : public cbfloat_to_complex64
{
private:
    // Nothing to declare in this block.
    static float bfloat16_to_float(uint32_t bf) {
        uint32_t x = bf << 16;
        float f;
        std::memcpy(&f, &x, sizeof(uint32_t));
        return f;
    }
    
    static gr_complex cbfloat_to_complex(uint32_t val){
        return gr_complex(bfloat16_to_float(val & 0xFFFFu), bfloat16_to_float(val >>16));
    }
    
    
    #ifndef _mm512_castbh_si512
    static inline __m512i _mm512_castbh_si512(__m512bh a) {
        return (__m512i)a;
    }
    #endif

public:
    cbfloat_to_complex64_impl();
    ~cbfloat_to_complex64_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_CBFLOAT_TO_COMPLEX64_IMPL_H */
