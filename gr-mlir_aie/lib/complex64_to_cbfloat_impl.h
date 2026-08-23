/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_IMPL_H
#define INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_IMPL_H

#include <gnuradio/mlir_aie/complex64_to_cbfloat.h>
#include <gnuradio/io_signature.h>
#include <immintrin.h> 

namespace gr {
namespace mlir_aie {

class complex64_to_cbfloat_impl : public complex64_to_cbfloat
{
private:
    static uint32_t float_to_bfloat16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(uint32_t));
        // Truncate the lower 16 bits (standard bfloat16 conversion)
        return x >> 16;
    }
    
    static uint32_t complex_to_cbfloat(gr_complex f){
        return (float_to_bfloat16(f.imag()) << 16) + float_to_bfloat16(f.real());
    }
    
    #ifndef _mm512_castbh_si512
    static inline __m512i _mm512_castbh_si512(__m512bh a) {
        return (__m512i)a;
    }
    #endif
    

public:
    complex64_to_cbfloat_impl();
    ~complex64_to_cbfloat_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_IMPL_H */
