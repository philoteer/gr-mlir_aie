/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_CINT32_TO_COMPLEX64_IMPL_H
#define INCLUDED_MLIR_AIE_CINT32_TO_COMPLEX64_IMPL_H

#include <gnuradio/mlir_aie/cint32_to_complex64.h>

namespace gr {
namespace mlir_aie {

class cint32_to_complex64_impl : public cint32_to_complex64
{
private:
    const float d_scale;

public:
    cint32_to_complex64_impl(unsigned int integer_bits, unsigned int fractional_bits);
    ~cint32_to_complex64_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_CINT32_TO_COMPLEX64_IMPL_H */
