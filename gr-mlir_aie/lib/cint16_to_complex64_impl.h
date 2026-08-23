/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_CINT16_TO_COMPLEX64_IMPL_H
#define INCLUDED_MLIR_AIE_CINT16_TO_COMPLEX64_IMPL_H

#include <gnuradio/mlir_aie/cint16_to_complex64.h>

namespace gr {
namespace mlir_aie {

class cint16_to_complex64_impl : public cint16_to_complex64
{
private:
    // Nothing to declare in this block.

public:
    cint16_to_complex64_impl();
    ~cint16_to_complex64_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_CINT16_TO_COMPLEX64_IMPL_H */
