/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_INT32_Q_TO_FLOAT32_IMPL_H
#define INCLUDED_MLIR_AIE_INT32_Q_TO_FLOAT32_IMPL_H

#include <gnuradio/mlir_aie/int32_Q_to_float32.h>

namespace gr {
namespace mlir_aie {

class int32_Q_to_float32_impl : public int32_Q_to_float32
{
private:
    const float d_scale;

public:
    int32_Q_to_float32_impl(unsigned int integer_bits, unsigned int fractional_bits);
    ~int32_Q_to_float32_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_INT32_Q_TO_FLOAT32_IMPL_H */
