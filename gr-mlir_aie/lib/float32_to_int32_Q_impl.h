/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_IMPL_H
#define INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_IMPL_H

#include <gnuradio/mlir_aie/float32_to_int32_Q.h>

namespace gr {
namespace mlir_aie {

class float32_to_int32_Q_impl : public float32_to_int32_Q
{
private:
    const float d_scale;
    const bool d_safe;

public:
    float32_to_int32_Q_impl(unsigned int integer_bits, unsigned int fractional_bits, bool safe);
    ~float32_to_int32_Q_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_IMPL_H */
