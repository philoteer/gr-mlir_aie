/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_FLOAT32_TO_Q15_IMPL_H
#define INCLUDED_MLIR_AIE_FLOAT32_TO_Q15_IMPL_H

#include <gnuradio/mlir_aie/float32_to_Q15.h>

namespace gr {
namespace mlir_aie {

class float32_to_Q15_impl : public float32_to_Q15
{
private:
    const bool d_safe;

public:
    explicit float32_to_Q15_impl(bool safe);
    ~float32_to_Q15_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_FLOAT32_TO_Q15_IMPL_H */
