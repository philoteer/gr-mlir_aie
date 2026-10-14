/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H
#define INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H

#include <gnuradio/mlir_aie/sigmf_tag_injector.h>

namespace gr {
namespace mlir_aie {

class sigmf_tag_injector_impl : public sigmf_tag_injector
{
private:
    // Nothing to declare in this block.

public:
    sigmf_tag_injector_impl();
    ~sigmf_tag_injector_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H */
