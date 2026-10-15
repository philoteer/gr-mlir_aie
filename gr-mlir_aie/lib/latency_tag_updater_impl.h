/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H
#define INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H

#include <gnuradio/mlir_aie/latency_tag_updater.h>

namespace gr {
namespace mlir_aie {

class latency_tag_updater_impl : public latency_tag_updater
{
private:
    // Nothing to declare in this block.

public:
    latency_tag_updater_impl();
    ~latency_tag_updater_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H */
