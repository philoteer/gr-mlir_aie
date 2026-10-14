/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H
#define INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H

#include <gnuradio/mlir_aie/sigmf_tag_injector.h>
#include <pmt/pmt.h>
#include <utility>
#include <vector>

namespace gr {
namespace mlir_aie {

class sigmf_tag_injector_impl : public sigmf_tag_injector
{
private:
    size_t d_itemsize;
    pmt::pmt_t d_tag_key;
    std::vector<std::pair<uint64_t, double>> d_changes;
    size_t d_next_change = 0;

public:
    sigmf_tag_injector_impl(const std::string& metadata_file,
                            size_t itemsize,
                            const std::string& tag_key,
                            uint64_t start_sample);
    ~sigmf_tag_injector_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_IMPL_H */
