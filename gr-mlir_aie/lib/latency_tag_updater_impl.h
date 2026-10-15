/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H
#define INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H

#include <gnuradio/mlir_aie/latency_tag_updater.h>
#include <pmt/pmt.h>

namespace gr {
namespace mlir_aie {

class latency_tag_updater_impl : public latency_tag_updater
{
private:
    size_t d_itemsize;
    pmt::pmt_t d_name;
    const pmt::pmt_t d_start_key = pmt::intern("frame_start");
    const pmt::pmt_t d_end_key = pmt::intern("frame_end");
    const pmt::pmt_t d_timestamp_key = pmt::intern("timestamp_ns");
    const pmt::pmt_t d_timestamps_key = pmt::intern("timestamps_ns");

public:
    latency_tag_updater_impl(const std::string& name, size_t itemsize);
    ~latency_tag_updater_impl();

    // Where all the action really happens
    int work(int noutput_items,
             gr_vector_const_void_star& input_items,
             gr_vector_void_star& output_items) override;
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_IMPL_H */
