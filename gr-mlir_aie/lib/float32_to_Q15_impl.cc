/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "float32_to_Q15_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gr {
namespace mlir_aie {

using input_type = float;
using output_type = std::int16_t;

namespace {

std::int16_t float_to_q15(float value)
{
    if (std::isnan(value)) {
        return 0;
    }
    if (value >= 1.0F) {
        return std::numeric_limits<std::int16_t>::max();
    }
    if (value <= -1.0F) {
        return std::numeric_limits<std::int16_t>::min();
    }

    const long quantized = std::lround(value * 32768.0F);
    return static_cast<std::int16_t>(
        std::min(quantized,
                 static_cast<long>(std::numeric_limits<std::int16_t>::max())));
}

std::int16_t float_to_q15_unsafe(float value)
{
    return static_cast<std::int16_t>(value * 32768.0F);
}

} // namespace

float32_to_Q15::sptr float32_to_Q15::make(bool safe)
{
    return gnuradio::make_block_sptr<float32_to_Q15_impl>(safe);
}


/*
 * The private constructor
 */
float32_to_Q15_impl::float32_to_Q15_impl(bool safe)
    : gr::sync_block("float32_to_Q15",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_safe(safe)
{
}

/*
 * Our virtual destructor.
 */
float32_to_Q15_impl::~float32_to_Q15_impl() {}

int float32_to_Q15_impl::work(int noutput_items,
                              gr_vector_const_void_star& input_items,
                              gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    if (d_safe) {
        for (int i = 0; i < noutput_items; ++i) {
            out[i] = float_to_q15(in[i]);
        }
    } else {
        for (int i = 0; i < noutput_items; ++i) {
            out[i] = float_to_q15_unsafe(in[i]);
        }
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
