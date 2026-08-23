/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "int32_Q_to_float32_impl.h"
#include <gnuradio/io_signature.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

using input_type = std::int32_t;
using output_type = float;

namespace {

float fixed_point_scale(unsigned int integer_bits, unsigned int fractional_bits)
{
    if (integer_bits > 31 || fractional_bits > 31 || integer_bits != 31 - fractional_bits) {
        throw std::invalid_argument("integer_bits and fractional_bits must sum to 31");
    }
    return std::ldexp(1.0F, -static_cast<int>(fractional_bits));
}

} // namespace

int32_Q_to_float32::sptr int32_Q_to_float32::make(unsigned int integer_bits,
                                                    unsigned int fractional_bits)
{
    return gnuradio::make_block_sptr<int32_Q_to_float32_impl>(integer_bits, fractional_bits);
}


/*
 * The private constructor
 */
int32_Q_to_float32_impl::int32_Q_to_float32_impl(unsigned int integer_bits,
                                                   unsigned int fractional_bits)
    : gr::sync_block("int32_Q_to_float32",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_scale(fixed_point_scale(integer_bits, fractional_bits))
{
}

/*
 * Our virtual destructor.
 */
int32_Q_to_float32_impl::~int32_Q_to_float32_impl() {}

int int32_Q_to_float32_impl::work(int noutput_items,
                                  gr_vector_const_void_star& input_items,
                                  gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (int i = 0; i < noutput_items; ++i) {
        out[i] = static_cast<float>(in[i]) * d_scale;
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
