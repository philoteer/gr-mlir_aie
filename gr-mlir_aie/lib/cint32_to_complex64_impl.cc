/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cint32_to_complex64_impl.h"
#include <gnuradio/io_signature.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

using input_type = std::int64_t;
using output_type = gr_complex;

namespace {

std::int64_t sign_extend_int32(std::uint64_t value)
{
    const auto component = static_cast<std::uint32_t>(value);
    return (component & 0x80000000U) != 0U ? static_cast<std::int64_t>(component) - (1LL << 32)
                                             : static_cast<std::int64_t>(component);
}

float fixed_point_scale(unsigned int integer_bits, unsigned int fractional_bits)
{
    if (integer_bits > 31 || fractional_bits > 31 || integer_bits != 31 - fractional_bits) {
        throw std::invalid_argument("integer_bits and fractional_bits must sum to 31");
    }
    return std::ldexp(1.0F, -static_cast<int>(fractional_bits));
}

} // namespace

cint32_to_complex64::sptr cint32_to_complex64::make(unsigned int integer_bits,
                                                      unsigned int fractional_bits)
{
    return gnuradio::make_block_sptr<cint32_to_complex64_impl>(integer_bits, fractional_bits);
}


/*
 * The private constructor
 */
cint32_to_complex64_impl::cint32_to_complex64_impl(unsigned int integer_bits,
                                                     unsigned int fractional_bits)
    : gr::sync_block("cint32_to_complex64",
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
cint32_to_complex64_impl::~cint32_to_complex64_impl() {}

int cint32_to_complex64_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (int i = 0; i < noutput_items; ++i) {
        const auto packed = static_cast<std::uint64_t>(in[i]);
        const auto real = static_cast<float>(sign_extend_int32(packed)) * d_scale;
        const auto imag = static_cast<float>(sign_extend_int32(packed >> 32)) * d_scale;
        out[i] = output_type(real, imag);
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
