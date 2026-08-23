/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "complex64_to_cint16_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace gr {
namespace mlir_aie {

using input_type = gr_complex;
using output_type = std::uint32_t;

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

template <typename Converter>
output_type complex_to_cint16(const input_type& value, Converter convert)
{
    // XDNA cint16 stores the real component first, followed by imaginary.
    const auto real = static_cast<std::uint16_t>(convert(value.real()));
    const auto imag = static_cast<std::uint16_t>(convert(value.imag()));
    return static_cast<output_type>(real) | (static_cast<output_type>(imag) << 16);
}

} // namespace

complex64_to_cint16::sptr complex64_to_cint16::make(bool safe)
{
    return gnuradio::make_block_sptr<complex64_to_cint16_impl>(safe);
}


/*
 * The private constructor
 */
complex64_to_cint16_impl::complex64_to_cint16_impl(bool safe)
    : gr::sync_block("complex64_to_cint16",
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
complex64_to_cint16_impl::~complex64_to_cint16_impl() {}

int complex64_to_cint16_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    if (d_safe) {
        for (int i = 0; i < noutput_items; i++) {
            out[i] = complex_to_cint16(in[i], float_to_q15);
        }
    } else {
        for (int i = 0; i < noutput_items; i++) {
            out[i] = complex_to_cint16(in[i], float_to_q15_unsafe);
        }
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
