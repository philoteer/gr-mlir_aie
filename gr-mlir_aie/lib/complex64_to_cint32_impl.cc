/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "complex64_to_cint32_impl.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

using input_type = gr_complex;
using output_type = std::int64_t;

namespace {

float fixed_point_scale(unsigned int integer_bits, unsigned int fractional_bits)
{
    if (integer_bits > 31 || fractional_bits > 31 || integer_bits != 31 - fractional_bits) {
        throw std::invalid_argument("integer_bits and fractional_bits must sum to 31");
    }
    return std::ldexp(1.0F, static_cast<int>(fractional_bits));
}

std::int32_t float_to_fixed_safe(float value, float scale)
{
    if (std::isnan(value)) {
        return 0;
    }
    if (value >= static_cast<float>(std::numeric_limits<std::int32_t>::max()) / scale) {
        return std::numeric_limits<std::int32_t>::max();
    }
    if (value <= static_cast<float>(std::numeric_limits<std::int32_t>::min()) / scale) {
        return std::numeric_limits<std::int32_t>::min();
    }

    const auto scaled = std::llround(value * scale);
    return static_cast<std::int32_t>(
        std::clamp(scaled,
                   static_cast<long long>(std::numeric_limits<std::int32_t>::min()),
                   static_cast<long long>(std::numeric_limits<std::int32_t>::max())));
}

std::int32_t float_to_fixed_unsafe(float value, float scale)
{
    return static_cast<std::int32_t>(value * scale);
}

template <typename Converter>
output_type complex_to_cint32(const input_type& value, float scale, Converter convert)
{
    const auto real = static_cast<std::uint32_t>(convert(value.real(), scale));
    const auto imag = static_cast<std::uint32_t>(convert(value.imag(), scale));
    const auto packed = static_cast<std::uint64_t>(real) | (static_cast<std::uint64_t>(imag) << 32);
    return static_cast<output_type>(packed);
}

} // namespace

complex64_to_cint32::sptr complex64_to_cint32::make(unsigned int integer_bits,
                                                      unsigned int fractional_bits,
                                                      bool safe)
{
    return gnuradio::make_block_sptr<complex64_to_cint32_impl>(
        integer_bits, fractional_bits, safe);
}


/*
 * The private constructor
 */
complex64_to_cint32_impl::complex64_to_cint32_impl(unsigned int integer_bits,
                                                     unsigned int fractional_bits,
                                                     bool safe)
    : gr::sync_block("complex64_to_cint32",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type))),
      d_scale(fixed_point_scale(integer_bits, fractional_bits)),
      d_safe(safe)
{
}

/*
 * Our virtual destructor.
 */
complex64_to_cint32_impl::~complex64_to_cint32_impl() {}

int complex64_to_cint32_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    if (d_safe) {
        for (int i = 0; i < noutput_items; ++i) {
            out[i] = complex_to_cint32(in[i], d_scale, float_to_fixed_safe);
        }
    } else {
        for (int i = 0; i < noutput_items; ++i) {
            out[i] = complex_to_cint32(in[i], d_scale, float_to_fixed_unsafe);
        }
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
