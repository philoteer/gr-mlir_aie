/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cint16_to_complex64_impl.h"
#include <gnuradio/io_signature.h>
#include <cstdint>

namespace gr {
namespace mlir_aie {

using input_type = std::uint32_t;
using output_type = gr_complex;

namespace {

float q15_to_float(std::uint32_t value)
{
    const auto component = static_cast<std::uint16_t>(value);
    const auto signed_component = (component & 0x8000U) != 0U
                                      ? static_cast<std::int32_t>(component) - 65536
                                      : static_cast<std::int32_t>(component);
    return static_cast<float>(signed_component) * (1.0F / 32768.0F);
}

output_type cint16_to_complex(input_type value)
{
    // On two's-complement hosts, this can instead use a direct signed cast:
    // return output_type(static_cast<float>(static_cast<std::int16_t>(value)) *
    //                        (1.0F / 32768.0F),
    //                    static_cast<float>(static_cast<std::int16_t>(value >> 16)) *
    //                        (1.0F / 32768.0F));
    return output_type(q15_to_float(value), q15_to_float(value >> 16));
}

} // namespace

cint16_to_complex64::sptr cint16_to_complex64::make()
{
    return gnuradio::make_block_sptr<cint16_to_complex64_impl>();
}


/*
 * The private constructor
 */
cint16_to_complex64_impl::cint16_to_complex64_impl()
    : gr::sync_block("cint16_to_complex64",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
}

/*
 * Our virtual destructor.
 */
cint16_to_complex64_impl::~cint16_to_complex64_impl() {}

int cint16_to_complex64_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (int i = 0; i < noutput_items; ++i) {
        out[i] = cint16_to_complex(in[i]);
    }

    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
