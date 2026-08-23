/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "bfloat16_to_float32_impl.h"
#include <gnuradio/io_signature.h>
#include <cstdint>

namespace gr {
namespace mlir_aie {

using input_type = std::int16_t;
using output_type = float;
bfloat16_to_float32::sptr bfloat16_to_float32::make()
{
    return gnuradio::make_block_sptr<bfloat16_to_float32_impl>();
}


/*
 * The private constructor
 */
bfloat16_to_float32_impl::bfloat16_to_float32_impl()
    : gr::sync_block("bfloat16_to_float32",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
}

/*
 * Our virtual destructor.
 */
bfloat16_to_float32_impl::~bfloat16_to_float32_impl() {}

int bfloat16_to_float32_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (int j = 0; j < noutput_items; j++) {
        out[j] = bfloat16_to_float(in[j]);
    }
    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
