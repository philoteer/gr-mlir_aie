/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "cbfloat_to_complex64_impl.h"
#include <gnuradio/io_signature.h>
#include <cstdint>

namespace gr {
namespace mlir_aie {

using input_type = std::int32_t;
using output_type = gr_complex;
cbfloat_to_complex64::sptr cbfloat_to_complex64::make()
{
    return gnuradio::make_block_sptr<cbfloat_to_complex64_impl>();
}


/*
 * The private constructor
 */
cbfloat_to_complex64_impl::cbfloat_to_complex64_impl()
    : gr::sync_block("cbfloat_to_complex64",
                     gr::io_signature::make(
                         1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                     gr::io_signature::make(
                         1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
}

/*
 * Our virtual destructor.
 */
cbfloat_to_complex64_impl::~cbfloat_to_complex64_impl() {}

int cbfloat_to_complex64_impl::work(int noutput_items,
                                    gr_vector_const_void_star& input_items,
                                    gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    for (int j = 0; j < noutput_items; j++) {
        out[j] = cbfloat_to_complex(in[j]);
    }
    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
