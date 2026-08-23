/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_H
#define INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_H

#include <gnuradio/mlir_aie/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief <+description of block+>
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API complex64_to_cbfloat : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<complex64_to_cbfloat> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::complex64_to_cbfloat.
     *
     * To avoid accidental use of raw pointers, mlir_aie::complex64_to_cbfloat's
     * constructor is in a private implementation
     * class. mlir_aie::complex64_to_cbfloat::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_COMPLEX64_TO_CBFLOAT_H */
