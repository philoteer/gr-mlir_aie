/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_H
#define INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_H

#include <gnuradio/mlir_aie/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief <+description of block+>
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API bfloat16_to_float32 : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<bfloat16_to_float32> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::bfloat16_to_float32.
     *
     * To avoid accidental use of raw pointers, mlir_aie::bfloat16_to_float32's
     * constructor is in a private implementation
     * class. mlir_aie::bfloat16_to_float32::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_BFLOAT16_TO_FLOAT32_H */
