/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_H
#define INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_H

#include <gnuradio/mlir_aie/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief <+description of block+>
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API float32_to_int32_Q : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<float32_to_int32_Q> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::float32_to_int32_Q.
     *
     * To avoid accidental use of raw pointers, mlir_aie::float32_to_int32_Q's
     * constructor is in a private implementation
     * class. mlir_aie::float32_to_int32_Q::make is the public interface for
     * creating new instances.
     */
    static sptr make(unsigned int integer_bits, unsigned int fractional_bits, bool safe = false);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_FLOAT32_TO_INT32_Q_H */
