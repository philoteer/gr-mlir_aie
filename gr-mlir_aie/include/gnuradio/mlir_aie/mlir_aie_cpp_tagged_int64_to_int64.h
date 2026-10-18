/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT64_TO_INT64_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT64_TO_INT64_H

#include <gnuradio/block.h>
#include <gnuradio/mlir_aie/api.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief <+description of block+>
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API mlir_aie_cpp_tagged_int64_to_int64 : virtual public gr::block
{
public:
    typedef std::shared_ptr<mlir_aie_cpp_tagged_int64_to_int64> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of
     * mlir_aie::mlir_aie_cpp_tagged_int64_to_int64.
     *
     * To avoid accidental use of raw pointers,
     * mlir_aie::mlir_aie_cpp_tagged_int64_to_int64's constructor is in a private
     * implementation class. mlir_aie::mlir_aie_cpp_tagged_int64_to_int64::make is the
     * public interface for creating new instances.
     */
    static sptr make();
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT64_TO_INT64_H */
