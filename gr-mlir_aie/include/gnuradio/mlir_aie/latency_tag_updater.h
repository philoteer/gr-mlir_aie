/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_H
#define INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_H

#include <gnuradio/mlir_aie/api.h>
#include <gnuradio/sync_block.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief <+description of block+>
 * \ingroup mlir_aie
 *
 */
class MLIR_AIE_API latency_tag_updater : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<latency_tag_updater> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::latency_tag_updater.
     *
     * To avoid accidental use of raw pointers, mlir_aie::latency_tag_updater's
     * constructor is in a private implementation
     * class. mlir_aie::latency_tag_updater::make is the public interface for
     * creating new instances.
     */
    static sptr make();
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_H */
