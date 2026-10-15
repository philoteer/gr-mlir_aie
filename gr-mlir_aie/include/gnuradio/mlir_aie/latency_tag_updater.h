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
#include <cstddef>
#include <string>

namespace gr {
namespace mlir_aie {

/*!
 * \brief Append a named monotonic timestamp to SigMF frame-event tags.
 * \ingroup mlir_aie
 * Samples and all tags pass through, preserving offsets and tag source IDs.
 * For frame_start/frame_end PMT dictionaries with a uint64 timestamp_ns,
 * adds timestamps_ns[name] as uint64 CLOCK_MONOTONIC nanoseconds. The original
 * timestamp_ns and all other fields and named timestamps are preserved.
 * One timestamp is taken at work() entry and shared by events in that callback.
 * An existing entry with the same name is retained (first observation wins).
 * Tags with a non-dictionary timestamps_ns field are passed through unchanged.
 * Use distinct, nonempty names for distinct measurement points. itemsize is
 * the full stream item size in bytes, including vector length if applicable.
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
    static sptr make(const std::string& name, size_t itemsize = 8);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_LATENCY_TAG_UPDATER_H */
