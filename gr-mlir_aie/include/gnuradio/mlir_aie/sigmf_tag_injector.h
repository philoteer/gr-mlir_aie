/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_H
#define INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_H

#include <gnuradio/mlir_aie/api.h>
#include <gnuradio/sync_block.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace gr {
namespace mlir_aie {

/*!
 * \brief Pass samples through and inject SigMF frequency and frame-start tags.
 * \ingroup mlir_aie
 * One stream item must represent one SigMF sample (both I and Q for complex
 * data). Existing tags and sample bytes are preserved. The first known
 * frequency and each subsequent change produce a PMT double tag in Hz.
 * Annotations with core:label beginning "wifi-frame-" produce a "frame_start"
 * PMT dictionary at core:sample_start. Fields are frame_id (the unique label,
 * PMT symbol), sample_start (original SigMF offset, uint64), and timestamp_ns
 * (CLOCK_MONOTONIC nanoseconds at tag emission, uint64). Labels must be unique
 * within the metadata. Unrelated annotations are ignored. Frame starts before
 * start_sample are skipped; annotation order need not be sorted.
 * Timestamps measure software passage through this block, including scheduler
 * batching effects, and are directly comparable to Python time.monotonic_ns().
 * Metadata is read once at construction. Playback must not repeat or seek
 * after construction; start_sample supports an initial File Source offset.
 */
class MLIR_AIE_API sigmf_tag_injector : virtual public gr::sync_block
{
public:
    typedef std::shared_ptr<sigmf_tag_injector> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of mlir_aie::sigmf_tag_injector.
     *
     * To avoid accidental use of raw pointers, mlir_aie::sigmf_tag_injector's
     * constructor is in a private implementation
     * class. mlir_aie::sigmf_tag_injector::make is the public interface for
     * creating new instances.
     */
    static sptr make(const std::string& metadata_file,
                     size_t itemsize = 8,
                     const std::string& tag_key = "rx_freq",
                     uint64_t start_sample = 0);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_SIGMF_TAG_INJECTOR_H */
