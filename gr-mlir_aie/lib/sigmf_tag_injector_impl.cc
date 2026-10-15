/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "sigmf_tag_injector_impl.h"
#include <gnuradio/io_signature.h>
#include <boost/json.hpp>
// Compile Boost.JSON here so no separate libboost_json is required.
#include <boost/json/src.hpp>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <time.h>
#include <unordered_set>

namespace gr {
namespace mlir_aie {
namespace {
int checked_itemsize(size_t itemsize)
{
    if (itemsize == 0 || itemsize > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("sigmf_tag_injector: invalid item size");
    return static_cast<int>(itemsize);
}
} // namespace

sigmf_tag_injector::sptr sigmf_tag_injector::make(const std::string& metadata_file,
                                               size_t itemsize,
                                               const std::string& tag_key,
                                               uint64_t start_sample)
{
    return gnuradio::make_block_sptr<sigmf_tag_injector_impl>(
        metadata_file, itemsize, tag_key, start_sample);
}

sigmf_tag_injector_impl::sigmf_tag_injector_impl(const std::string& metadata_file,
                                               size_t itemsize,
                                               const std::string& tag_key,
                                               uint64_t start_sample)
    : gr::sync_block("sigmf_tag_injector",
                     gr::io_signature::make(1, 1, checked_itemsize(itemsize)),
                     gr::io_signature::make(1, 1, checked_itemsize(itemsize))),
      d_itemsize(itemsize),
      d_tag_key(pmt::intern(tag_key))
{
    if (tag_key.empty())
        throw std::invalid_argument("sigmf_tag_injector: tag key must not be empty");

    try {
        std::ifstream file(metadata_file);
        if (!file)
            throw std::runtime_error("cannot open metadata file");
        const std::string text((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
        const auto metadata = boost::json::parse(text);
        const auto& captures = metadata.as_object().at("captures").as_array();
        bool have_start = false;
        uint64_t previous_start = 0;
        bool have_frequency = false;
        double previous_frequency = 0;
        bool have_initial = false;
        double initial_frequency = 0;
        for (const auto& value : captures) {
            const auto& capture = value.as_object();
            const auto& offset = capture.at("core:sample_start");
            uint64_t sample_start;
            if (offset.is_uint64())
                sample_start = offset.as_uint64();
            else if (offset.is_int64() && offset.as_int64() >= 0)
                sample_start = static_cast<uint64_t>(offset.as_int64());
            else
                throw std::runtime_error("core:sample_start must be a nonnegative integer");
            if (have_start && sample_start <= previous_start)
                throw std::runtime_error("capture sample offsets must be strictly increasing");
            have_start = true;
            previous_start = sample_start;

            const auto* frequency_value = capture.if_contains("core:frequency");
            if (!frequency_value)
                continue; // A capture without a frequency does not change tuning.
            if (!frequency_value->is_number())
                throw std::runtime_error("core:frequency must be numeric");
            const double frequency = boost::json::value_to<double>(*frequency_value);
            if (!std::isfinite(frequency))
                throw std::runtime_error("core:frequency must be finite");
            if (sample_start <= start_sample) {
                have_initial = true;
                initial_frequency = frequency;
            } else if (!have_frequency || frequency != previous_frequency) {
                d_changes.emplace_back(sample_start - start_sample, frequency);
            }
            have_frequency = true;
            previous_frequency = frequency;
        }
        if (have_initial)
            d_changes.insert(d_changes.begin(), { 0, initial_frequency });

        // Other SigMF annotations need not describe frames. Only consume the
        // labels written by the Wi-Fi latency sample generator.
        const auto* annotations = metadata.as_object().if_contains("annotations");
        std::unordered_set<std::string> frame_ids;
        if (annotations && annotations->is_array()) {
            for (const auto& value : annotations->as_array()) {
                if (!value.is_object())
                    continue;
                const auto& annotation = value.as_object();
                const auto* label = annotation.if_contains("core:label");
                if (!label || !label->is_string())
                    continue;
                const std::string id(label->as_string().c_str());
                if (id.compare(0, 11, "wifi-frame-") != 0)
                    continue;
                if (!frame_ids.insert(id).second)
                    throw std::runtime_error("duplicate Wi-Fi frame label: " + id);
                const auto& offset = annotation.at("core:sample_start");
                uint64_t sample_start;
                if (offset.is_uint64())
                    sample_start = offset.as_uint64();
                else if (offset.is_int64() && offset.as_int64() >= 0)
                    sample_start = static_cast<uint64_t>(offset.as_int64());
                else
                    throw std::runtime_error(
                        "frame core:sample_start must be a nonnegative integer");
                if (sample_start >= start_sample)
                    d_frames.push_back(
                        { sample_start - start_sample, sample_start, sample_start, id, false });
                // Keep start-only annotations working, but validate every count
                // that is provided, even for frames preceding an initial seek.
                const auto* count = annotation.if_contains("core:sample_count");
                if (!count)
                    continue;
                uint64_t sample_count;
                if (count->is_uint64() && count->as_uint64() > 0)
                    sample_count = count->as_uint64();
                else if (count->is_int64() && count->as_int64() > 0)
                    sample_count = static_cast<uint64_t>(count->as_int64());
                else
                    throw std::runtime_error(
                        "frame core:sample_count must be a positive integer");
                if (sample_count - 1 > std::numeric_limits<uint64_t>::max() - sample_start)
                    throw std::runtime_error("frame end sample offset overflows uint64");
                const uint64_t sample_end = sample_start + (sample_count - 1);
                if (sample_end >= start_sample)
                    d_frames.push_back(
                        { sample_end - start_sample, sample_start, sample_end, id, true });
            }
        }
        std::stable_sort(d_frames.begin(), d_frames.end(),
                         [](const frame_event& a, const frame_event& b) {
                             return a.offset < b.offset;
                         });
    } catch (const std::exception& error) {
        throw std::runtime_error("sigmf_tag_injector: " + metadata_file + ": " +
                                 error.what());
    }
}

sigmf_tag_injector_impl::~sigmf_tag_injector_impl() {}

int sigmf_tag_injector_impl::work(int noutput_items,
                                gr_vector_const_void_star& input_items,
                                gr_vector_void_star& output_items)
{
    // One ingress timestamp for the entire scheduler buffer, before copying
    // samples or processing tags. Events in this callback share this time.
    timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        throw std::system_error(errno, std::generic_category(), "clock_gettime");
    const uint64_t timestamp_ns = static_cast<uint64_t>(now.tv_sec) *
                                      1000000000ULL + now.tv_nsec;
    std::memcpy(output_items[0], input_items[0],
                static_cast<size_t>(noutput_items) * d_itemsize);
    const uint64_t begin = nitems_written(0);
    while (d_next_change < d_changes.size()) {
        const auto& change = d_changes[d_next_change];
        if (change.first >= begin && change.first - begin >=
                                         static_cast<uint64_t>(noutput_items))
            break;
        if (change.first >= begin)
            add_item_tag(0, change.first, d_tag_key, pmt::from_double(change.second),
                         pmt::intern(alias()));
        ++d_next_change;
    }
    while (d_next_frame < d_frames.size()) {
        const auto& frame = d_frames[d_next_frame];
        if (frame.offset >= begin && frame.offset - begin >=
                                        static_cast<uint64_t>(noutput_items))
            break;
        if (frame.offset >= begin) {
            auto value = pmt::make_dict();
            value = pmt::dict_add(value, pmt::intern("frame_id"), pmt::intern(frame.id));
            value = pmt::dict_add(value, pmt::intern("sample_start"),
                                  pmt::from_uint64(frame.sample_start));
            value = pmt::dict_add(value, pmt::intern("sample_offset"),
                                  pmt::from_uint64(frame.sample_offset));
            value = pmt::dict_add(value, pmt::intern("timestamp_ns"),
                                  pmt::from_uint64(timestamp_ns));
            add_item_tag(0, frame.offset,
                         pmt::intern(frame.is_end ? "frame_end" : "frame_start"), value,
                         pmt::intern(alias()));
        }
        ++d_next_frame;
    }
    return noutput_items;
}

} // namespace mlir_aie
} // namespace gr
