/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "latency_tag_updater_impl.h"
#include <gnuradio/io_signature.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <time.h>

namespace gr {
namespace mlir_aie {

namespace {
int checked_itemsize(size_t itemsize)
{
    if (itemsize == 0 || itemsize > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("latency_tag_updater: invalid item size");
    return static_cast<int>(itemsize);
}
} // namespace

latency_tag_updater::sptr latency_tag_updater::make(const std::string& name,
                                                  size_t itemsize)
{
    return gnuradio::make_block_sptr<latency_tag_updater_impl>(name, itemsize);
}

latency_tag_updater_impl::latency_tag_updater_impl(const std::string& name, size_t itemsize)
    : gr::sync_block("latency_tag_updater",
                     gr::io_signature::make(1, 1, checked_itemsize(itemsize)),
                     gr::io_signature::make(1, 1, checked_itemsize(itemsize))),
      d_itemsize(itemsize),
      d_name(pmt::intern(name))
{
    if (name.empty())
        throw std::invalid_argument("latency_tag_updater: name must not be empty");
    // Forward each tag explicitly so updated tags replace their originals
    // instead of being duplicated by automatic propagation.
    set_tag_propagation_policy(TPP_DONT);
}

/*
 * Our virtual destructor.
 */
latency_tag_updater_impl::~latency_tag_updater_impl() {}

int latency_tag_updater_impl::work(int noutput_items,
                                   gr_vector_const_void_star& input_items,
                                   gr_vector_void_star& output_items)
{
    timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        throw std::system_error(errno, std::generic_category(), "clock_gettime");
    const uint64_t timestamp_ns = static_cast<uint64_t>(now.tv_sec) *
                                      1000000000ULL + now.tv_nsec;
    std::memcpy(output_items[0], input_items[0],
                static_cast<size_t>(noutput_items) * d_itemsize);

    std::vector<tag_t> tags;
    const uint64_t begin = nitems_read(0);
    get_tags_in_range(tags, 0, begin, begin + static_cast<uint64_t>(noutput_items));
    for (auto& tag : tags) {
        if ((pmt::eq(tag.key, d_start_key) || pmt::eq(tag.key, d_end_key)) &&
            pmt::is_dict(tag.value) &&
            pmt::is_uint64(pmt::dict_ref(tag.value, d_timestamp_key, pmt::PMT_NIL))) {
            auto timestamps = pmt::dict_ref(tag.value, d_timestamps_key, pmt::PMT_NIL);
            if (!pmt::dict_has_key(tag.value, d_timestamps_key))
                timestamps = pmt::make_dict();
            if (pmt::is_dict(timestamps) && !pmt::dict_has_key(timestamps, d_name)) {
                timestamps = pmt::dict_add(timestamps, d_name, pmt::from_uint64(timestamp_ns));
                tag.value = pmt::dict_add(tag.value, d_timestamps_key, timestamps);
            }
        }
        tag.offset = nitems_written(0) + (tag.offset - begin);
        add_item_tag(0, tag);
    }
    return noutput_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
