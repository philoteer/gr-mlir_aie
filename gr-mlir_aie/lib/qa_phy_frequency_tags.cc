/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "phy_frequency_tags.h"
#include <iostream>
#include <stdexcept>
#include <string>

using gr::mlir_aie::plan_phy_frequency_batch;

namespace {
gr::tag_t frequency_tag(std::uint64_t offset, double frequency)
{
    gr::tag_t tag;
    tag.offset = offset;
    tag.key = pmt::intern("rx_freq");
    tag.value = pmt::from_double(frequency);
    return tag;
}

// Model successive scheduler calls and check every launched vector against
// the most recent capture frequency at that vector's first input sample.
void check_playback(std::vector<gr::tag_t> tags, std::uint64_t start)
{
    constexpr int vector_size = 64;
    constexpr int total_chunks = 6;
    constexpr double initial_frequency = 2412e6;
    double frequency = initial_frequency;
    int processed_chunks = 0;
    while (processed_chunks < total_chunks) {
        const auto begin = start + processed_chunks * vector_size;
        const auto end = start + total_chunks * vector_size;
        std::vector<gr::tag_t> available;
        for (const auto& tag : tags) {
            if (begin <= tag.offset && tag.offset < end)
                available.push_back(tag);
        }
        const auto batch = plan_phy_frequency_batch(
            available, begin, vector_size, total_chunks - processed_chunks);
        if (batch.n_chunks <= 0 || batch.n_chunks > total_chunks - processed_chunks)
            throw std::runtime_error("batch must make bounded forward progress");
        if (batch.frequency)
            frequency = *batch.frequency;
        for (int chunk = 0; chunk < batch.n_chunks; ++chunk) {
            const auto vector_start = begin + chunk * vector_size;
            double expected = initial_frequency;
            std::optional<std::uint64_t> latest_offset;
            for (const auto& tag : tags) {
                if (tag.offset <= vector_start &&
                    (!latest_offset || tag.offset >= *latest_offset)) {
                    latest_offset = tag.offset;
                    expected = pmt::to_double(tag.value);
                }
            }
            if (frequency != expected)
                throw std::runtime_error("wrong frequency at sample " +
                                         std::to_string(vector_start));
        }
        if (batch.next_frequency)
            frequency = *batch.next_frequency;
        processed_chunks += batch.n_chunks;
    }
}
} // namespace

int main()
{
    try {
        check_playback({}, 0);
        // Exhaust all aligned and interior positions, including the last
        // sample and a tag exactly outside the processed input range.
        for (std::uint64_t offset = 0; offset <= 6 * 64; ++offset)
            check_playback({ frequency_tag(offset, 2417000000.001862) }, 0);

        // Use offsets from the supplied capture (>32 bits), unsorted tags,
        // repeated frequencies, and several changes within a single vector.
        const std::uint64_t start = 18001116414ULL;
        check_playback({ frequency_tag(start + 129, 2432e6),
                         frequency_tag(start + 65, 2427e6),
                         frequency_tag(start, 2417000000.001862),
                         frequency_tag(start + 2, 2422e6),
                         frequency_tag(start + 128, 2427e6),
                         frequency_tag(start + 63, 2417000000.001862),
                         frequency_tag(start + 320, 2412e6) },
                       start);

        bool rejected = false;
        try {
            plan_phy_frequency_batch({ frequency_tag(0, -2412e6) }, 0, 64, 1);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected)
            throw std::runtime_error("invalid frequency was accepted");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
