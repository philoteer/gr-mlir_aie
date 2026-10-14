#ifndef INCLUDED_MLIR_AIE_PHY_FREQUENCY_TAGS_H
#define INCLUDED_MLIR_AIE_PHY_FREQUENCY_TAGS_H

#include "phy_frequency_rtp.h"
#include <gnuradio/tags.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

namespace gr {
namespace mlir_aie {

struct phy_frequency_batch {
    int n_chunks;
    std::optional<double> frequency;
    std::optional<double> next_frequency;
};

// End the batch at the first vector boundary at or after a frequency tag.
// An interior tag is consumed with its vector, so retain its value for the
// next batch. Never apply a future tag to samples preceding its offset.
inline phy_frequency_batch plan_phy_frequency_batch(std::vector<tag_t> tags,
                                                     std::uint64_t input_start,
                                                     int vector_size,
                                                     int n_chunks)
{
    std::stable_sort(tags.begin(), tags.end(), [](const tag_t& a, const tag_t& b) {
        return a.offset < b.offset;
    });
    phy_frequency_batch batch{ n_chunks, std::nullopt, std::nullopt };
    for (const auto& tag : tags) {
        if (tag.offset > input_start) {
            const auto relative_offset = tag.offset - input_start;
            const auto boundary = relative_offset / vector_size +
                                  (relative_offset % vector_size != 0);
            batch.n_chunks = static_cast<int>(
                std::min<std::uint64_t>(batch.n_chunks, boundary));
            break;
        }
    }
    const auto input_end = input_start +
                           static_cast<std::uint64_t>(batch.n_chunks) * vector_size;
    for (const auto& tag : tags) {
        if (tag.offset < input_start)
            continue;
        if (tag.offset >= input_end)
            break;
        const double frequency = pmt::to_double(tag.value);
        phy_frequency_rtp(frequency);
        if (tag.offset == input_start)
            batch.frequency = frequency;
        else
            batch.next_frequency = frequency;
    }
    return batch;
}

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_PHY_FREQUENCY_TAGS_H */
