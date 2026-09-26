#ifndef INCLUDED_MLIR_AIE_PHY_FREQUENCY_RTP_H
#define INCLUDED_MLIR_AIE_PHY_FREQUENCY_RTP_H

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gr {
namespace mlir_aie {

inline std::pair<std::int32_t, std::int32_t> phy_frequency_rtp(double frequency_hz)
{
    const double center_frequency_mhz = frequency_hz / 1e6;
    if (!std::isfinite(center_frequency_mhz) || center_frequency_mhz <= 0.0 ||
        center_frequency_mhz > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("nominal_frequency must be a positive frequency in Hz");
    }
    const auto center_mhz = static_cast<std::int32_t>(std::llround(center_frequency_mhz));
    const auto reciprocal_q30 = static_cast<std::int32_t>(
        std::llround(20.0 / center_mhz * (std::int64_t{ 1 } << 30)));
    return { center_mhz, reciprocal_q30 };
}

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_PHY_FREQUENCY_RTP_H */
