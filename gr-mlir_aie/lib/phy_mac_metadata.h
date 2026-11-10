/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef INCLUDED_MLIR_AIE_PHY_MAC_METADATA_H
#define INCLUDED_MLIR_AIE_PHY_MAC_METADATA_H

#include <pmt/pmt.h>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gr {
namespace mlir_aie {
namespace phy_mac {

// Host mirror of base/kernel/common/{frame_equalizer,decode_mac}_metadata.h.
// Keep this ABI independent of the kernel source tree and its AIE headers.
constexpr int samples_per_tile = 1024;
constexpr int output_bytes_per_tile = 8192;
constexpr int max_frames_per_tile = 16;
constexpr int max_psdu = 4095;

struct csi_value {
    std::int32_t real;
    std::int32_t imag;
};

struct alignas(8) tag_metadata {
    std::int32_t offset;
    std::uint32_t frame_start;
    std::uint64_t frame_bytes;
    std::uint64_t encoding;
    std::uint64_t snr_linear;
    std::uint32_t center_frequency_mhz;
    std::int32_t frequency_offset;
    std::int32_t beta;
    csi_value csi[64];
};

struct alignas(8) frame_metadata {
    std::int32_t offset;
    std::int32_t length;
    tag_metadata phy;
};

struct alignas(8) tile_metadata {
    std::int32_t output_length;
    std::int32_t frame_count;
    frame_metadata frames[max_frames_per_tile];
};

static_assert(sizeof(tag_metadata) == 560, "PHY tag ABI changed");
static_assert(offsetof(tag_metadata, snr_linear) == 24, "PHY SNR ABI changed");
static_assert(offsetof(tag_metadata, center_frequency_mhz) == 32, "PHY frequency ABI changed");
static_assert(offsetof(tag_metadata, frequency_offset) == 36, "PHY CFO ABI changed");
static_assert(offsetof(tag_metadata, csi) == 44, "PHY CSI ABI changed");
static_assert(sizeof(frame_metadata) == 568, "MAC frame ABI changed");
static_assert(offsetof(frame_metadata, phy) == 8, "MAC PHY metadata ABI changed");
static_assert(sizeof(tile_metadata) == 9096, "MAC tile ABI changed");

inline pmt::pmt_t make_pdu(const std::uint8_t* output, const frame_metadata& frame)
{
    const auto& phy = frame.phy;
    auto meta = pmt::make_dict();
    const auto add = [&meta](const char* key, const pmt::pmt_t& value) {
        meta = pmt::dict_add(meta, pmt::intern(key), value);
    };
    add("frame bytes", pmt::from_uint64(phy.frame_bytes));
    add("encoding", pmt::from_uint64(phy.encoding));
    add("snr", pmt::from_double(10.0 * std::log10(phy.snr_linear / 32.0)));
    add("nominal frequency", pmt::from_double(phy.center_frequency_mhz * 1e6));
    constexpr double q29_scale = 1.0 / (std::int64_t{ 1 } << 29);
    constexpr double pi = 3.14159265358979323846;
    add("frequency offset", pmt::from_double(phy.frequency_offset * q29_scale *
                                           20e6 / (2.0 * pi)));
    add("beta", pmt::from_double(phy.beta * q29_scale));
    std::vector<std::complex<float>> csi(52);
    for (std::size_t i = 0; i < csi.size(); ++i)
        csi[i] = { phy.csi[i].real / 32768.0f, phy.csi[i].imag / 32768.0f };
    add("csi", pmt::init_c32vector(csi.size(), csi));
    add("dlt", pmt::from_long(105)); // LINKTYPE_IEEE802_11, like decode_mac.
    // make_blob copies the bytes before the asynchronous slot is reused.
    return pmt::cons(meta, pmt::make_blob(output + frame.offset, frame.length));
}

} // namespace phy_mac
} // namespace mlir_aie
} // namespace gr
#endif
