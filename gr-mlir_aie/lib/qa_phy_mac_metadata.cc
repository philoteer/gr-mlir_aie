/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "phy_mac_metadata.h"
#include <array>
#include <cstring>
#include <iostream>
#include <limits>

using namespace gr::mlir_aie::phy_mac;

namespace {
void check(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

}

int main()
{
    try {
        tile_metadata tile{};
        std::array<std::uint8_t, output_bytes_per_tile> output{};
        tile.frame_count = max_frames_per_tile;
        tile.output_length = max_frames_per_tile * 3;
        for (int i = 0; i < max_frames_per_tile; ++i) {
            auto& frame = tile.frames[i];
            frame.offset = 3 * i;
            frame.length = 3;
            frame.phy.frame_bytes = 7;
            frame.phy.encoding = i % 8;
            frame.phy.snr_linear = 320;
            frame.phy.center_frequency_mhz = 2412;
            frame.phy.frequency_offset = -(1 << 20);
            frame.phy.beta = 1 << 28;
            frame.phy.csi[0] = { 32768, -16384 };
            output[3 * i] = i;
        }
        for (int i = 0; i < max_frames_per_tile; ++i) {
            const auto message = make_pdu(output.data(), tile.frames[i]);
            const auto meta = pmt::car(message);
            const auto get = [&meta](const char* key) {
                return pmt::dict_ref(meta, pmt::intern(key), pmt::PMT_NIL);
            };
            check(pmt::is_blob(pmt::cdr(message)), "payload must be a blob");
            check(pmt::blob_length(pmt::cdr(message)) == 3, "wrong payload length");
            check(static_cast<const std::uint8_t*>(pmt::blob_data(pmt::cdr(message)))[0] == i,
                  "wrong payload offset/order");
            output[3 * i] = 255;
            check(static_cast<const std::uint8_t*>(pmt::blob_data(pmt::cdr(message)))[0] == i,
                  "PDU aliases a reused DMA buffer");
            check(pmt::to_uint64(get("frame bytes")) == 7, "PSDU includes original FCS length");
            check(pmt::to_uint64(get("encoding")) == static_cast<unsigned>(i % 8), "encoding");
            check(pmt::to_long(get("dlt")) == 105, "link type");
            check(pmt::to_double(get("snr")) == 10.0, "Q4.4 SNR conversion");
            check(pmt::to_double(get("nominal frequency")) == 2412e6, "frequency conversion");
            check(pmt::to_double(get("frequency offset")) < 0, "signed CFO conversion");
            check(pmt::to_double(get("beta")) == 0.5, "Q29 beta conversion");
            const auto csi = pmt::c32vector_elements(get("csi"));
            check(csi.size() == 52 && csi[0] == std::complex<float>(1.0f, -0.5f), "CSI conversion");
        }
        auto bad = tile;
        // FCS-only PSDUs are legal zero-length MAC blobs.
        tile = {};
        tile.frame_count = 1;
        tile.frames[0].phy.frame_bytes = 4;
        check(pmt::blob_length(pmt::cdr(make_pdu(output.data(), tile.frames[0]))) == 0,
              "empty MAC frame");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
