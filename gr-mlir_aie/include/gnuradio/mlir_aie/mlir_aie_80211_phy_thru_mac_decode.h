/* -*- c++ -*- */
/* Copyright 2026 gr-mlir_aie author.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_THRU_MAC_DECODE_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_THRU_MAC_DECODE_H

#include <gnuradio/block.h>
#include <gnuradio/mlir_aie/api.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief Run the MLIR-AIE 802.11 PHY through CRC-validated MAC decoding.
 * \ingroup mlir_aie
 *
 * Input items are packed complex int16 samples (one int32 item per sample).
 * The message output "out" is compatible with ieee802_11.decode_mac: a pair
 * of a PHY metadata dictionary and a blob of MAC bytes, without SERVICE/FCS.
 * There is no stream output or CPU decoder fallback.
 *
 * VECTOR_SIZE and N_TILES must match the compiled graph, with 1024 samples
 * per tile. Only complete VECTOR_SIZE input batches are processed. Frequency
 * updates from the setter or rx_freq tags take effect at batch boundaries,
 * just as in mlir_aie_80211_phy. Use the paired MAC-output xclbin/insts.bin,
 * not an equalizer-only PHY binary.
 */
class MLIR_AIE_API mlir_aie_80211_phy_thru_mac_decode : virtual public gr::block
{
public:
    typedef std::shared_ptr<mlir_aie_80211_phy_thru_mac_decode> sptr;

    static sptr make(const char* path_xclbin,
                     const char* path_insts_bin,
                     const char* kernel_name,
                     int VECTOR_SIZE = 16384,
                     double nominal_frequency = 2.417e9,
                     int num_slots = 2,
                     int N_TILES = 16);

    virtual void set_nominal_frequency(double nominal_frequency) = 0;
    virtual double nominal_frequency() const = 0;
};

} // namespace mlir_aie
} // namespace gr
#endif
