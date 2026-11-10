/* -*- c++ -*- */
/* Copyright 2026 gr-mlir_aie author.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_THRU_MAC_DECODE_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_THRU_MAC_DECODE_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_80211_phy_thru_mac_decode.h>
#include "phy_mac_metadata.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include <atomic>
#include <vector>

namespace gr {
namespace mlir_aie {

class mlir_aie_80211_phy_thru_mac_decode_impl
    : public mlir_aie_80211_phy_thru_mac_decode
{
private:
    struct io_slot {
        xrt::bo input_bo;
        xrt::bo output_bo;
        xrt::bo metadata_bo;
        xrt::run run;
        std::int32_t* input = nullptr;
        std::uint8_t* output = nullptr;
        phy_mac::tile_metadata* metadata = nullptr;
    };

    const int _VECTOR_SIZE;
    const int _N_TILES;
    std::atomic<double> _nominal_frequency;
    // Device must outlive its kernel, BOs, and runs (reverse destruction order).
    xrt::device _device;
    xrt::kernel _kernel;
    xrt::bo _bo_instr;
    std::vector<std::uint32_t> _instr_v;
    std::size_t _center_mhz_index;
    std::size_t _reciprocal_q30_index;
    std::uint32_t* _instructions = nullptr;
    std::vector<io_slot> _slots;

public:
    mlir_aie_80211_phy_thru_mac_decode_impl(const char* path_xclbin,
                                           const char* path_insts_bin,
                                           const char* kernel_name,
                                           int VECTOR_SIZE,
                                           double nominal_frequency,
                                           int num_slots,
                                           int N_TILES);
    ~mlir_aie_80211_phy_thru_mac_decode_impl() override;

    void forecast(int noutput_items, gr_vector_int& ninput_items_required) override;
    void set_nominal_frequency(double nominal_frequency) override;
    double nominal_frequency() const override;
    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items) override;
};

} // namespace mlir_aie
} // namespace gr
#endif
