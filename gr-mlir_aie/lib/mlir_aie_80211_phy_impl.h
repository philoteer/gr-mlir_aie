/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_80211_phy.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gr {
namespace mlir_aie {

using phy_input_type = std::int32_t;
using phy_output_type = std::int8_t;

class mlir_aie_80211_phy_impl : public mlir_aie_80211_phy
{
private:
    static constexpr int _MAX_TAGS_PER_TILE = 16;
    static constexpr int _CSI_SIZE = 64;
    static constexpr int _CSI_TAG_SIZE = 52;

    struct csi_value {
        std::int32_t real;
        std::int32_t imag;
    };

    struct tag_metadata {
        std::int32_t offset;
        std::uint32_t frame_start;
        std::uint64_t frame_bytes;
        std::uint64_t encoding;
        std::uint64_t snr_linear;
        std::uint32_t center_frequency_mhz;
        std::int32_t frequency_offset;
        std::int32_t beta;
        csi_value csi[_CSI_SIZE];
    };

    struct tile_metadata {
        std::int32_t output_length;
        std::int32_t tag_count;
        tag_metadata tags[_MAX_TAGS_PER_TILE];
    };

    struct io_slot {
        xrt::bo input_bo;
        xrt::bo weights_bo;
        xrt::bo output_bo;
        xrt::bo metadata_bo;
        xrt::run run;
        phy_input_type* input = nullptr;
        phy_output_type* output = nullptr;
        tile_metadata* metadata = nullptr;
    };

    static_assert(sizeof(tag_metadata) == 560,
                  "frame equalizer tag metadata ABI changed");
    static_assert(offsetof(tag_metadata, snr_linear) == 24,
                  "frame equalizer Q4.4 SNR ABI changed");
    static_assert(offsetof(tag_metadata, center_frequency_mhz) == 32,
                  "frame equalizer center frequency ABI changed");
    static_assert(offsetof(tag_metadata, frequency_offset) == 36,
                  "frame equalizer frequency offset ABI changed");
    static_assert(offsetof(tag_metadata, csi) == 44,
                  "frame equalizer CSI ABI changed");
    static_assert(sizeof(tile_metadata) == 8968,
                  "frame equalizer tile metadata ABI changed");

    const char* _path_xclbin;
    const char* _path_insts_bin;
    int _VECTOR_SIZE;
    const int _N_TILES;
    int _TILE_SIZE;
    std::atomic<double> _nominal_frequency;
    const char* _kernel_name;
    int _trace_size;
    unsigned int _opcode_run;
    xrt::kernel _kernel;
    xrt::bo _bo_instr;
    std::vector<uint32_t> _instr_v;
    xrt::device _device;
    std::vector<io_slot> _slots;

    void* bufInstr;

public:
    mlir_aie_80211_phy_impl(const char* path_xclbin,
                             const char* path_insts_bin,
                             const char* kernel_name,
                              int VECTOR_SIZE,
                              double nominal_frequency,
                              int num_slots,
                              int N_TILES,
                              const char* weights_path);
    ~mlir_aie_80211_phy_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    void set_nominal_frequency(double nominal_frequency) override;
    double nominal_frequency() const override;

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_80211_PHY_IMPL_H */
