/* SPDX-License-Identifier: GPL-3.0-or-later */
#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_DECODE_MAC_TEST_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_DECODE_MAC_TEST_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_decode_mac_test.h>
#include "phy_mac_metadata.h"
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <map>
#include <vector>

namespace gr {
namespace mlir_aie {

// Host mirror of frame_equalizer_tile_metadata (input sideband).
struct alignas(8) decode_mac_input_metadata {
    std::int32_t output_length;
    std::int32_t tag_count;
    phy_mac::tag_metadata tags[16];
};
static_assert(sizeof(decode_mac_input_metadata) == 8968, "MAC input ABI changed");

class mlir_aie_cpp_decode_mac_test_impl : public mlir_aie_cpp_decode_mac_test
{
private:
    // Device must outlive the kernel, buffers, and run.
    xrt::device _device;
    xrt::kernel _kernel;
    xrt::bo _bo_instr, _bo_input, _bo_input_meta, _bo_output, _bo_output_meta;
    xrt::run _run;
    std::vector<std::uint32_t> _instr_v;
    std::uint8_t* _input = nullptr;
    decode_mac_input_metadata* _input_meta = nullptr;
    std::uint8_t* _output = nullptr;
    phy_mac::tile_metadata* _output_meta = nullptr;
    std::uint32_t _frame_id = 0;
    std::map<std::uint32_t, pmt::pmt_t> _frame_metadata;

    void run_tile();
    void pack_tags(int count);
    void publish_frames();

public:
    mlir_aie_cpp_decode_mac_test_impl(const char* path_xclbin,
                                      const char* path_insts_bin,
                                      const char* kernel_name);
    ~mlir_aie_cpp_decode_mac_test_impl() override;
    bool start() override;
    void forecast(int, gr_vector_int& ninput_items_required) override;
    int general_work(int, gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star&) override;
};
} // namespace mlir_aie
} // namespace gr
#endif
