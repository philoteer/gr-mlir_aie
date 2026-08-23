/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_SYNC_LONG_TEST_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_SYNC_LONG_TEST_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_sync_long_test.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstdint>
#include <vector>

namespace gr {
namespace mlir_aie {

using sync_long_input_type = std::int32_t;
using sync_long_output_type = std::int32_t;

class mlir_aie_cpp_sync_long_test_impl : public mlir_aie_cpp_sync_long_test
{
private:
    static constexpr int _MAX_TAGS_PER_TILE = 7;
    static constexpr int _N_TILES = 4;
    static constexpr int _METADATA_WORDS_PER_TILE = 2 + 2 * _MAX_TAGS_PER_TILE;
    static constexpr int _DELAY_SAMPLES = 320;

    int _VECTOR_SIZE;
    int _TILE_SIZE;
    int _IN_TILE_SIZE;
    int _IN_VECTOR_SIZE;
    unsigned int _opcode_run;
    xrt::kernel _kernel;
    xrt::bo _bo_instr, _bo_in, _bo_in_meta, _bo_out, _bo_out_meta;
    std::vector<uint32_t> _instr_v;
    std::vector<sync_long_input_type> _delay_history;
    xrt::device _device;
    xrt::run _run;

    sync_long_input_type* _buf_in;
    std::int32_t* _buf_in_meta;
    sync_long_output_type* _buf_out;
    std::int32_t* _buf_out_meta;

public:
    mlir_aie_cpp_sync_long_test_impl(const char* path_xclbin,
                                     const char* path_insts_bin,
                                     const char* kernel_name,
                                     int VECTOR_SIZE);
    ~mlir_aie_cpp_sync_long_test_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_SYNC_LONG_TEST_IMPL_H */
