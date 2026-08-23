/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT32_TO_INT64_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT32_TO_INT64_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_tagged_int32_to_int64.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstdint>
#include <vector>

namespace gr {
namespace mlir_aie {

using tagged_int64_input_type = std::int32_t;
using tagged_int64_output_type = std::int64_t;

class mlir_aie_cpp_tagged_int32_to_int64_impl : public mlir_aie_cpp_tagged_int32_to_int64
{
private:
    struct io_slot {
        xrt::bo input_bo;
        xrt::bo output_bo;
        xrt::bo output_meta_bo;
        xrt::run run;
        tagged_int64_input_type* input = nullptr;
        tagged_int64_output_type* output = nullptr;
        std::int32_t* output_meta = nullptr;
    };

    static constexpr int _MAX_TAGS_PER_TILE = 31;
    static constexpr int _N_TILES = 4;
    static constexpr int _METADATA_WORDS_PER_TILE = 2 + 2 * _MAX_TAGS_PER_TILE;

    const char* _path_xclbin;
    const char* _path_insts_bin;
    int _VECTOR_SIZE;
    int _TILE_SIZE;
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
    mlir_aie_cpp_tagged_int32_to_int64_impl(const char* path_xclbin,
                                             const char* path_insts_bin,
                                             const char* kernel_name,
                                             int VECTOR_SIZE,
                                             int num_slots);
    ~mlir_aie_cpp_tagged_int32_to_int64_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_TAGGED_INT32_TO_INT64_IMPL_H */
