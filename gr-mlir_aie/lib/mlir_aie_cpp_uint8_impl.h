/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_UINT8_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_UINT8_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_uint8.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>

namespace gr {
namespace mlir_aie {

using input_type = std::uint8_t;
using output_type = std::uint8_t;

class mlir_aie_cpp_uint8_impl : public mlir_aie_cpp_uint8
{
private:
    const char* _path_xclbin;
    const char* _path_insts_bin;
    int _VECTOR_SIZE;
    const char* _kernel_name;
    int _trace_size;
    unsigned int _opcode_run;
    xrt::kernel _kernel;
    xrt::bo _bo_instr, _bo_inA, _bo_out;
    std::vector<uint32_t> _instr_v;
    xrt::device _device;
    xrt::run _run;
    
    input_type *_bufInA ;        
    output_type *_bufOut ;
    void *bufInstr;

    
    // Nothing to declare in this block.

public:
    mlir_aie_cpp_uint8_impl(const char* path_xclbin,
                            const char* path_insts_bin,
                            int VECTOR_SIZE);
    ~mlir_aie_cpp_uint8_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_UINT8_IMPL_H */
