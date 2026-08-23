/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_BFLOAT16_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_BFLOAT16_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_bfloat16.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <immintrin.h> 
#include <vector>

namespace gr {
namespace mlir_aie {
    
using gr_input_type = float;
using gr_output_type = float;

using aie_input_type = std::int16_t;
using aie_output_type = std::int16_t;

class mlir_aie_cpp_bfloat16_impl : public mlir_aie_cpp_bfloat16
{
private:
    struct io_slot {
        xrt::bo input_bo;
        xrt::bo output_bo;
        xrt::run run;
        aie_input_type* input = nullptr;
        aie_output_type* output = nullptr;
    };

    const char* _path_xclbin;
    const char* _path_insts_bin;
    int _VECTOR_SIZE;
    const char* _kernel_name;
    int _trace_size;
    unsigned int _opcode_run;
    xrt::kernel _kernel;
    xrt::bo _bo_instr;
    std::vector<uint32_t> _instr_v;
    xrt::device _device;
    std::vector<io_slot> _slots;

    void *bufInstr;

    static uint16_t float_to_bfloat16(float f) {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(uint32_t));
        // Truncate the lower 16 bits (standard bfloat16 conversion)
        return static_cast<uint16_t>(x >> 16);
    }

    static float bfloat16_to_float(uint16_t bf) {
        uint32_t x = static_cast<uint32_t>(bf) << 16;
        float f;
        std::memcpy(&f, &x, sizeof(uint32_t));
        return f;
    }
    #ifndef _mm512_castbh_si512
    static inline __m512i _mm512_castbh_si512(__m512bh a) {
        return (__m512i)a;
    }
    #endif
public:
    mlir_aie_cpp_bfloat16_impl(const char* path_xclbin,
                               const char* path_insts_bin,
                               const char* kernel_name,
                               int VECTOR_SIZE,
                               int num_slots);
    ~mlir_aie_cpp_bfloat16_impl();

    // Where all the action really happens
    void forecast(int noutput_items, gr_vector_int& ninput_items_required);

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items);
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_BFLOAT16_IMPL_H */
