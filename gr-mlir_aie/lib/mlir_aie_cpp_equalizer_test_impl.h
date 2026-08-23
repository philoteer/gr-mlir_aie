/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_IMPL_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_IMPL_H

#include <gnuradio/mlir_aie/mlir_aie_cpp_equalizer_test.h>

#include "runtime_lib/test_lib/test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include <atomic>
#include <cstddef>
#include <complex>
#include <cstdint>
#include <vector>

namespace gr {
namespace mlir_aie {

using equalizer_input_type = std::complex<float>;
using equalizer_output_type = std::int8_t;

class mlir_aie_cpp_equalizer_test_impl : public mlir_aie_cpp_equalizer_test
{
private:
    static constexpr int _N_TILES = 4;
    static constexpr int _FFT_MAX_TAGS_PER_TILE = 31;
    static constexpr int _FFT_METADATA_WORDS_PER_TILE = 2 + 2 * _FFT_MAX_TAGS_PER_TILE;
    static constexpr int _MAX_OUTPUT_TAGS_PER_TILE = 16;
    static constexpr int _CSI_SIZE = 64;
    static constexpr int _CSI_TAG_SIZE = 52;

    struct kernel_input_type {
        std::int32_t real;
        std::int32_t imag;
    };

    struct csi_value {
        std::int32_t real;
        std::int32_t imag;
    };

    struct alignas(8) tag_metadata {
        std::int32_t offset;
        std::uint32_t reserved;
        std::uint64_t frame_bytes;
        std::uint64_t encoding;
        std::uint64_t snr_linear;
        std::uint32_t center_frequency_mhz;
        std::int32_t frequency_offset;
        std::int32_t beta;
        csi_value csi[_CSI_SIZE];
    };

    struct alignas(8) tile_metadata {
        std::int32_t output_length;
        std::int32_t tag_count;
        tag_metadata tags[_MAX_OUTPUT_TAGS_PER_TILE];
    };

    static_assert(sizeof(kernel_input_type) == 8, "cint32 input ABI changed");
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

    int _VECTOR_SIZE;
    int _TILE_SIZE;
    std::atomic<double> _nominal_frequency;
    unsigned int _opcode_run;
    xrt::kernel _kernel;
    xrt::bo _bo_instr, _bo_in, _bo_in_meta, _bo_out, _bo_out_meta;
    std::vector<std::uint32_t> _instr_v;
    xrt::device _device;
    xrt::run _run;

    kernel_input_type* _buf_in;
    std::int32_t* _buf_in_meta;
    equalizer_output_type* _buf_out;
    tile_metadata* _buf_out_meta;

public:
    mlir_aie_cpp_equalizer_test_impl(const char* path_xclbin,
                                     const char* path_insts_bin,
                                     const char* kernel_name,
                                     int VECTOR_SIZE,
                                     double nominal_frequency);
    ~mlir_aie_cpp_equalizer_test_impl();

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

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_IMPL_H */
