/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_fft_test_impl.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_fft_test::sptr mlir_aie_cpp_fft_test::make(const char* path_xclbin,
                                                        const char* path_insts_bin,
                                                        const char* kernel_name,
                                                        int VECTOR_SIZE)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_fft_test_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE);
}

mlir_aie_cpp_fft_test_impl::mlir_aie_cpp_fft_test_impl(const char* path_xclbin,
                                                       const char* path_insts_bin,
                                                       const char* kernel_name,
                                                       int VECTOR_SIZE)
    : gr::block("mlir_aie_cpp_fft_test",
                gr::io_signature::make(1, 1, sizeof(fft_input_type)),
                gr::io_signature::make(1, 1, sizeof(fft_output_type))),
      _VECTOR_SIZE(VECTOR_SIZE),
      _TILE_SIZE(VECTOR_SIZE / _N_TILES),
      _opcode_run(3)
{
    if (_VECTOR_SIZE <= 0 || _VECTOR_SIZE % _N_TILES != 0) {
        throw std::invalid_argument("VECTOR_SIZE must be a positive multiple of 4");
    }

    set_tag_propagation_policy(TPP_DONT);
    set_output_multiple(_VECTOR_SIZE);

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, kernel_name);

    _bo_instr = xrt::bo(_device,
                        _instr_v.size() * sizeof(uint32_t),
                        XCL_BO_FLAGS_CACHEABLE,
                        _kernel.group_id(1));
    _bo_in = xrt::bo(_device,
                     _VECTOR_SIZE * sizeof(fft_input_type),
                     XRT_BO_FLAGS_HOST_ONLY,
                     _kernel.group_id(3));
    _bo_in_meta = xrt::bo(_device,
                          _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t),
                          XRT_BO_FLAGS_HOST_ONLY,
                          _kernel.group_id(3));
    _bo_out = xrt::bo(_device,
                      _VECTOR_SIZE * sizeof(fft_output_type),
                      XRT_BO_FLAGS_HOST_ONLY,
                      _kernel.group_id(3));
    _bo_out_meta = xrt::bo(_device,
                           _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t),
                           XRT_BO_FLAGS_HOST_ONLY,
                           _kernel.group_id(3));

    auto* buf_instr = _bo_instr.map<void*>();
    std::memcpy(buf_instr, _instr_v.data(), _instr_v.size() * sizeof(uint32_t));
    _buf_in = _bo_in.map<fft_input_type*>();
    _buf_in_meta = _bo_in_meta.map<std::int32_t*>();
    _buf_out = _bo_out.map<fft_output_type*>();
    _buf_out_meta = _bo_out_meta.map<std::int32_t*>();

    std::memset(_buf_in, 0, _VECTOR_SIZE * sizeof(fft_input_type));
    std::memset(
        _buf_in_meta, 0, _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));
    std::memset(_buf_out, 0, _VECTOR_SIZE * sizeof(fft_output_type));
    std::memset(
        _buf_out_meta, 0, _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));

    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_out_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    _run = xrt::run(_kernel);
    _run.set_arg(0, _opcode_run);
    _run.set_arg(1, _bo_instr);
    _run.set_arg(2, _instr_v.size());
    _run.set_arg(3, _bo_in);
    _run.set_arg(4, _bo_in_meta);
    _run.set_arg(5, _bo_out);
    _run.set_arg(6, _bo_out_meta);
}

mlir_aie_cpp_fft_test_impl::~mlir_aie_cpp_fft_test_impl() {}

void mlir_aie_cpp_fft_test_impl::forecast(int noutput_items,
                                          gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = std::max(_VECTOR_SIZE, noutput_items);
}

int mlir_aie_cpp_fft_test_impl::general_work(int noutput_items,
                                             gr_vector_int& ninput_items,
                                             gr_vector_const_void_star& input_items,
                                             gr_vector_void_star& output_items)
{
    const auto* in = static_cast<const fft_input_type*>(input_items[0]);
    auto* out = static_cast<fft_output_type*>(output_items[0]);

    const int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    const auto tag_key = pmt::intern("wifi_start");
    const auto tag_srcid = pmt::intern("sync_short");
    const uint64_t input_abs_start = nitems_read(0);
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
        const int chunk_start = chunk_idx * _VECTOR_SIZE;
        const uint64_t chunk_abs_start = input_abs_start + chunk_start;
        std::memset(
            _buf_in_meta, 0, _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));

        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            const int src_start = chunk_start + tile_idx * _TILE_SIZE;
            const int src_end = src_start + _TILE_SIZE;
            const int dst_start = tile_idx * _TILE_SIZE;

            std::copy(in + src_start, in + src_end, _buf_in + dst_start);

            auto* tile_meta = _buf_in_meta + tile_idx * _METADATA_WORDS_PER_TILE;
            const uint64_t tile_abs_start = chunk_abs_start + tile_idx * _TILE_SIZE;
            std::vector<tag_t> tags;
            get_tags_in_range(
                tags, 0, tile_abs_start, tile_abs_start + _TILE_SIZE, tag_key);

            const int tag_count = std::min<int>(tags.size(), _MAX_TAGS_PER_TILE);
            tile_meta[0] = _TILE_SIZE;
            tile_meta[1] = tag_count;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                const double scaled = std::nearbyint(pmt::to_double(tags[tag_idx].value) *
                                                     (std::int64_t{ 1 } << 29));
                const double clipped = std::clamp(
                    scaled,
                    static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                    static_cast<double>(std::numeric_limits<std::int32_t>::max()));
                tile_meta[2 + 2 * tag_idx] =
                    static_cast<std::int32_t>(tags[tag_idx].offset - tile_abs_start);
                tile_meta[3 + 2 * tag_idx] = static_cast<std::int32_t>(clipped);
            }
        }

        _bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        _bo_in_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        _run.start();
        _run.wait();
        _bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        _bo_out_meta.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            const auto* tile_meta = _buf_out_meta + tile_idx * _METADATA_WORDS_PER_TILE;
            const int tile_len = std::clamp(tile_meta[0], 0, _TILE_SIZE);
            const int tag_count = std::clamp(tile_meta[1], 0, _MAX_TAGS_PER_TILE);
            const int tile_start = tile_idx * _TILE_SIZE;

            std::copy(_buf_out + tile_start,
                      _buf_out + tile_start + tile_len,
                      out + total_produced);

            const uint64_t tile_abs_start = output_abs_start + total_produced;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                const int tag_offset = tile_meta[2 + 2 * tag_idx];
                const double tag_value = static_cast<double>(tile_meta[3 + 2 * tag_idx]) /
                                         (std::int64_t{ 1 } << 29);
                if (tag_offset >= 0 && tag_offset < tile_len) {
                    add_item_tag(0,
                                 tile_abs_start + tag_offset,
                                 tag_key,
                                 pmt::from_double(tag_value),
                                 tag_srcid);
                }
            }
            total_produced += tile_len;
        }
    }

    consume_each(n_chunks * _VECTOR_SIZE);
    return total_produced;
}

} /* namespace mlir_aie */
} /* namespace gr */
