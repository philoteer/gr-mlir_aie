/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_tagged_int32_to_int64_impl.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_tagged_int32_to_int64::sptr
mlir_aie_cpp_tagged_int32_to_int64::make(const char* path_xclbin,
                                         const char* path_insts_bin,
                                         const char* kernel_name,
                                         int VECTOR_SIZE,
                                         int num_slots)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_tagged_int32_to_int64_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE, num_slots);
}

mlir_aie_cpp_tagged_int32_to_int64_impl::mlir_aie_cpp_tagged_int32_to_int64_impl(
    const char* path_xclbin,
    const char* path_insts_bin,
    const char* kernel_name,
    int VECTOR_SIZE,
    int num_slots)
    : gr::block(
          "mlir_aie_cpp_tagged_int32_to_int64",
          gr::io_signature::make(
              1, 1, sizeof(tagged_int64_input_type)),
          gr::io_signature::make(
              1, 1, sizeof(tagged_int64_output_type)))
{
    if (num_slots < 1) {
        throw std::invalid_argument("num_slots must be at least 1");
    }

    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    _VECTOR_SIZE = VECTOR_SIZE;
    _TILE_SIZE = _VECTOR_SIZE / _N_TILES;
    _kernel_name = kernel_name;
    _trace_size = 0;
    _opcode_run = 3;

    set_tag_propagation_policy(TPP_DONT);

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    std::cout << "Sequence instr count: " << _instr_v.size() << "\n";

    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, _kernel_name);

    std::cout << "kernel load ok";
    _bo_instr = xrt::bo(_device,
                        _instr_v.size() * sizeof(int),
                        XCL_BO_FLAGS_CACHEABLE,
                        _kernel.group_id(1));
    std::cout << "Writing data into buffer objects.\n";

    bufInstr = _bo_instr.map<void*>();
    std::memcpy(bufInstr, _instr_v.data(), _instr_v.size() * sizeof(int));
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const auto output_metadata_size =
        _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t);
    _slots.resize(num_slots);
    for (auto& slot : _slots) {
        slot.input_bo = xrt::bo(_device,
                                _VECTOR_SIZE * sizeof(tagged_int64_input_type),
                                XRT_BO_FLAGS_HOST_ONLY,
                                _kernel.group_id(3));
        slot.output_bo = xrt::bo(_device,
                                 _VECTOR_SIZE * sizeof(tagged_int64_output_type) +
                                     _trace_size,
                                 XRT_BO_FLAGS_HOST_ONLY,
                                 _kernel.group_id(3));
        slot.output_meta_bo = xrt::bo(_device,
                                      output_metadata_size,
                                      XRT_BO_FLAGS_HOST_ONLY,
                                      _kernel.group_id(3));

        slot.input = slot.input_bo.map<tagged_int64_input_type*>();
        slot.output = slot.output_bo.map<tagged_int64_output_type*>();
        slot.output_meta = slot.output_meta_bo.map<std::int32_t*>();
        std::memset(slot.output,
                    42,
                    _VECTOR_SIZE * sizeof(tagged_int64_output_type) + _trace_size);
        std::memset(slot.output_meta, 0, output_metadata_size);
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.output_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, _opcode_run);
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        slot.run.set_arg(4, slot.output_bo);
        slot.run.set_arg(5, slot.output_meta_bo);
    }
}

mlir_aie_cpp_tagged_int32_to_int64_impl::~mlir_aie_cpp_tagged_int32_to_int64_impl() {}

void mlir_aie_cpp_tagged_int32_to_int64_impl::forecast(
    int noutput_items, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int mlir_aie_cpp_tagged_int32_to_int64_impl::general_work(
    int noutput_items,
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    auto in = static_cast<const tagged_int64_input_type*>(input_items[0]);
    auto out = static_cast<tagged_int64_output_type*>(output_items[0]);

    const int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    const auto tag_key = pmt::intern("wifi_start");
    const auto tag_srcid = pmt::intern("sync_short");
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    const int depth = static_cast<int>(_slots.size());
    const auto launch = [this, in](int chunk_idx) {
        auto& slot = _slots[chunk_idx % _slots.size()];
        std::memcpy(slot.input,
                    in + (chunk_idx * _VECTOR_SIZE),
                    _VECTOR_SIZE * sizeof(tagged_int64_input_type));
        slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.run.start();
    };

    for (int i = 0; i < std::min(depth, n_chunks); ++i) {
        launch(i);
    }
    for (int i = 0; i < n_chunks; ++i) {
        auto& slot = _slots[i % depth];
        slot.run.wait();
        slot.output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        slot.output_meta_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            const std::int32_t* tile_meta =
                slot.output_meta + (tile_idx * _METADATA_WORDS_PER_TILE);

            int tile_len = tile_meta[0];
            const int tag_count = std::clamp(tile_meta[1], 0, _MAX_TAGS_PER_TILE);
            const int tile_start = tile_idx * _TILE_SIZE;

            if (tile_len < 0) {
                tile_len = 0;
            } else if (tile_len > _TILE_SIZE) {
                tile_len = _TILE_SIZE;
            }

            std::memcpy(out + total_produced,
                        slot.output + tile_start,
                        tile_len * sizeof(tagged_int64_output_type));

            const uint64_t tile_abs_start = output_abs_start + total_produced;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                const int tag_offset = tile_meta[2 + 2 * tag_idx];
                const double tag_value =
                    static_cast<double>(tile_meta[3 + 2 * tag_idx]) /
                    (std::int64_t{ 1 } << 29);

                if (0 <= tag_offset && tag_offset < tile_len) {
                    add_item_tag(0,
                                 tile_abs_start + tag_offset,
                                 tag_key,
                                 pmt::from_double(tag_value),
                                 tag_srcid);
                }
            }

            total_produced += tile_len;
        }

        if (i + depth < n_chunks) {
            launch(i + depth);
        }
    }

    const int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);

    return total_produced;
}

} /* namespace mlir_aie */
} /* namespace gr */
