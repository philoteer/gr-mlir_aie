/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_tagged_int64_to_int64_impl.h"
#include "phy_frequency_rtp.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <iterator>
#include <limits>
#include <iostream>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_tagged_int64_to_int64::sptr
mlir_aie_cpp_tagged_int64_to_int64::make(const char* path_xclbin,
                                      const char* path_insts_bin,
                                      const char* kernel_name,
                                      int VECTOR_SIZE,
                                      int num_slots,
                                      int N_TILES,
                                      bool preprocess,
                                      double nominal_frequency,
                                      bool channel_estimate)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_tagged_int64_to_int64_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE, num_slots, N_TILES,
        preprocess, nominal_frequency, channel_estimate);
}

mlir_aie_cpp_tagged_int64_to_int64_impl::mlir_aie_cpp_tagged_int64_to_int64_impl(
    const char* path_xclbin,
    const char* path_insts_bin,
    const char* kernel_name,
    int VECTOR_SIZE,
    int num_slots,
    int N_TILES,
    bool preprocess,
    double nominal_frequency,
    bool channel_estimate)
    : gr::block("mlir_aie_cpp_tagged_int64_to_int64",
                gr::io_signature::make(1, 1, sizeof(input_type)),
                gr::io_signature::make(1, 1, sizeof(output_type))),
      _preprocess(preprocess),
      _channel_estimate(channel_estimate),
      _output_meta_words(channel_estimate ? _ESTIMATED_WORDS_PER_TILE :
                         preprocess ? _PREPARED_WORDS_PER_TILE : _METADATA_WORDS_PER_TILE),
      _VECTOR_SIZE(VECTOR_SIZE),
      _N_TILES(N_TILES),
      _TILE_SIZE(0)
{
    if (num_slots < 1) {
        throw std::invalid_argument("num_slots must be at least 1");
    }

    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    if (_N_TILES < 1) {
        throw std::invalid_argument("N_TILES must be at least 1");
    }
    if (_VECTOR_SIZE <= 0 || _VECTOR_SIZE % _N_TILES != 0) {
        throw std::invalid_argument("VECTOR_SIZE must be positive and divisible by N_TILES");
    }
    _TILE_SIZE = _VECTOR_SIZE / _N_TILES;
    if (_preprocess && _channel_estimate) {
        throw std::invalid_argument("preprocess and channel_estimate are mutually exclusive");
    }
    if ((_preprocess || _channel_estimate) && _TILE_SIZE != 1024) {
        throw std::invalid_argument("split frame equalizer requires 1024-sample tiles");
    }
    _kernel_name = kernel_name;
    set_output_multiple(_VECTOR_SIZE);
    _trace_size = 0;
    _opcode_run = 3;

    set_tag_propagation_policy(TPP_DONT);

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    if (_preprocess) {
        const auto [center_mhz, reciprocal_q30] = phy_frequency_rtp(nominal_frequency);
        for (const auto& [marker, value] : {
                 std::pair<std::uint32_t, std::int32_t>{0x13579BDFu, center_mhz},
                 std::pair<std::uint32_t, std::int32_t>{0x2468ACE0u, reciprocal_q30}}) {
            const auto it = std::find(_instr_v.begin(), _instr_v.end(), marker);
            if (it == _instr_v.end() ||
                std::find(std::next(it), _instr_v.end(), marker) != _instr_v.end()) {
                throw std::runtime_error("center-frequency RTP marker is missing or ambiguous");
            }
            *it = static_cast<std::uint32_t>(value);
        }
    }
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
        _N_TILES * _output_meta_words * sizeof(std::int32_t);
    _slots.resize(num_slots);
    for (auto& slot : _slots) {
        slot.input_bo = xrt::bo(_device,
                                _VECTOR_SIZE * sizeof(input_type),
                                XRT_BO_FLAGS_HOST_ONLY,
                                _kernel.group_id(3));
        slot.output_bo = xrt::bo(_device,
                                 _VECTOR_SIZE * sizeof(output_type) + _trace_size,
                                 XRT_BO_FLAGS_HOST_ONLY,
                                 _kernel.group_id(3));
        slot.output_meta_bo = xrt::bo(_device,
                                      output_metadata_size,
                                      XRT_BO_FLAGS_HOST_ONLY,
                                      _kernel.group_id(3));

        slot.input = slot.input_bo.map<input_type*>();
        if (_preprocess || _channel_estimate) {
            slot.input_meta_bo = xrt::bo(_device,
                                        _N_TILES * (_channel_estimate ? _PREPARED_WORDS_PER_TILE :
                                                    _METADATA_WORDS_PER_TILE) * sizeof(std::int32_t),
                                        XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(3));
            slot.input_meta = slot.input_meta_bo.map<std::int32_t*>();
        }
        slot.output = slot.output_bo.map<output_type*>();
        slot.output_meta = slot.output_meta_bo.map<std::int32_t*>();
        std::memset(slot.output, 42, _VECTOR_SIZE * sizeof(output_type) + _trace_size);
        std::memset(slot.output_meta, 0, output_metadata_size);
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.output_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, _opcode_run);
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        if (_channel_estimate) {
            slot.run.set_arg(4, slot.input_meta_bo);
            slot.run.set_arg(5, slot.output_meta_bo);
        } else if (_preprocess) {
            slot.run.set_arg(4, slot.input_meta_bo);
            slot.run.set_arg(5, slot.output_bo);
            slot.run.set_arg(6, slot.output_meta_bo);
        } else {
            slot.run.set_arg(4, slot.output_bo);
            slot.run.set_arg(5, slot.output_meta_bo);
        }
    }
}

mlir_aie_cpp_tagged_int64_to_int64_impl::~mlir_aie_cpp_tagged_int64_to_int64_impl() {}

void mlir_aie_cpp_tagged_int64_to_int64_impl::forecast(
    int noutput_items, gr_vector_int& ninput_items_required)
{
    // A single complete batch is enough, even when the output buffer is larger.
    // Requiring noutput_items would discard complete trailing batches at EOF.
    ninput_items_required[0] = _VECTOR_SIZE;
}

int mlir_aie_cpp_tagged_int64_to_int64_impl::general_work(
    int noutput_items,
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    const int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    const auto tag_key = pmt::intern("wifi_start");
    const auto tag_srcid = pmt::intern("sync_short");
    const auto prepared_key = pmt::intern("frame_equalizer_prepared");
    const auto estimated_key = pmt::intern("frame_equalizer_estimated");
    const uint64_t input_abs_start = nitems_read(0);
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    const int depth = static_cast<int>(_slots.size());
    const auto launch = [this, in, input_abs_start, tag_key, prepared_key](int chunk_idx) {
        auto& slot = _slots[chunk_idx % _slots.size()];
        std::memcpy(slot.input,
                    in + (chunk_idx * _VECTOR_SIZE),
                    _VECTOR_SIZE * sizeof(input_type));
        slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (_channel_estimate) {
            for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
                const auto start = input_abs_start + chunk_idx * _VECTOR_SIZE + tile_idx * _TILE_SIZE;
                std::vector<tag_t> tags;
                get_tags_in_range(tags, 0, start, start + 1, prepared_key);
                if (tags.size() != 1 || !pmt::is_s32vector(tags[0].value)) {
                    throw std::runtime_error("missing or ambiguous prepared metadata at tile boundary");
                }
                std::size_t count;
                const auto* words = pmt::s32vector_elements(tags[0].value, count);
                if (count != _PREPARED_WORDS_PER_TILE || words[0] != _TILE_SIZE ||
                    words[1] < 0 || words[1] > _MAX_TAGS_PER_TILE) {
                    throw std::runtime_error("channel estimator prepared metadata ABI mismatch");
                }
                std::copy(words, words + count, slot.input_meta + tile_idx * count);
            }
            slot.input_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        } else if (_preprocess) {
            std::memset(slot.input_meta, 0,
                        _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));
            for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
                auto* meta = slot.input_meta + tile_idx * _METADATA_WORDS_PER_TILE;
                const auto start = input_abs_start + chunk_idx * _VECTOR_SIZE + tile_idx * _TILE_SIZE;
                std::vector<tag_t> tags;
                get_tags_in_range(tags, 0, start, start + _TILE_SIZE, tag_key);
                if (tags.size() > _MAX_TAGS_PER_TILE) {
                    throw std::runtime_error("too many wifi_start tags for preprocess tile");
                }
                meta[0] = _TILE_SIZE;
                meta[1] = tags.size();
                for (std::size_t i = 0; i < tags.size(); ++i) {
                    meta[2 + 2 * i] = tags[i].offset - start;
                    meta[3 + 2 * i] = static_cast<std::int32_t>(std::clamp(
                        std::nearbyint(pmt::to_double(tags[i].value) * (std::int64_t{1} << 29)),
                        static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                        static_cast<double>(std::numeric_limits<std::int32_t>::max())));
                }
            }
            slot.input_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        slot.run.start();
    };

    for (int i = 0; i < std::min(depth, n_chunks); ++i) {
        launch(i);
    }
    for (int i = 0; i < n_chunks; ++i) {
        auto& slot = _slots[i % depth];
        slot.run.wait();
        if (!_channel_estimate) {
            slot.output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        }
        slot.output_meta_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            const std::int32_t* result_meta =
                slot.output_meta + (tile_idx * _output_meta_words);
            const auto* tile_meta = result_meta + (_channel_estimate ? _CHANNEL_WORDS_PER_TILE : 0);

            int tile_len = tile_meta[0];
            const int tag_count = std::clamp(tile_meta[1], 0, _MAX_TAGS_PER_TILE);
            const int tile_start = tile_idx * _TILE_SIZE;

            if (tile_len < 0) {
                tile_len = 0;
            } else if (tile_len > _TILE_SIZE) {
                tile_len = _TILE_SIZE;
            }

            std::memcpy(out + total_produced,
                        (_channel_estimate ? slot.input : slot.output) + tile_start,
                        tile_len * sizeof(output_type));

            const uint64_t tile_abs_start = output_abs_start + total_produced;
            if (_preprocess || _channel_estimate) {
                if (tile_len != _TILE_SIZE) {
                    throw std::runtime_error("split frame equalizer must preserve complete physical tiles");
                }
                add_item_tag(0, tile_abs_start, _channel_estimate ? estimated_key : prepared_key,
                             pmt::init_s32vector(_channel_estimate ? _ESTIMATED_WORDS_PER_TILE :
                                                                   _PREPARED_WORDS_PER_TILE, result_meta),
                             pmt::intern(_channel_estimate ? "frame_equalizer_channel_estimate" :
                                                            "frame_equalizer_preprocess"));
            }
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
