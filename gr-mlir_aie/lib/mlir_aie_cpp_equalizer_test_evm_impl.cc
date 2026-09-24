/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_equalizer_test_evm_impl.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_equalizer_test_evm::sptr
mlir_aie_cpp_equalizer_test_evm::make(const char* path_xclbin,
                                  const char* path_insts_bin,
                                  const char* kernel_name,
                                   int VECTOR_SIZE,
                                   double nominal_frequency,
                                   int num_slots,
                                   int N_TILES,
                                   const char* weights_path)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_equalizer_test_evm_impl>(path_xclbin,
                                                                       path_insts_bin,
                                                                       kernel_name,
                                                                       VECTOR_SIZE,
                                                                       nominal_frequency,
                                                                       num_slots,
                                                                       N_TILES,
                                                                       weights_path);
}

mlir_aie_cpp_equalizer_test_evm_impl::mlir_aie_cpp_equalizer_test_evm_impl(
    const char* path_xclbin,
    const char* path_insts_bin,
    const char* kernel_name,
    int VECTOR_SIZE,
    double nominal_frequency,
    int num_slots,
    int N_TILES,
    const char* weights_path)
    : gr::block("mlir_aie_cpp_equalizer_test_evm",
                gr::io_signature::make(1, 1, sizeof(equalizer_evm_input_type)),
                gr::io_signature::make(1, 1, sizeof(equalizer_evm_output_type))),
      _VECTOR_SIZE(VECTOR_SIZE),
      _N_TILES(N_TILES),
      _TILE_SIZE(0),
      _nominal_frequency(nominal_frequency),
      _opcode_run(3)
{
    if (_N_TILES < 1) {
        throw std::invalid_argument("N_TILES must be at least 1");
    }
    if (_VECTOR_SIZE <= 0 || _VECTOR_SIZE % _N_TILES != 0) {
        throw std::invalid_argument("VECTOR_SIZE must be positive and divisible by N_TILES");
    }
    if (num_slots < 1) {
        throw std::invalid_argument("num_slots must be at least 1");
    }
    const bool streamed_weights = weights_path != nullptr && weights_path[0] != '\0';
    if (streamed_weights && _VECTOR_SIZE % 64 != 0) {
        throw std::invalid_argument("VECTOR_SIZE must be divisible by 64 for streamed weights");
    }
    _TILE_SIZE = _VECTOR_SIZE / _N_TILES;

    std::vector<std::uint8_t> expanded_weights;
    if (streamed_weights) {
        const auto model_weights = test_utils::load_binary(weights_path);
        if (model_weights.empty()) {
            throw std::invalid_argument("weights file must not be empty");
        }
        const auto symbol_count = static_cast<std::size_t>(_VECTOR_SIZE / 64);
        if (model_weights.size() >
            std::numeric_limits<std::size_t>::max() / symbol_count) {
            throw std::overflow_error("expanded weights size overflow");
        }
        expanded_weights.resize(model_weights.size() * symbol_count);
        for (std::size_t offset = 0; offset < expanded_weights.size();
             offset += model_weights.size()) {
            std::memcpy(expanded_weights.data() + offset,
                        model_weights.data(),
                        model_weights.size());
        }
    }

    set_tag_propagation_policy(TPP_DONT);
    set_output_multiple(_VECTOR_SIZE);

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    const double center_frequency_mhz = nominal_frequency / 1e6;
    if (!std::isfinite(center_frequency_mhz) || center_frequency_mhz <= 0.0 ||
        center_frequency_mhz > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("nominal_frequency must be a positive frequency in Hz");
    }
    const auto center_mhz = static_cast<std::int32_t>(std::llround(center_frequency_mhz));
    const auto reciprocal_q30 = static_cast<std::int32_t>(
        std::llround(20.0 / center_mhz * (std::int64_t{ 1 } << 30)));
    const auto patch_rtp = [this](std::uint32_t marker, std::int32_t value) {
        const auto it = std::find(_instr_v.begin(), _instr_v.end(), marker);
        if (it == _instr_v.end() ||
            std::find(std::next(it), _instr_v.end(), marker) != _instr_v.end()) {
            throw std::runtime_error("center-frequency RTP marker is missing or ambiguous");
        }
        *it = static_cast<std::uint32_t>(value);
    };
    patch_rtp(0x13579BDFu, center_mhz);
    patch_rtp(0x2468ACE0u, reciprocal_q30);
    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, kernel_name);

    _bo_instr = xrt::bo(_device,
                        _instr_v.size() * sizeof(std::uint32_t),
                        XCL_BO_FLAGS_CACHEABLE,
                        _kernel.group_id(1));
    bufInstr = _bo_instr.map<void*>();
    std::memcpy(bufInstr, _instr_v.data(), _instr_v.size() * sizeof(std::uint32_t));
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const auto input_metadata_size =
        _N_TILES * _FFT_METADATA_WORDS_PER_TILE * sizeof(std::int32_t);
    const auto output_metadata_size = _N_TILES * sizeof(tile_metadata);
    _slots.resize(num_slots);
    for (auto& slot : _slots) {
        slot.input_bo = xrt::bo(_device,
                                _VECTOR_SIZE * sizeof(kernel_input_type),
                                XRT_BO_FLAGS_HOST_ONLY,
                                _kernel.group_id(3));
        slot.input_meta_bo = xrt::bo(
            _device,
            input_metadata_size,
            XRT_BO_FLAGS_HOST_ONLY,
            _kernel.group_id(streamed_weights ? 4 : 3));
        if (streamed_weights) {
            slot.weights_bo = xrt::bo(_device,
                                      expanded_weights.size(),
                                      XRT_BO_FLAGS_HOST_ONLY,
                                      _kernel.group_id(5));
        }
        slot.output_bo = xrt::bo(_device,
                                  _VECTOR_SIZE * sizeof(equalizer_evm_output_type),
                                  XRT_BO_FLAGS_HOST_ONLY,
                                  _kernel.group_id(streamed_weights ? 6 : 3));
        slot.output_meta_bo = xrt::bo(
            _device,
            output_metadata_size,
            XRT_BO_FLAGS_HOST_ONLY,
            _kernel.group_id(streamed_weights ? 7 : 3));

        slot.input = slot.input_bo.map<kernel_input_type*>();
        slot.input_meta = slot.input_meta_bo.map<std::int32_t*>();
        slot.output = slot.output_bo.map<equalizer_evm_output_type*>();
        slot.output_meta = slot.output_meta_bo.map<tile_metadata*>();

        if (streamed_weights) {
            std::memcpy(slot.weights_bo.map<void*>(),
                        expanded_weights.data(),
                        expanded_weights.size());
            slot.weights_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        std::memset(slot.output, 0, _VECTOR_SIZE * sizeof(equalizer_evm_output_type));
        std::memset(slot.output_meta, 0, output_metadata_size);
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.output_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, _opcode_run);
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        slot.run.set_arg(4, slot.input_meta_bo);
        if (streamed_weights) {
            slot.run.set_arg(5, slot.weights_bo);
            slot.run.set_arg(6, slot.output_bo);
            slot.run.set_arg(7, slot.output_meta_bo);
        } else {
            slot.run.set_arg(5, slot.output_bo);
            slot.run.set_arg(6, slot.output_meta_bo);
        }
    }
}

mlir_aie_cpp_equalizer_test_evm_impl::~mlir_aie_cpp_equalizer_test_evm_impl() {}

void mlir_aie_cpp_equalizer_test_evm_impl::set_nominal_frequency(double nominal_frequency)
{
    _nominal_frequency.store(nominal_frequency);
}

double mlir_aie_cpp_equalizer_test_evm_impl::nominal_frequency() const
{
    return _nominal_frequency.load();
}

void mlir_aie_cpp_equalizer_test_evm_impl::forecast(int noutput_items,
                                                gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = std::max(_VECTOR_SIZE, noutput_items);
}

int mlir_aie_cpp_equalizer_test_evm_impl::general_work(int noutput_items,
                                                   gr_vector_int& ninput_items,
                                                   gr_vector_const_void_star& input_items,
                                                   gr_vector_void_star& output_items)
{
    const auto* in = static_cast<const equalizer_evm_input_type*>(input_items[0]);
    auto* out = static_cast<equalizer_evm_output_type*>(output_items[0]);

    const int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    const auto wifi_start_key = pmt::intern("wifi_start");
    const auto frame_bytes_key = pmt::intern("frame bytes");
    const auto encoding_key = pmt::intern("encoding");
    const auto snr_key = pmt::intern("snr");
    const auto nominal_frequency_key = pmt::intern("nominal frequency");
    const auto frequency_offset_key = pmt::intern("frequency offset");
    const auto beta_key = pmt::intern("beta");
    const auto csi_key = pmt::intern("csi");
    const auto tag_srcid = pmt::intern("frame_equalizer");
    constexpr double q16_15_scale = 1.0 / (std::int64_t{ 1 } << 15);
    constexpr double q29_full_scale = static_cast<double>(std::int64_t{ 1 } << 29);
    constexpr double q29_scale = 1.0 / q29_full_scale;
    constexpr double sample_rate = 20e6;
    constexpr double pi = 3.14159265358979323846;
    constexpr double snr_q4_scale = static_cast<double>(1 << 4);
    const uint64_t input_abs_start = nitems_read(0);
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    const auto to_int32 = [](double value) {
        const double rounded = std::nearbyint(value);
        const double clipped =
            std::clamp(rounded,
                       static_cast<double>(std::numeric_limits<std::int32_t>::min()),
                       static_cast<double>(std::numeric_limits<std::int32_t>::max()));
        return static_cast<std::int32_t>(clipped);
    };

    const int depth = static_cast<int>(_slots.size());
    const auto launch = [this, in, input_abs_start, wifi_start_key, to_int32](
                            int chunk_idx) {
        auto& slot = _slots[chunk_idx % _slots.size()];
        const int chunk_start = chunk_idx * _VECTOR_SIZE;
        const uint64_t chunk_abs_start = input_abs_start + chunk_start;

        for (int sample_idx = 0; sample_idx < _VECTOR_SIZE; ++sample_idx) {
            const auto sample = in[chunk_start + sample_idx];
            slot.input[sample_idx].real = to_int32(sample.real() * (1 << 15));
            slot.input[sample_idx].imag = to_int32(sample.imag() * (1 << 15));
        }

        std::memset(slot.input_meta,
                    0,
                    _N_TILES * _FFT_METADATA_WORDS_PER_TILE * sizeof(std::int32_t));
        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            auto* tile_meta = slot.input_meta + tile_idx * _FFT_METADATA_WORDS_PER_TILE;
            const uint64_t tile_abs_start = chunk_abs_start + tile_idx * _TILE_SIZE;
            std::vector<tag_t> tags;
            get_tags_in_range(
                tags, 0, tile_abs_start, tile_abs_start + _TILE_SIZE, wifi_start_key);

            const int tag_count = std::min<int>(tags.size(), _FFT_MAX_TAGS_PER_TILE);
            tile_meta[0] = _TILE_SIZE;
            tile_meta[1] = tag_count;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                tile_meta[2 + 2 * tag_idx] =
                    static_cast<std::int32_t>(tags[tag_idx].offset - tile_abs_start);
                tile_meta[3 + 2 * tag_idx] =
                    to_int32(pmt::to_double(tags[tag_idx].value) * q29_full_scale);
            }
        }

        slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.input_meta_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
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
            const tile_metadata& tile_meta = slot.output_meta[tile_idx];
            const int tile_len = std::clamp(tile_meta.output_length, 0, _TILE_SIZE);
            const int tag_count =
                std::clamp(tile_meta.tag_count, 0, _MAX_OUTPUT_TAGS_PER_TILE);
            const int tile_start = tile_idx * _TILE_SIZE;

            std::copy(slot.output + tile_start,
                      slot.output + tile_start + tile_len,
                      out + total_produced);

            const uint64_t tile_abs_start = output_abs_start + total_produced;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                const tag_metadata& tag = tile_meta.tags[tag_idx];
                if (tag.offset < 0 || tag.offset >= tile_len) {
                    continue;
                }

                const uint64_t tag_offset = tile_abs_start + tag.offset;
                std::vector<std::complex<float>> csi(_CSI_TAG_SIZE);
                for (int csi_idx = 0; csi_idx < _CSI_TAG_SIZE; ++csi_idx) {
                    csi[csi_idx] = {
                        static_cast<float>(tag.csi[csi_idx].real * q16_15_scale),
                        static_cast<float>(tag.csi[csi_idx].imag * q16_15_scale)
                    };
                }

                const double snr = 10.0 * std::log10(
                    static_cast<double>(tag.snr_linear) / (2.0 * snr_q4_scale));
                add_item_tag(0,
                             tag_offset,
                             frame_bytes_key,
                             pmt::from_uint64(tag.frame_bytes),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             encoding_key,
                             pmt::from_uint64(tag.encoding),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             snr_key,
                             pmt::from_double(snr),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             nominal_frequency_key,
                             pmt::from_double(tag.center_frequency_mhz * 1e6),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             frequency_offset_key,
                             pmt::from_double(tag.frequency_offset * q29_scale *
                                              sample_rate / (2.0 * pi)),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             beta_key,
                             pmt::from_double(tag.beta * q29_scale),
                             tag_srcid);
                add_item_tag(0,
                             tag_offset,
                             csi_key,
                             pmt::init_c32vector(csi.size(), csi),
                             tag_srcid);
            }
            total_produced += tile_len;
        }

        if (i + depth < n_chunks) {
            launch(i + depth);
        }
    }

    consume_each(n_chunks * _VECTOR_SIZE);
    return total_produced;
}

} /* namespace mlir_aie */
} /* namespace gr */
