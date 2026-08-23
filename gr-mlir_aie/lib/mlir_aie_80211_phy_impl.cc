/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_80211_phy_impl.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_80211_phy::sptr mlir_aie_80211_phy::make(const char* path_xclbin,
                                                   const char* path_insts_bin,
                                                   const char* kernel_name,
                                                   int VECTOR_SIZE,
                                                   double nominal_frequency,
                                                   int num_slots)
{
    return gnuradio::make_block_sptr<mlir_aie_80211_phy_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE, nominal_frequency, num_slots);
}

mlir_aie_80211_phy_impl::mlir_aie_80211_phy_impl(const char* path_xclbin,
                                                 const char* path_insts_bin,
                                                 const char* kernel_name,
                                                 int VECTOR_SIZE,
                                                 double nominal_frequency,
                                                 int num_slots)
    : gr::block("mlir_aie_80211_phy",
                 gr::io_signature::make(1, 1, sizeof(phy_input_type)),
                 gr::io_signature::make(1, 1, sizeof(phy_output_type))),
      _nominal_frequency(nominal_frequency)
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

    _slots.resize(num_slots);
    for (auto& slot : _slots) {
        slot.input_bo = xrt::bo(_device,
                                _VECTOR_SIZE * sizeof(phy_input_type),
                                XRT_BO_FLAGS_HOST_ONLY,
                                _kernel.group_id(3));
        slot.output_bo = xrt::bo(_device,
                                 _VECTOR_SIZE * sizeof(phy_output_type) + _trace_size,
                                 XRT_BO_FLAGS_HOST_ONLY,
                                 _kernel.group_id(3));
        slot.metadata_bo = xrt::bo(_device,
                                   _N_TILES * sizeof(tile_metadata),
                                   XRT_BO_FLAGS_HOST_ONLY,
                                   _kernel.group_id(3));

        slot.input = slot.input_bo.map<phy_input_type*>();
        slot.output = slot.output_bo.map<phy_output_type*>();
        slot.metadata = slot.metadata_bo.map<tile_metadata*>();
        std::memset(slot.output,
                    42,
                    _VECTOR_SIZE * sizeof(phy_output_type) + _trace_size);
        std::memset(slot.metadata, 0, _N_TILES * sizeof(tile_metadata));
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.metadata_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, _opcode_run);
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        slot.run.set_arg(4, slot.output_bo);
        slot.run.set_arg(5, slot.metadata_bo);
    }

    set_output_multiple(_VECTOR_SIZE);
}

mlir_aie_80211_phy_impl::~mlir_aie_80211_phy_impl() {}

void mlir_aie_80211_phy_impl::set_nominal_frequency(double nominal_frequency)
{
    _nominal_frequency.store(nominal_frequency);
}

double mlir_aie_80211_phy_impl::nominal_frequency() const
{
    return _nominal_frequency.load();
}

void mlir_aie_80211_phy_impl::forecast(int noutput_items,
                                       gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int mlir_aie_80211_phy_impl::general_work(int noutput_items,
                                          gr_vector_int& ninput_items,
                                          gr_vector_const_void_star& input_items,
                                          gr_vector_void_star& output_items)
{
    auto in = static_cast<const phy_input_type*>(input_items[0]);
    auto out = static_cast<phy_output_type*>(output_items[0]);

    const int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    const auto frame_bytes_key = pmt::intern("frame bytes");
    const auto encoding_key = pmt::intern("encoding");
    const auto snr_key = pmt::intern("snr");
    const auto nominal_frequency_key = pmt::intern("nominal frequency");
    const auto frequency_offset_key = pmt::intern("frequency offset");
    const auto beta_key = pmt::intern("beta");
    const auto csi_key = pmt::intern("csi");
    const auto tag_srcid = pmt::intern("frame_equalizer");
    constexpr double q16_15_scale = 1.0 / (std::int64_t{ 1 } << 15);
    constexpr double q29_scale = 1.0 / (std::int64_t{ 1 } << 29);
    constexpr double sample_rate = 20e6;
    constexpr double pi = 3.14159265358979323846;
    constexpr double snr_q4_scale = static_cast<double>(1 << 4);
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    const auto launch = [this, in](int chunk_idx) {
        auto& slot = _slots[chunk_idx % _slots.size()];
        std::memcpy(slot.input,
                    in + (chunk_idx * _VECTOR_SIZE),
                    _VECTOR_SIZE * sizeof(phy_input_type));
        slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.run.start();
    };

    const int depth = static_cast<int>(_slots.size());
    for (int i = 0; i < std::min(depth, n_chunks); ++i) {
        launch(i);
    }
    for (int i = 0; i < n_chunks; ++i) {
        auto& slot = _slots[i % depth];

        slot.run.wait();
        slot.output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        slot.metadata_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Keep the command queue fed while this slot is retired on the host.
        if (i + depth < n_chunks) {
            launch(i + depth);
        }

        for (int tile_idx = 0; tile_idx < _N_TILES; ++tile_idx) {
            const tile_metadata& tile_meta = slot.metadata[tile_idx];

            int tile_len = tile_meta.output_length;
            const int tag_count =
                std::clamp(tile_meta.tag_count, 0, _MAX_TAGS_PER_TILE);
            const int tile_start = tile_idx * _TILE_SIZE;

            if (tile_len < 0) {
                tile_len = 0;
            } else if (tile_len > _TILE_SIZE) {
                tile_len = _TILE_SIZE;
            }

            std::memcpy(out + total_produced,
                        slot.output + tile_start,
                        tile_len * sizeof(phy_output_type));

            const uint64_t tile_abs_start = output_abs_start + total_produced;
            for (int tag_idx = 0; tag_idx < tag_count; ++tag_idx) {
                const tag_metadata& tag = tile_meta.tags[tag_idx];

                if (0 <= tag.offset && tag.offset < tile_len) {
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
                    add_item_tag(
                        0,
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
                    add_item_tag(
                        0,
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
            }

            total_produced += tile_len;
        }
    }

    const int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);

    return total_produced;
}

} /* namespace mlir_aie */
} /* namespace gr */
