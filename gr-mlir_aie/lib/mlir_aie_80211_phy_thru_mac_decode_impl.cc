/* -*- c++ -*- */
/* Copyright 2026 gr-mlir_aie author.
 * SPDX-License-Identifier: GPL-3.0-or-later
 */
#include "mlir_aie_80211_phy_thru_mac_decode_impl.h"
#include "phy_frequency_rtp.h"
#include "phy_frequency_tags.h"
#include "runtime_lib/test_lib/test_utils.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_80211_phy_thru_mac_decode::sptr
mlir_aie_80211_phy_thru_mac_decode::make(const char* path_xclbin,
                                       const char* path_insts_bin,
                                       const char* kernel_name,
                                       int VECTOR_SIZE,
                                       double nominal_frequency,
                                       int num_slots,
                                       int N_TILES)
{
    return gnuradio::make_block_sptr<mlir_aie_80211_phy_thru_mac_decode_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE,
        nominal_frequency, num_slots, N_TILES);
}

mlir_aie_80211_phy_thru_mac_decode_impl::mlir_aie_80211_phy_thru_mac_decode_impl(
    const char* path_xclbin,
    const char* path_insts_bin,
    const char* kernel_name,
    int VECTOR_SIZE,
    double nominal_frequency,
    int num_slots,
    int N_TILES)
    : gr::block("mlir_aie_80211_phy_thru_mac_decode",
                gr::io_signature::make(1, 1, sizeof(std::int32_t)),
                gr::io_signature::make(0, 0, 0)),
      _VECTOR_SIZE(VECTOR_SIZE),
      _N_TILES(N_TILES),
      _nominal_frequency(nominal_frequency)
{
    if (num_slots < 1)
        throw std::invalid_argument("num_slots must be at least 1");
    if (N_TILES < 1 || VECTOR_SIZE <= 0 || VECTOR_SIZE % N_TILES != 0 ||
        VECTOR_SIZE / N_TILES != phy_mac::samples_per_tile)
        throw std::invalid_argument("MAC decoder requires VECTOR_SIZE = N_TILES * 1024");
    const auto [center_mhz, reciprocal_q30] = phy_frequency_rtp(nominal_frequency);
    message_port_register_out(pmt::intern("out"));
    set_tag_propagation_policy(TPP_DONT);

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    const auto locate_rtp = [this](std::uint32_t marker) {
        const auto it = std::find(_instr_v.begin(), _instr_v.end(), marker);
        if (it == _instr_v.end() ||
            std::find(std::next(it), _instr_v.end(), marker) != _instr_v.end())
            throw std::runtime_error("center-frequency RTP marker is missing or ambiguous");
        return static_cast<std::size_t>(std::distance(_instr_v.begin(), it));
    };
    _center_mhz_index = locate_rtp(0x13579BDFu);
    _reciprocal_q30_index = locate_rtp(0x2468ACE0u);
    _instr_v[_center_mhz_index] = static_cast<std::uint32_t>(center_mhz);
    _instr_v[_reciprocal_q30_index] = static_cast<std::uint32_t>(reciprocal_q30);

    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, kernel_name);
    _bo_instr = xrt::bo(_device, _instr_v.size() * sizeof(std::uint32_t),
                        XCL_BO_FLAGS_CACHEABLE, _kernel.group_id(1));
    _instructions = _bo_instr.map<std::uint32_t*>();
    std::memcpy(_instructions, _instr_v.data(), _instr_v.size() * sizeof(std::uint32_t));
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    _slots.resize(num_slots);
    for (auto& slot : _slots) {
        slot.input_bo = xrt::bo(_device,
                                static_cast<std::size_t>(_VECTOR_SIZE) * sizeof(std::int32_t),
                                XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(3));
        slot.output_bo = xrt::bo(_device,
                                 static_cast<std::size_t>(_N_TILES) * phy_mac::output_bytes_per_tile,
                                 XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(4));
        slot.metadata_bo = xrt::bo(_device,
                                   static_cast<std::size_t>(_N_TILES) * sizeof(phy_mac::tile_metadata),
                                   XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(5));
        slot.input = slot.input_bo.map<std::int32_t*>();
        slot.output = slot.output_bo.map<std::uint8_t*>();
        slot.metadata = slot.metadata_bo.map<phy_mac::tile_metadata*>();
        std::memset(slot.output, 0, slot.output_bo.size());
        std::memset(slot.metadata, 0, slot.metadata_bo.size());
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.metadata_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, 3); // NPU run opcode
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        slot.run.set_arg(4, slot.output_bo);
        slot.run.set_arg(5, slot.metadata_bo);
    }
    // Also informs GNU Radio's upstream buffer allocation for this sink.
    // Without it a default buffer may hold only VECTOR_SIZE - 1 items,
    // making forecast's first complete NPU batch impossible to satisfy.
    set_output_multiple(_VECTOR_SIZE);
}

mlir_aie_80211_phy_thru_mac_decode_impl::~mlir_aie_80211_phy_thru_mac_decode_impl() = default;

void mlir_aie_80211_phy_thru_mac_decode_impl::set_nominal_frequency(double frequency)
{
    phy_frequency_rtp(frequency);
    _nominal_frequency.store(frequency);
}

double mlir_aie_80211_phy_thru_mac_decode_impl::nominal_frequency() const
{
    return _nominal_frequency.load();
}

void mlir_aie_80211_phy_thru_mac_decode_impl::forecast(
    int, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = _VECTOR_SIZE;
}

int mlir_aie_80211_phy_thru_mac_decode_impl::general_work(
    int, gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items, gr_vector_void_star&)
{
    const auto* in = static_cast<const std::int32_t*>(input_items[0]);
    int n_chunks = ninput_items[0] / _VECTOR_SIZE;
    if (n_chunks == 0)
        return 0;

    const auto input_start = nitems_read(0);
    std::vector<tag_t> frequency_tags;
    get_tags_in_range(frequency_tags, 0, input_start,
                      input_start + static_cast<std::uint64_t>(n_chunks) * _VECTOR_SIZE,
                      pmt::intern("rx_freq"));
    const auto batch = plan_phy_frequency_batch(
        frequency_tags, input_start, _VECTOR_SIZE, n_chunks);
    n_chunks = batch.n_chunks;
    if (batch.frequency)
        set_nominal_frequency(*batch.frequency);

    // All previous calls are retired: never patch instructions used by a run.
    const auto [center_mhz, reciprocal_q30] = phy_frequency_rtp(_nominal_frequency.load());
    if (_instr_v[_center_mhz_index] != static_cast<std::uint32_t>(center_mhz) ||
        _instr_v[_reciprocal_q30_index] != static_cast<std::uint32_t>(reciprocal_q30)) {
        _instr_v[_center_mhz_index] = static_cast<std::uint32_t>(center_mhz);
        _instr_v[_reciprocal_q30_index] = static_cast<std::uint32_t>(reciprocal_q30);
        _instructions[_center_mhz_index] = _instr_v[_center_mhz_index];
        _instructions[_reciprocal_q30_index] = _instr_v[_reciprocal_q30_index];
        _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::cout << "mlir_aie_80211_phy: nominal frequency changed to "
                  << _instr_v[_center_mhz_index] << " MHz" 
                  << " MHz" << std::endl;

    }

    const int depth = static_cast<int>(_slots.size());
    const auto launch = [this, in, depth](int chunk) {
        auto& slot = _slots[chunk % depth];
        std::memcpy(slot.input, in + static_cast<std::size_t>(chunk) * _VECTOR_SIZE,
                    static_cast<std::size_t>(_VECTOR_SIZE) * sizeof(std::int32_t));
        slot.input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        slot.run.start();
    };
    for (int i = 0; i < std::min(depth, n_chunks); ++i)
        launch(i);
    for (int i = 0; i < n_chunks; ++i) {
        auto& slot = _slots[i % depth];
        slot.run.wait();
        slot.output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        slot.metadata_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        for (int tile = 0; tile < _N_TILES; ++tile) {
            const auto& meta = slot.metadata[tile];
            const auto* output = slot.output +
                                 static_cast<std::size_t>(tile) * phy_mac::output_bytes_per_tile;
            for (int frame = 0; frame < meta.frame_count; ++frame)
                message_port_pub(pmt::intern("out"), phy_mac::make_pdu(output, meta.frames[frame]));
        }
        // Publish/copy before reusing this slot: DMA must not overwrite the
        // mapped output or metadata while the host is still reading it.
        if (i + depth < n_chunks)
            launch(i + depth);
    }
    if (batch.next_frequency)
        set_nominal_frequency(*batch.next_frequency);
    consume_each(n_chunks * _VECTOR_SIZE);
    return 0; // PDUs only; input progress is reported via consume_each.
}

} // namespace mlir_aie
} // namespace gr
