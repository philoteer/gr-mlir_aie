/* SPDX-License-Identifier: GPL-3.0-or-later */
#include "mlir_aie_cpp_decode_mac_test_impl.h"
#include "runtime_lib/test_lib/test_utils.h"
#include <gnuradio/io_signature.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gr {
namespace mlir_aie {
namespace {
// nearbyint matches Python/NumPy's round-to-even quantization.
std::int32_t q32(double value)
{
    if (std::isnan(value))
        throw std::runtime_error("NaN in MAC PHY metadata");
    return static_cast<std::int32_t>(std::clamp(
        std::nearbyint(value),
        double(std::numeric_limits<std::int32_t>::min()),
        double(std::numeric_limits<std::int32_t>::max())));
}
}

mlir_aie_cpp_decode_mac_test::sptr mlir_aie_cpp_decode_mac_test::make(
    const char* path_xclbin, const char* path_insts_bin, const char* kernel_name)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_decode_mac_test_impl>(
        path_xclbin, path_insts_bin, kernel_name);
}

mlir_aie_cpp_decode_mac_test_impl::mlir_aie_cpp_decode_mac_test_impl(
    const char* path_xclbin, const char* path_insts_bin, const char* kernel_name)
    : gr::block("mlir_aie_cpp_decode_mac_test",
                gr::io_signature::make(1, 1, 48 * sizeof(std::uint8_t)),
                gr::io_signature::make(0, 0, 0))
{
    message_port_register_out(pmt::intern("out"));
    set_tag_propagation_policy(TPP_DONT);
    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, kernel_name);
    _bo_instr = xrt::bo(_device, _instr_v.size() * sizeof(std::uint32_t),
                        XCL_BO_FLAGS_CACHEABLE, _kernel.group_id(1));
    std::memcpy(_bo_instr.map<std::uint32_t*>(), _instr_v.data(), _bo_instr.size());
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_input = xrt::bo(_device, 1024, XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(3));
    _bo_input_meta = xrt::bo(_device, sizeof(decode_mac_input_metadata),
                            XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(4));
    _bo_output = xrt::bo(_device, 8192, XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(5));
    _bo_output_meta = xrt::bo(_device, sizeof(phy_mac::tile_metadata),
                             XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(6));
    _input = _bo_input.map<std::uint8_t*>();
    _input_meta = _bo_input_meta.map<decode_mac_input_metadata*>();
    _output = _bo_output.map<std::uint8_t*>();
    _output_meta = _bo_output_meta.map<phy_mac::tile_metadata*>();
    std::memset(_output, 0, _bo_output.size());
    std::memset(_output_meta, 0, _bo_output_meta.size());
    _bo_output.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_output_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _run = xrt::run(_kernel);
    _run.set_arg(0, 3);
    _run.set_arg(1, _bo_instr);
    _run.set_arg(2, _instr_v.size());
    _run.set_arg(3, _bo_input);
    _run.set_arg(4, _bo_input_meta);
    _run.set_arg(5, _bo_output);
    _run.set_arg(6, _bo_output_meta);
}

mlir_aie_cpp_decode_mac_test_impl::~mlir_aie_cpp_decode_mac_test_impl() = default;

void mlir_aie_cpp_decode_mac_test_impl::run_tile()
{
    // Stateful kernel: finish each invocation before submitting the next tile.
    _bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_input_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _run.start();
    _run.wait();
    _bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    _bo_output_meta.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
}

bool mlir_aie_cpp_decode_mac_test_impl::start()
{
    _frame_id = 0;
    _frame_metadata.clear();
    std::memset(_input, 0, _bo_input.size());
    *_input_meta = {};
    // Explicitly abandon device state, including on flowgraph restarts.
    _input_meta->output_length = -1;
    run_tile();
    return true;
}

void mlir_aie_cpp_decode_mac_test_impl::forecast(int, gr_vector_int& required)
{
    // Do not require 16 items: a final short tile must also be decoded.
    required[0] = 1;
}

void mlir_aie_cpp_decode_mac_test_impl::pack_tags(int count)
{
    *_input_meta = {};
    _input_meta->output_length = count * 48;
    const auto first = nitems_read(0);
    for (int i = 0; i < count; ++i) {
        std::vector<tag_t> tags;
        get_tags_in_range(tags, 0, first + i, first + i + 1);
        auto original = pmt::make_dict();
        bool has_symbol = false;
        for (const auto& tag : tags) {
            if (pmt::eq(tag.key, pmt::intern("iqdata")))
                continue;
            original = pmt::dict_add(original, tag.key, tag.value);
            has_symbol |= pmt::is_symbol(tag.key);
        }
        if (!has_symbol)
            continue;
        auto& record = _input_meta->tags[_input_meta->tag_count++];
        record.offset = i * 48;
        const auto has = [&original](const char* key) {
            return pmt::dict_has_key(original, pmt::intern(key));
        };
        const auto get = [&original](const char* key) {
            return pmt::dict_ref(original, pmt::intern(key), pmt::PMT_NIL);
        };
        if (has("frame bytes")) {
            _frame_id = _frame_id % 0x7fffffffu + 1;
            record.frame_start = (_frame_id << 1) | 1;
            record.frame_bytes = pmt::to_uint64(get("frame bytes"));
            _frame_metadata[_frame_id] = original;
        }
        if (has("encoding"))
            record.encoding = pmt::to_uint64(get("encoding"));
        if (has("snr")) {
            const double ratio = 32.0 * std::pow(10.0, pmt::to_double(get("snr")) / 10.0);
            record.snr_linear = std::isnan(ratio) || ratio <= 0 ? 0 :
                ratio >= double(std::numeric_limits<std::uint64_t>::max()) ?
                std::numeric_limits<std::uint64_t>::max() :
                static_cast<std::uint64_t>(std::nearbyint(ratio));
        }
        if (has("nominal frequency"))
            record.center_frequency_mhz =
                static_cast<std::uint32_t>(std::nearbyint(pmt::to_double(get("nominal frequency")) / 1e6));
        constexpr double q29 = 536870912.0;
        constexpr double pi = 3.14159265358979323846;
        if (has("frequency offset"))
            record.frequency_offset = q32(pmt::to_double(get("frequency offset")) * (q29 * 2 * pi / 20e6));
        if (has("beta"))
            record.beta = q32(pmt::to_double(get("beta")) * q29);
        if (has("csi")) {
            const auto csi = pmt::c32vector_elements(get("csi"));
            for (std::size_t k = 0; k < std::min<std::size_t>(64, csi.size()); ++k) {
                record.csi[k].real = q32(double(csi[k].real()) * 32768.0);
                record.csi[k].imag = q32(double(csi[k].imag()) * 32768.0);
            }
        }
    }
}

void mlir_aie_cpp_decode_mac_test_impl::publish_frames()
{
    const auto& meta = *_output_meta;
    if (meta.output_length < 0 || meta.output_length > 8192 ||
        meta.frame_count < 0 || meta.frame_count > 16)
        throw std::runtime_error("invalid NPU MAC output header");
    for (int i = 0; i < meta.frame_count; ++i) {
        const auto& frame = meta.frames[i];
        if (frame.offset < 0 || frame.length < 0 ||
            frame.offset > meta.output_length || frame.length > meta.output_length - frame.offset)
            throw std::runtime_error("invalid NPU MAC frame bounds");
        const auto it = _frame_metadata.find(frame.phy.frame_start >> 1);
        if (it == _frame_metadata.end())
            throw std::runtime_error("unknown NPU MAC frame cookie");
        const auto original = pmt::dict_add(it->second, pmt::intern("dlt"), pmt::from_long(105));
        // Copy bytes before reusing mapped output, preserving Python's u8vector PDU.
        message_port_pub(pmt::intern("out"), pmt::cons(original,
            pmt::init_u8vector(frame.length, _output + frame.offset)));
        _frame_metadata.erase(it);
    }
    // Only the newest start may still be incomplete; discard aborted/bad-FCS frames.
    for (auto it = _frame_metadata.begin(); it != _frame_metadata.end();) {
        if (it->first != _frame_id)
            it = _frame_metadata.erase(it);
        else
            ++it;
    }
}

int mlir_aie_cpp_decode_mac_test_impl::general_work(
    int, gr_vector_int& ninput_items, gr_vector_const_void_star& input_items,
    gr_vector_void_star&)
{
    const int count = std::min(ninput_items[0], 16);
    if (!count)
        return 0;
    std::memset(_input, 0, _bo_input.size());
    std::memcpy(_input, input_items[0], count * 48);
    pack_tags(count);
    run_tile();
    publish_frames();
    consume_each(count);
    return 0; // PDUs only.
}
} // namespace mlir_aie
} // namespace gr
