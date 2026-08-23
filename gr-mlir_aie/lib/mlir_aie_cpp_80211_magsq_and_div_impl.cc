/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_80211_magsq_and_div_impl.h"
#include <gnuradio/io_signature.h>

#include <algorithm>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_80211_magsq_and_div::sptr
mlir_aie_cpp_80211_magsq_and_div::make(const char* path_xclbin,
                                       const char* path_insts_bin,
                                       const char* kernel_name,
                                       int VECTOR_SIZE)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_80211_magsq_and_div_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE);
}


/*
 * The private constructor
 */
mlir_aie_cpp_80211_magsq_and_div_impl::mlir_aie_cpp_80211_magsq_and_div_impl(
    const char* path_xclbin,
    const char* path_insts_bin,
    const char* kernel_name,
    int VECTOR_SIZE)
    : gr::block(
          "mlir_aie_cpp_80211_magsq_and_div",
          gr::io_signature::make2(2,
                                  2,
                                  sizeof(magsq_complex_input_type),
                                  sizeof(magsq_mag_input_type)),
          gr::io_signature::make(1, 1, sizeof(magsq_output_type)))
{
    if (VECTOR_SIZE <= 0 || VECTOR_SIZE % 16 != 0) {
        throw std::invalid_argument(
            "mlir_aie_cpp_80211_magsq_and_div VECTOR_SIZE must be a positive multiple of 16");
    }

    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    _VECTOR_SIZE = VECTOR_SIZE;
    _kernel_name = kernel_name;
    _trace_size = 0;
    _opcode_run = 3;

    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    std::cout << "Sequence instr count: " << _instr_v.size() << "\n";

    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, _kernel_name);

    std::cout << "kernel load ok";
    _bo_instr = xrt::bo(_device,
                        _instr_v.size() * sizeof(std::uint32_t),
                        XCL_BO_FLAGS_CACHEABLE,
                        _kernel.group_id(1));
    _bo_ac_in = xrt::bo(_device,
                        _VECTOR_SIZE * sizeof(magsq_complex_input_type),
                        XRT_BO_FLAGS_HOST_ONLY,
                        _kernel.group_id(3));
    _bo_mag_in = xrt::bo(_device,
                         _VECTOR_SIZE * sizeof(magsq_mag_input_type),
                         XRT_BO_FLAGS_HOST_ONLY,
                         _kernel.group_id(3));
    _bo_out = xrt::bo(_device,
                      _VECTOR_SIZE * sizeof(magsq_output_type) + _trace_size,
                      XRT_BO_FLAGS_HOST_ONLY,
                      _kernel.group_id(3));

    std::cout << "Writing data into buffer objects.\n";

    _bufInstr = _bo_instr.map<void*>();
    memcpy(_bufInstr, _instr_v.data(), _instr_v.size() * sizeof(std::uint32_t));

    _bufAcIn = _bo_ac_in.map<magsq_complex_input_type*>();
    _bufMagIn = _bo_mag_in.map<magsq_mag_input_type*>();
    _bufOut = _bo_out.map<magsq_output_type*>();
    memset(_bufOut, 42, _VECTOR_SIZE * sizeof(magsq_output_type) + _trace_size);

    _bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    _run = xrt::run(_kernel);

    _run.set_arg(0, _opcode_run);
    _run.set_arg(1, _bo_instr);
    _run.set_arg(2, _instr_v.size());
    _run.set_arg(3, _bo_ac_in);
    _run.set_arg(4, _bo_mag_in);
    _run.set_arg(5, _bo_out);
}

/*
 * Our virtual destructor.
 */
mlir_aie_cpp_80211_magsq_and_div_impl::~mlir_aie_cpp_80211_magsq_and_div_impl() {}

void mlir_aie_cpp_80211_magsq_and_div_impl::forecast(int noutput_items,
                                                      gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
    ninput_items_required[1] = noutput_items;
}

int mlir_aie_cpp_80211_magsq_and_div_impl::general_work(
    int noutput_items,
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    auto ac_in = static_cast<const magsq_complex_input_type*>(input_items[0]);
    auto mag_in = static_cast<const magsq_mag_input_type*>(input_items[1]);
    auto out = static_cast<magsq_output_type*>(output_items[0]);

    const int available_items = std::min(std::min(ninput_items[0], ninput_items[1]),
                                         noutput_items);
    const int n_chunks = available_items / _VECTOR_SIZE;
    if (n_chunks == 0) {
        return 0;
    }

    for (int i = 0; i < n_chunks; i++) {
        const magsq_complex_input_type* ac_ptr = ac_in + (i * _VECTOR_SIZE);
        const magsq_mag_input_type* mag_ptr = mag_in + (i * _VECTOR_SIZE);
        magsq_output_type* out_ptr = out + (i * _VECTOR_SIZE);

        memcpy(_bufAcIn, ac_ptr, _VECTOR_SIZE * sizeof(magsq_complex_input_type));
        memcpy(_bufMagIn, mag_ptr, _VECTOR_SIZE * sizeof(magsq_mag_input_type));

        _bo_ac_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        _bo_mag_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        _run.start();
        _run.wait();

        _bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(out_ptr, _bufOut, _VECTOR_SIZE * sizeof(magsq_output_type));
    }

    const int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);

    return processed_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
