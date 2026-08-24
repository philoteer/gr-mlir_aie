/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_uint8_impl.h"
#include <gnuradio/io_signature.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_uint8::sptr mlir_aie_cpp_uint8::make(const char* path_xclbin,
                                                   const char* path_insts_bin,
                                                   int VECTOR_SIZE,
                                                   int num_slots)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_uint8_impl>(
        path_xclbin, path_insts_bin, VECTOR_SIZE, num_slots);
}


/*
 * The private constructor
 * Based on the official AMD mlir-aie passthru kernel test.cpp example
 */
mlir_aie_cpp_uint8_impl::mlir_aie_cpp_uint8_impl(const char* path_xclbin,
                                                  const char* path_insts_bin,
                                                  int VECTOR_SIZE,
                                                  int num_slots)
    : gr::block("mlir_aie_cpp_uint8",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
    if (num_slots < 1) {
        throw std::invalid_argument("num_slots must be at least 1");
    }

    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    _VECTOR_SIZE = VECTOR_SIZE;
    _kernel_name =  "MLIR_AIE"; //TODO FIX (make this a parameter?)
    set_output_multiple(_VECTOR_SIZE);
    _trace_size = 0;
    _opcode_run = 3;

    // Load instruction sequence
    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    std::cout << "Sequence instr count: " << _instr_v.size() << "\n";
    
    // Start the XRT context and load the kernel
    test_utils::init_xrt_load_kernel(_device, _kernel, 1,
                                   path_xclbin,
                                   _kernel_name);

    std::cout << "kernel load ok";
    // set up the buffer objects
    _bo_instr = xrt::bo(_device, _instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, _kernel.group_id(1));

    std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    bufInstr = _bo_instr.map<void *>();
    std::memcpy(bufInstr, _instr_v.data(), _instr_v.size() * sizeof(int));
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

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
        slot.input = slot.input_bo.map<input_type*>();
        slot.output = slot.output_bo.map<output_type*>();
        std::memset(slot.output,
                    42,
                    _VECTOR_SIZE * sizeof(output_type) + _trace_size);
        slot.output_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        slot.run = xrt::run(_kernel);
        slot.run.set_arg(0, _opcode_run);
        slot.run.set_arg(1, _bo_instr);
        slot.run.set_arg(2, _instr_v.size());
        slot.run.set_arg(3, slot.input_bo);
        slot.run.set_arg(4, slot.output_bo);
    }
}

/*
 * Our virtual destructor.
 */
mlir_aie_cpp_uint8_impl::~mlir_aie_cpp_uint8_impl() {}

void mlir_aie_cpp_uint8_impl::forecast(int noutput_items,
                                       gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = std::max(_VECTOR_SIZE, noutput_items);
}

int mlir_aie_cpp_uint8_impl::general_work(int noutput_items,
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

    const int depth = static_cast<int>(_slots.size());
    const auto launch = [this, in](int chunk_idx) {
        auto& slot = _slots[chunk_idx % _slots.size()];
        const input_type* in_ptr = in + (chunk_idx * _VECTOR_SIZE);
        std::memcpy(slot.input, in_ptr, _VECTOR_SIZE * sizeof(input_type));
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
        output_type* out_ptr = out + (i * _VECTOR_SIZE);
        std::memcpy(out_ptr, slot.output, _VECTOR_SIZE * sizeof(output_type));

        if (i + depth < n_chunks) {
            launch(i + depth);
        }
    }

    // ## Back to GNURadio
    int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);
    
    return processed_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
