/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_tagged_int32_to_int32_impl.h"
#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>

#include <algorithm>
#include <cstring>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_tagged_int32_to_int32::sptr
mlir_aie_cpp_tagged_int32_to_int32::make(const char* path_xclbin,
                                         const char* path_insts_bin,
                                         const char* kernel_name,
                                         int VECTOR_SIZE)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_tagged_int32_to_int32_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE);
}


/*
 * The private constructor
 */
mlir_aie_cpp_tagged_int32_to_int32_impl::mlir_aie_cpp_tagged_int32_to_int32_impl(const char* path_xclbin,
                                                                                const char* path_insts_bin,
                                                                                const char* kernel_name,
                                                                                int VECTOR_SIZE)
    : gr::block("mlir_aie_cpp_tagged_int32_to_int32",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(tagged_input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(tagged_output_type)))
{
    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    _VECTOR_SIZE = VECTOR_SIZE;
    _TILE_SIZE = _VECTOR_SIZE / _N_TILES;   //TODO FIX (should have been args, I guess)
    _kernel_name = kernel_name;
    _trace_size = 0;
    _opcode_run = 3;

    set_tag_propagation_policy(TPP_DONT);

    // Load instruction sequence
    _instr_v = test_utils::load_instr_binary(path_insts_bin);
    std::cout << "Sequence instr count: " << _instr_v.size() << "\n";

    // Start the XRT context and load the kernel
    test_utils::init_xrt_load_kernel(_device, _kernel, 1, path_xclbin, _kernel_name);

    std::cout << "kernel load ok";
    // set up the buffer objects
    _bo_instr = xrt::bo(_device,
                        _instr_v.size() * sizeof(int),
                        XCL_BO_FLAGS_CACHEABLE,
                        _kernel.group_id(1));
    _bo_inA = xrt::bo(_device,
                      _VECTOR_SIZE * sizeof(tagged_input_type),
                      XRT_BO_FLAGS_HOST_ONLY,
                      _kernel.group_id(3));
    _bo_out = xrt::bo(_device,
                      _VECTOR_SIZE * sizeof(tagged_output_type) + _trace_size,
                      XRT_BO_FLAGS_HOST_ONLY,
                      _kernel.group_id(3));
    _bo_out_meta = xrt::bo(_device,
                           _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t),
                           XRT_BO_FLAGS_HOST_ONLY,
                           _kernel.group_id(3));

    std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    bufInstr = _bo_instr.map<void*>();
    memcpy(bufInstr, _instr_v.data(), _instr_v.size() * sizeof(int));

    // Initialize buffers
    _bufInA = _bo_inA.map<tagged_input_type*>();
    _bufOut = _bo_out.map<tagged_output_type*>();
    _bufOutMeta = _bo_out_meta.map<std::int32_t*>();
    memset(_bufOut, 42, _VECTOR_SIZE * sizeof(tagged_output_type) + _trace_size);
    memset(_bufOutMeta, 0, _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));

    _bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_out_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    _run = xrt::run(_kernel);

    _run.set_arg(0, _opcode_run);
    _run.set_arg(1, _bo_instr);
    _run.set_arg(2, _instr_v.size());
    _run.set_arg(3, _bo_inA);
    _run.set_arg(4, _bo_out);
    _run.set_arg(5, _bo_out_meta);
}

/*
 * Our virtual destructor.
 */
mlir_aie_cpp_tagged_int32_to_int32_impl::~mlir_aie_cpp_tagged_int32_to_int32_impl() {}

void mlir_aie_cpp_tagged_int32_to_int32_impl::forecast(
    int noutput_items, gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int mlir_aie_cpp_tagged_int32_to_int32_impl::general_work(
    int noutput_items,
    gr_vector_int& ninput_items,
    gr_vector_const_void_star& input_items,
    gr_vector_void_star& output_items)
{
    auto in = static_cast<const tagged_input_type*>(input_items[0]);
    auto out = static_cast<tagged_output_type*>(output_items[0]);

    int n_chunks = std::min(ninput_items[0], noutput_items) / _VECTOR_SIZE;

    if (n_chunks == 0) {
        return 0;
    }

    const auto tag_key = pmt::intern("wifi_start");
    const auto tag_srcid = pmt::intern("sync_short");
    const uint64_t output_abs_start = nitems_written(0);
    int total_produced = 0;

    for (int i = 0; i < n_chunks; i++) {
        const tagged_input_type* in_ptr = in + (i * _VECTOR_SIZE);

        memcpy(_bufInA, in_ptr, _VECTOR_SIZE * sizeof(tagged_input_type));
        //memset(_bufOutMeta, 0, _N_TILES * _METADATA_WORDS_PER_TILE * sizeof(std::int32_t));

        _bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        //_bo_out_meta.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        _run.start();
        _run.wait();

        _bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        _bo_out_meta.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        //TODO FIX (AI-generated, unverified)
        //for each tile
        for (int tile_idx = 0; tile_idx < _N_TILES; tile_idx++) {
            
            //n-th tile's metadata
            const std::int32_t* tile_meta =
                _bufOutMeta + (tile_idx * _METADATA_WORDS_PER_TILE);
            
            //main data, tag len
            int tile_len = tile_meta[0];
            int tag_count = tile_meta[1];
            int tile_start = tile_idx * _TILE_SIZE; //tile start position in bufout

            // should not happen IF you wrote the kernel correctly:
            if (tile_len < 0) {
                tile_len = 0;
            } else if (tile_len > _TILE_SIZE) {
                tile_len = _TILE_SIZE;
            }

            //copy the main stream
            memcpy(out + total_produced,
                   _bufOut + tile_start,
                   tile_len * sizeof(tagged_output_type));

            //generate tag messages
            const uint64_t tile_abs_start = output_abs_start + total_produced;
            for (int tag_idx = 0; tag_idx < tag_count; tag_idx++) {
                const int tag_offset = tile_meta[2 + 2 * tag_idx];
                const double tag_value =
                    static_cast<double>(tile_meta[3 + 2 * tag_idx]) / (std::int64_t{1} << 29);

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
    }

    int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);

    return total_produced;
}

} /* namespace mlir_aie */
} /* namespace gr */
