/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "mlir_aie_cpp_int32_impl.h"
#include <gnuradio/io_signature.h>

namespace gr {
namespace mlir_aie {

mlir_aie_cpp_int32::sptr mlir_aie_cpp_int32::make(const char* path_xclbin,
                                                  const char* path_insts_bin,
                                                  const char* kernel_name,
                                                  int VECTOR_SIZE)
{
    return gnuradio::make_block_sptr<mlir_aie_cpp_int32_impl>(
        path_xclbin, path_insts_bin, kernel_name, VECTOR_SIZE);
}


/*
 * The private constructor
 */
mlir_aie_cpp_int32_impl::mlir_aie_cpp_int32_impl(const char* path_xclbin,
                                                 const char* path_insts_bin,
                                                 const char* kernel_name,
                                                 int VECTOR_SIZE)
    : gr::block("mlir_aie_cpp_int32",
                gr::io_signature::make(
                    1 /* min inputs */, 1 /* max inputs */, sizeof(input_type)),
                gr::io_signature::make(
                    1 /* min outputs */, 1 /*max outputs */, sizeof(output_type)))
{
    _path_xclbin = path_xclbin;
    _path_insts_bin = path_insts_bin;
    _VECTOR_SIZE = VECTOR_SIZE;
    _kernel_name =  kernel_name; 
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
    _bo_inA = xrt::bo(_device,  _VECTOR_SIZE * sizeof(input_type),
                        XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(3));
    _bo_out =
      xrt::bo(_device,  _VECTOR_SIZE * sizeof(output_type) + _trace_size,
              XRT_BO_FLAGS_HOST_ONLY, _kernel.group_id(3));
              
    std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    bufInstr = _bo_instr.map<void *>();
    memcpy(bufInstr, _instr_v.data(), _instr_v.size() * sizeof(int));
    
    // Initialize buffers
    _bufInA = _bo_inA.map<input_type *>();     
    _bufOut = _bo_out.map<output_type *>();
     memset(_bufOut, 42, _VECTOR_SIZE * sizeof(output_type) + _trace_size); //TODO does this matter? maybe I can remove this line.
     
    _bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);   
    _bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    _run = xrt::run(_kernel);
    
    _run.set_arg(0, _opcode_run);
    _run.set_arg(1, _bo_instr);
    _run.set_arg(2, _instr_v.size());
    _run.set_arg(3, _bo_inA);
    _run.set_arg(4, _bo_out);
    
}

/*
 * Our virtual destructor.
 */
mlir_aie_cpp_int32_impl::~mlir_aie_cpp_int32_impl() {}

void mlir_aie_cpp_int32_impl::forecast(int noutput_items,
                                       gr_vector_int& ninput_items_required)
{
    ninput_items_required[0] = noutput_items;
}

int mlir_aie_cpp_int32_impl::general_work(int noutput_items,
                                          gr_vector_int& ninput_items,
                                          gr_vector_const_void_star& input_items,
                                          gr_vector_void_star& output_items)
{
    auto in = static_cast<const input_type*>(input_items[0]);
    auto out = static_cast<output_type*>(output_items[0]);

    if(noutput_items < _VECTOR_SIZE)
    {
        return 0;
    }
    
    int n_chunks = noutput_items / _VECTOR_SIZE;
    for (int i = 0; i < n_chunks; i++) {
        
        const input_type* in_ptr = in + (i * _VECTOR_SIZE);
        output_type* out_ptr = out + (i * _VECTOR_SIZE);

        memcpy(_bufInA, in_ptr, _VECTOR_SIZE * sizeof(input_type));

        _bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        
        _run.start();
        _run.wait();
        
        _bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(out_ptr, _bufOut, _VECTOR_SIZE * sizeof(output_type));
    }

    // ## Back to GNURadio
    int processed_items = n_chunks * _VECTOR_SIZE;
    consume_each(processed_items);
    
    return processed_items;
}

} /* namespace mlir_aie */
} /* namespace gr */
