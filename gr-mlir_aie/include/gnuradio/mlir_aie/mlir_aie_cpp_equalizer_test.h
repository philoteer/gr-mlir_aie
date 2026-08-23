/* -*- c++ -*- */
/*
 * Copyright 2026 gr-mlir_aie author.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#ifndef INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_H
#define INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_H

#include <gnuradio/block.h>
#include <gnuradio/mlir_aie/api.h>

namespace gr {
namespace mlir_aie {

/*!
 * \brief Run the MLIR-AIE frame equalizer kernel.
 * \ingroup mlir_aie
 *
 * Complex float input samples are converted to packed Q16.15 cint32 values for
 * the kernel. Input wifi_start tags and output frame metadata are transferred
 * through the kernel's per-tile metadata buffers.
 */
class MLIR_AIE_API mlir_aie_cpp_equalizer_test : virtual public gr::block
{
public:
    typedef std::shared_ptr<mlir_aie_cpp_equalizer_test> sptr;

    /*!
     * \brief Return a shared_ptr to a new instance of
     * mlir_aie::mlir_aie_cpp_equalizer_test.
     *
     * To avoid accidental use of raw pointers, mlir_aie::mlir_aie_cpp_equalizer_test's
     * constructor is in a private implementation
     * class. mlir_aie::mlir_aie_cpp_equalizer_test::make is the public interface for
     * creating new instances.
     */
    static sptr make(const char* path_xclbin,
                     const char* path_insts_bin,
                     const char* kernel_name,
                     int VECTOR_SIZE,
                     double nominal_frequency);

    virtual void set_nominal_frequency(double nominal_frequency) = 0;
    virtual double nominal_frequency() const = 0;
};

} // namespace mlir_aie
} // namespace gr

#endif /* INCLUDED_MLIR_AIE_MLIR_AIE_CPP_EQUALIZER_TEST_H */
