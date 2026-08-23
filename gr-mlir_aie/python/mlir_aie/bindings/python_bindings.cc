/*
 * Copyright 2020 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

#include <pybind11/pybind11.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;

// Headers for binding functions
/**************************************/
// The following comment block is used for
// gr_modtool to insert function prototypes
// Please do not delete
/**************************************/
// BINDING_FUNCTION_PROTOTYPES(
    void bind_mlir_aie_cpp_uint8(py::module& m);
    void bind_mlir_aie_cpp_int32(py::module& m);
    void bind_mlir_aie_cpp_bfloat16(py::module& m);
    void bind_mlir_aie_cpp_int16(py::module& m);
    void bind_float32_to_bfloat16(py::module& m);
    void bind_bfloat16_to_float32(py::module& m);
    void bind_complex64_to_cbfloat(py::module& m);
    void bind_cbfloat_to_complex64(py::module& m);
    void bind_mlire_aie_cpp_int32_to_int16(py::module& m);
    void bind_mlir_aie_cpp_int32_source_1arg(py::module& m);
    void bind_mlir_aie_cpp_tagged_int32_to_int32(py::module& m);
    void bind_mlir_aie_cpp_80211_magsq_and_div(py::module& m);
    void bind_complex64_to_cint16(py::module& m);
    void bind_float32_to_Q15(py::module& m);
    void bind_cint16_to_complex64(py::module& m);
    void bind_Q15_to_float32(py::module& m);
    void bind_cint32_to_complex64(py::module& m);
    void bind_int32_Q_to_float32(py::module& m);
    void bind_float32_to_int32_Q(py::module& m);
    void bind_complex64_to_cint32(py::module& m);
    void bind_mlir_aie_cpp_int32_to_int64(py::module& m);
    void bind_mlir_aie_cpp_sync_long_test(py::module& m);
    void bind_mlir_aie_cpp_tagged_int32_to_int64(py::module& m);
    void bind_mlir_aie_cpp_fft_test(py::module& m);
    void bind_mlir_aie_80211_phy(py::module& m);
    void bind_mlir_aie_cpp_equalizer_test(py::module& m);
// ) END BINDING_FUNCTION_PROTOTYPES


// We need this hack because import_array() returns NULL
// for newer Python versions.
// This function is also necessary because it ensures access to the C API
// and removes a warning.
void* init_numpy()
{
    import_array();
    return NULL;
}

PYBIND11_MODULE(mlir_aie_python, m)
{
    // Initialize the numpy C API
    // (otherwise we will see segmentation faults)
    init_numpy();

    // Allow access to base block methods
    py::module::import("gnuradio.gr");

    /**************************************/
    // The following comment block is used for
    // gr_modtool to insert binding function calls
    // Please do not delete
    /**************************************/
    // BINDING_FUNCTION_CALLS(
    bind_mlir_aie_cpp_uint8(m);
    bind_mlir_aie_cpp_int32(m);
    bind_mlir_aie_cpp_bfloat16(m);
    bind_mlir_aie_cpp_int16(m);
    bind_float32_to_bfloat16(m);
    bind_bfloat16_to_float32(m);
    bind_complex64_to_cbfloat(m);
    bind_cbfloat_to_complex64(m);
    bind_mlire_aie_cpp_int32_to_int16(m);
    bind_mlir_aie_cpp_int32_source_1arg(m);
    bind_mlir_aie_cpp_tagged_int32_to_int32(m);
    bind_mlir_aie_cpp_80211_magsq_and_div(m);
    bind_complex64_to_cint16(m);
    bind_float32_to_Q15(m);
    bind_cint16_to_complex64(m);
    bind_Q15_to_float32(m);
    bind_cint32_to_complex64(m);
    bind_int32_Q_to_float32(m);
    bind_float32_to_int32_Q(m);
    bind_complex64_to_cint32(m);
    bind_mlir_aie_cpp_int32_to_int64(m);
    bind_mlir_aie_cpp_sync_long_test(m);
    bind_mlir_aie_cpp_tagged_int32_to_int64(m);
    bind_mlir_aie_cpp_fft_test(m);
    bind_mlir_aie_80211_phy(m);
    bind_mlir_aie_cpp_equalizer_test(m);
    // ) END BINDING_FUNCTION_CALLS
}