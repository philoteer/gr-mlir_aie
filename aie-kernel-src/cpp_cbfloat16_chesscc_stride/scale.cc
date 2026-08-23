//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <aie_api/aie_types.hpp>

extern "C" {

void vector_scalar_mul_vector(cbfloat16 *a, cbfloat16 *c, int32_t N) {
  
  constexpr int vec_factor = 16; 
  cbfloat16 *__restrict pA1 = a;
  cbfloat16 *__restrict pC1 = c;
  const int F = N / vec_factor;
  
  cbfloat16 fac;
  fac.real = (bfloat16) 2.0f;
  fac.imag = (bfloat16) 1.0f;

  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<cbfloat16, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;    
    aie::accum<caccfloat, vec_factor> cout = aie::mul(A0, fac);    
    aie::store_v(pC1, cout.template to_vector<cbfloat16>(0));
    pC1 += vec_factor;
  }
}

} // extern "C"
