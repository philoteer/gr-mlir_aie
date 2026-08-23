// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"

namespace {
constexpr int32_t kProfileMagic = 0x50524631; // "PRF1"
constexpr int32_t kProfileVersion = 1;
constexpr int32_t kProfileWords = 16;
}

extern "C" void profile_mul2(const int32_t *__restrict input,
                             int32_t *__restrict output,
                             const int32_t payload_words) {
  const int32_t vector_words = payload_words / 2;
  const uint64_t total_start = aie::tile::current().cycles();

  const uint64_t vector_start = aie::tile::current().cycles();
  constexpr int32_t vector_size = 16;
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = 0; i < vector_words; i += vector_size) {
    const aie::vector<int32_t, vector_size> values =
        aie::load_v<vector_size>(input + i);
    const aie::accum<acc64, vector_size> doubled = aie::mul(values, 2);
    aie::store_v(output + i, doubled.to_vector<int32_t>(0));
  }
  const uint64_t vector_cycles =
      aie::tile::current().cycles() - vector_start;

  const uint64_t scalar_start = aie::tile::current().cycles();
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = vector_words; i < payload_words; ++i)
    output[i] = input[i] * 2;
  const uint64_t scalar_cycles =
      aie::tile::current().cycles() - scalar_start;

  const uint64_t checksum_start = aie::tile::current().cycles();
  uint32_t checksum = 0;
  AIE_PREPARE_FOR_PIPELINING
  for (int32_t i = 0; i < payload_words; ++i)
    checksum += static_cast<uint32_t>(output[i]);
  const uint64_t checksum_cycles =
      aie::tile::current().cycles() - checksum_start;

  const uint64_t total_cycles =
      aie::tile::current().cycles() - total_start;
  int32_t *profile = output + payload_words;
  profile[0] = kProfileMagic;
  profile[1] = kProfileVersion;
  profile[2] = static_cast<int32_t>(total_cycles);
  profile[3] = static_cast<int32_t>(vector_cycles);
  profile[4] = static_cast<int32_t>(scalar_cycles);
  profile[5] = static_cast<int32_t>(checksum_cycles);
  profile[6] = payload_words;
  profile[7] = static_cast<int32_t>(checksum);
  profile[8] = 1;
  for (int32_t i = 9; i < kProfileWords; ++i)
    profile[i] = 0;
}
