#pragma once

#include <immintrin.h>

enum {
  isa_undef = 0,
  sse42,
  avx,
  avx2,
  avx512_common,
  avx512_core,
  avx512_core_vnni
};

static inline bool cpu_has(int isa)
{
#if __ICC_COMPILER
  switch (isa) {
  case sse42:
    return _may_i_use_cpu_feature(_FEATURE_SSE4_2);
  case avx:
     return _may_i_use_cpu_feature(_FEATURE_AVX);
  case avx2:
    return _may_i_use_cpu_feature(_FEATURE_AVX2);
  case avx512_common:
    return _may_i_use_cpu_feature(_FEATURE_AVX512F);
  case avx512_core:
    return _may_i_use_cpu_feature(_FEATURE_AVX512F | _FEATURE_AVX512BW);
#if defined(WITH_VNNI)
  case avx512_core_vnni:
    return _may_i_use_cpu_feature(
        _FEATURE_AVX512F | _FEATURE_AVX512BW | _FEATURE_AVX512_VNNI);
#endif
  default:
    return false;
  }
  return false;
#else
  // TODO
  return true;
#endif
}

// CPU V length by byte
static inline int cpu_vector_length() {
  if (cpu_has(avx512_common))
    return 64; // zmm
  else if (cpu_has(avx))
    return 32; // ymm
  else if (cpu_has(sse42))
    return 16; // xmm
  else
    return 8;
}
