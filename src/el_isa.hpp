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
  switch (isa) {
  case sse42:
    // return _may_i_use_cpu_feature(_FEATURE_SSE4_2);
    return __builtin_cpu_supports("sse4.2");
  case avx:
    // return _may_i_use_cpu_feature(_FEATURE_AVX);
    return __builtin_cpu_supports("avx");
  case avx2:
    // return _may_i_use_cpu_feature(_FEATURE_AVX2);
    return __builtin_cpu_supports("avx2");
  case avx512_common:
    // return _may_i_use_cpu_feature(_FEATURE_AVX512F);
    return __builtin_cpu_supports("avx512f");
  case avx512_core:
    // return _may_i_use_cpu_feature(_FEATURE_AVX512F | _FEATURE_AVX512BW);
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw");
#if defined(WITH_VNNI)
  case avx512_core_vnni:
    // return _may_i_use_cpu_feature(
    //    _FEATURE_AVX512F | _FEATURE_AVX512BW | _FEATURE_AVX512_VNNI);
    return __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vnni");
#endif
  default:
    return false;
  }
  return false;
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
