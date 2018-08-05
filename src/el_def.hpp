#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef __EL_DEF_HPP__
#define __EL_DEF_HPP__

namespace euler {

enum {
  ISA_GENERIC = 0,
  ISA_SKX_AVX512 = 512,
};

#ifdef __INTEL_COMPILER
#define ENABLE_AVX512F() _allow_cpu_features(_FEATURE_AVX512F)
#define pragma_opt_core_avx512                                                 \
  _Pragma("optimization_parameter target_arch=CORE-AVX512")
#define pragma_unroll _Pragma("unroll")
#else
#define ENABLE_AVX512F() // -mavx512f
#define pragma_opt_core_avx512
#define pragma_unroll
#endif

inline void el_error(const char *msg)
{
  printf("Euler:Error: %s\n", msg);
  abort();
}

inline void el_warn(const char *msg)
{
  printf("Euler:Warning: %s\n", msg);
}

} // namespace euler

#endif // __EL_DEF_HPP__
