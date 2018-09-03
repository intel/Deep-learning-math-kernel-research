#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef __EL_DEF_HPP__
#define __EL_DEF_HPP__

namespace euler {

enum {
  ISA_GENERIC = 0,
  ISA_AVX2 = 256,
  ISA_COSIM_AVX2,
  ISA_SKX_AVX512 = 512,
  ISA_COSIM_AVX512,
};


#define __GCC_COMPILER (__GNUC__ && !__INTEL_COMPILER && !__clang__)
#define __ICC_COMPILER __INTEL_COMPILER
#define __CLANG_COMPILER __clang__

// XXX: do we need this?
#if __ICC_COMPILER
#define ENABLE_AVX512F() _allow_cpu_features(_FEATURE_AVX512F)
#define pragma_opt_core_avx512                                                 \
  _Pragma("optimization_parameter target_arch=CORE-AVX512")
#define pragma_unroll _Pragma("unroll")
#define pragma_inline _Pragma("forceinline recursive")
#else
#define ENABLE_AVX512F() // -mavx512f
#define pragma_opt_core_avx512
#define pragma_unroll
#define pragma_inline
#endif

inline void el_error(const char *msg) {
  printf("Euler:Error: %s\n", msg);
  abort();
}

inline void el_warn(const char *msg) {
  printf("Euler:Warning: %s\n", msg);
}

} // namespace euler

#endif // __EL_DEF_HPP__
