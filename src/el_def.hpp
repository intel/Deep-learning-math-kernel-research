#include <stdio.h>
#include <omp.h>

#ifndef __EL_DEF_HPP__
#define __EL_DEF_HPP__

namespace euler {

enum {
    ISA_GENERIC     = 0,
    ISA_SKX_AVX512  = 512,
};

#ifdef __INTEL_COMPILER
#define ENABLE_AVX512F() _allow_cpu_features(_FEATURE_AVX512F)
#else
#define ENABLE_AVX512F() // -mavx512f
#endif

inline void eld_error(const char *msg) {
    printf("Euler:d: %s\n", msg);
}

inline void elx_error(const char *msg) {
    printf("Euler:x: %s\n", msg);
}

}

#endif // __EL_DEF_HPP__
