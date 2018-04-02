#include <stdio.h>
#include <omp.h>

#ifndef __EL_DEF_HPP__
#define __EL_DEF_HPP__

namespace euler {

enum {
    ISA_GENERIC     = 0,
    ISA_SKX_AVX512  = 512,
};

inline void eld_error(const char *msg) {
    printf("Euler:d: %s\n", msg);
}

inline void elx_error(const char *msg) {
    printf("Euler:x: %s\n", msg);
}

}

#endif // __EL_DEF_HPP__
