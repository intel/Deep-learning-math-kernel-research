#include <assert.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

namespace euler {

/*
template <typename Type, const int O2, const int T, const int V, const int I,
    const bool with_Tr, const bool with_bias, const bool with_relu,
    const bool with_sum>
void convolution_direct_1x1_kernel<Type, O2, T, V, I, with_Tr, with_bias,
    with_relu, with_sum>::gemm(elx_conv_t<Type> &, Type *, Type *, Type *,
    Type *)
{
}

template <typename Type, const int O2, const int T, const int V, const int I,
    const bool with_Tr, const bool with_bias, const bool with_relu,
    const bool with_sum>
void convolution_direct_1x1_kernel<Type, O2, T, V, I, with_Tr, with_bias,
    with_relu, with_sum>::gemm_tail(elx_conv_t<Type> &, Type *, Type *, Type *,
    Type *)
{
}
*/

}

// AVX512 registers allocation
// O=1, T:
//    T=31:    kernel: 1, output: 31
//    T=29,30: kernel: 2, output: 29 - 30 (pipeline: 2)
//    T=1..28: kenrel: 4, output: 1 - 28  (pipeline: 4)
// O=2, T:
//    T=14:    bcast: 1, kernel: 2, output: 28
//    T=12,13: bcast: 1, kernel: 4, output: 24,26 (pipeline: 2)
//    T=1..11: bcast: 1, kernel: 8, output: 2..22 (pipeline: 4)
// O=3, T:
//    T=8:     bcast: 1, kernel 3, output: 24
//    T=7:     bcast: 1, kernel 6, output: 21 (pipeline: 2)
//    T=1..6:  bcast: 1, kernel 12, output: 3..18 (pipeline: 4)
// O=4, T:
//    T=6:     bcast: 1, kernel: 4, outupt: 24
//    T=1..5:  bcast: 1, kernel: 8, outupt: 4..20 (pipeline: 2)
// O=5, T:
//    T=5:     bcast: 1, kernel: 5, output: 25
//    T=3,4:   bcast: 1, kernel: 10, output: 15,20 (pipeline: 2)
//    T=1,2:   bcast: 1, kernel: 20, output: 5,10 (pipeline: 2)
// O=6, T:
//    T=4:     bcast: 1, kenrel: 6, output: 24
//    T=2,3:   bcast: 1, kernel: 12, output: 12,18 (pipeline: 2)
//    T=1:     bcast: 1, kernel: 24, output: 6 (pipeline: 4)
// O=7, T:
//    T=3:     bcast: 1, kernel: 7, output: 21
//    T=1,2:   bcast: 1, kernel: 14, output: 7,14 (pipeline: 2)
// O=8, T:
//    T=2:     bcast: 1, kernel: 8, output: 16
//    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)

#define INCLUDE_DIRECT_CONVOLUTION_1X1_KERNEL
//#include "kernel/elk_conv_direct_1x1_gemm_1x.hxx"
//#include "kernel/elk_conv_direct_1x1_gemm_8x.hxx"
//#include "kernel/elk_conv_direct_1x1_gemm_Tr.hxx"
#include "kernel/elk_conv_direct_1x1_gemm.hxx"
