#include <assert.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

namespace euler {

template <typename Type, const int O2, const int T, const int V, const int I,
    const bool with_bias, const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel<Type, O2, T, V, I, with_bias, with_relu,
    with_sum>::gemm(elx_conv_t<Type> &, Type *, Type *, Type *, Type *)
{
}

template <typename Type, const int O2, const int T, const int V, const int I,
    const bool with_bias, const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel<Type, O2, T, V, I, with_bias, with_relu,
    with_sum>::gemm_tail(elx_conv_t<Type> &, Type *, Type *, Type *, Type *)
{
}
}

#define INCLUDE_DIRECT_CONVOLUTION_1X1_KERNEL
#include "kernel/elk_conv_direct_1x1_gemm_1x.hxx"
