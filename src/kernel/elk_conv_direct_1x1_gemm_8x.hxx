#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_def.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

#ifndef INCLUDE_DIRECT_CONVOLUTION_1X1_KERNEL
#error "Don'zmm_out include this file directly"
#endif

namespace euler {

#undef zmm_out
#undef zmm_wei

#define zmm_out(_O, _T) zmm_out##_O##_T // output
#define zmm_wei(_O, _P) zmm_wei##_O##_P // weights
#define DEF_OUTPUT(_O, _T) __m512 zmm_out(_O, _T)
#define DEF_WEIGHTS(_O, _P) __m512 zmm_wei(_O, _P)
#define LOAD_BIAS(_O, _T) zmm_out(_O, _T) = _mm512_load_ps(&md2(abias, _O, 0))
#define CLEAR_OUTPUT(_O, _T) zmm_out(_O, _T) = _mm512_setzero_ps()
#define LOAD_OUTPUT(_O, _T)                                                    \
  zmm_out(_O, _T) = _mm512_load_ps(&md4(aoutput, _O, 0, _T, 0))
#define STORE_OUTPUT(_O, _T)                                                   \
  _mm512_store_ps(&md4(aoutput, _O, 0, _T, 0), zmm_out(_O, _T))

// No pipeline version
#define LOAD_WEIGHTS(_O, _P)                                                   \
  zmm_wei(_O, _P) = _mm512_load_ps(&md5(aweights, _O, _ic3, _I2, _V, 0))
#define COMPUTE_OUTPUT(_O, _T)                                                 \
  bcast                                                                        \
      = _mm512_broadcastss_ps(*(__m128 *)&md5(ainput, _ic3, _I2, 0, _T, _V));  \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T))

// Pipeline version
#define P_PRELOAD_WEIGHTS_0(_O, _P)                                            \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, 0, 0, 0, 0));
#define P_LOAD_WEIGHTS_1(_O, _P)                                               \
  zmm_wei(_O, 1) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V, 1, 0));
#define P_LOAD_WEIGHTS_0(_O, _P)                                               \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V + 1, 0, 0));
#define P_COMPUTE_OUTPUT_0(_O, _T)                                             \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T))
#define P_COMPUTE_OUTPUT_1(_O, _T)                                             \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 1));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 1), bcast, zmm_out(_O, _T))

// 8, T:
//    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)
template <typename Type, const int V, const int I, const bool with_bias,
    const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel<Type, 8, 1, V, I, with_bias, with_relu,
    with_sum>::gemm(elx_conv_t<Type> &xc, Type *output, Type *input,
    Type *weights, Type *bias)
{
  ENABLE_AVX512F();

  MD4(Type, aoutput, output, 8, xc.t2, 1, V);
  MD6(Type, ainput, input, xc.ic3, xc.I2, xc.t2, 1, V / 2, 2);
  MD6(Type, aweights, weights, 8, xc.ic3, xc.I2, V / 2, 2, V);
  MD2(Type, abias, bias, 8, V);

  for_each (_ic3, xc.ic3) {
    __m512 bcast;
    MATRIX_OP(DEF_OUTPUT, 8, 1);
    MATRIX_OP(DEF_WEIGHTS, 8, 2);
    MATRIX_OP(P_PRELOAD_WEIGHTS_0, 8, 1);
    if (_ic3 == 0) {
      if (with_bias) {
        MATRIX_OP(LOAD_BIAS, 8, 1);
      } else {
        MATRIX_OP(CLEAR_OUTPUT, 8, 1);
      }
    } else {
      MATRIX_OP(LOAD_OUTPUT, 8, 1);
    }
    for_each (_I2, xc.I2) {
      for_each (_V, V / 2) {
        MATRIX_OP(P_LOAD_WEIGHTS_1, 8, 1);
        MATRIX_OP(P_COMPUTE_OUTPUT_0, 8, 1);

        MATRIX_OP(P_LOAD_WEIGHTS_0, 8, 1);
        MATRIX_OP(P_COMPUTE_OUTPUT_1, 8, 1);
      }
    }
    MATRIX_OP(STORE_OUTPUT, 8, 1);
  }
}

// 8, T:
//    T=2:     bcast: 1, kernel: 8, output: 16
template <typename Type, const int V, const int I, const bool with_bias,
    const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel<Type, 8, 2, V, I, with_bias, with_relu,
    with_sum>::gemm(elx_conv_t<Type> &xc, Type *output, Type *input,
    Type *weights, Type *bias)
{
  ENABLE_AVX512F();

  MD4(Type, aoutput, output, 8, xc.t2, 2, V);
  MD5(Type, ainput, input, xc.ic3, xc.I2, xc.t2, 2, V);
  MD5(Type, aweights, weights, 8, xc.ic3, xc.I2, V, V);
  MD2(Type, abias, bias, 8, V);

  for_each (_ic3, xc.ic3) {
    __m512 bcast;
    MATRIX_OP(DEF_OUTPUT, 8, 2);
    MATRIX_OP(DEF_WEIGHTS, 8, 1);
    if (_ic3 == 0) {
      if (with_bias) {
        MATRIX_OP(LOAD_BIAS, 8, 2);
      } else {
        MATRIX_OP(CLEAR_OUTPUT, 8, 2);
      }
    } else {
      MATRIX_OP(LOAD_OUTPUT, 8, 2);
    }
    for_each (_I2, xc.I2) {
      for_each (_V, V) {
        MATRIX_OP(LOAD_WEIGHTS, 8, 1);
        MATRIX_OP(COMPUTE_OUTPUT, 8, 2);
      }
    }
    MATRIX_OP(STORE_OUTPUT, 8, 2);
  }
}

template void convolution_direct_1x1_kernel<float, 8, 1, 16, ISA_SKX_AVX512,
    BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *,
    float *, float *, float *);
template void convolution_direct_1x1_kernel<float, 8, 1, 16, ISA_SKX_AVX512,
    BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *,
    float *, float *, float *);
template void convolution_direct_1x1_kernel<float, 8, 2, 16, ISA_SKX_AVX512,
    BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *,
    float *, float *, float *);
template void convolution_direct_1x1_kernel<float, 8, 2, 16, ISA_SKX_AVX512,
    BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *,
    float *, float *, float *);
}
