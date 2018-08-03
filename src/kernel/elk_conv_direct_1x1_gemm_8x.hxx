#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_def.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

#ifndef INCLUDE_DIRECT_CONVOLUTION_1X1_KERNEL
#error "Don't include this file directly"
#endif

namespace euler {

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

#undef t
#undef w
#define t(_O2,_T) t##_O2##_T   // output
#define w(_O2,_P) w##_O2##_P   // weights
  // 8, T:
  //    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)
  for_each (_ic3, xc.ic3) {
    __m512 bcast;
#undef OP
#define OP(_O2,_T) __m512 t(_O2,_T)
    MATRIX_DEF(8, 1);
#undef OP
#define OP(_O2, _P) __m512 w(_O2, _P)
    MATRIX_DEF(8, 2);
    // preload weights0
#undef OP
#define OP(_O2, _P)                                                            \
  w(_O2, 0) = _mm512_load_ps(&md6(aweights, _O2, _ic3, 0, 0, 0, 0));
    MATRIX_DEF(8, 1);

    if (_ic3 == 0) {
      if (with_bias) {
#undef OP
#define OP(_O2, _T) t(_O2, _T) = _mm512_load_ps(&md2(abias, _O2, 0))
        MATRIX_DEF(8, 1);
      } else {
#undef OP
#define OP(_O2, _T) t(_O2, _T) = _mm512_setzero_ps()
        MATRIX_DEF(8, 1);
      }
    } else {
#undef OP
#define OP(_O2, _T) t(_O2, _T) = _mm512_load_ps(&md4(aoutput, _O2, 0, _T, 0))
      MATRIX_DEF(8, 1);
    }
    for_each (_I2, xc.I2) {
      for_each (_V, V / 2) {
        // load weights1, load data (bcast), fma
#undef OP
#define OP(_O2, _P)                                                            \
  w(_O2, 1) = _mm512_load_ps(&md6(aweights, _O2, _ic3, _I2, _V, 1, 0));
        MATRIX_DEF(8, 1);
#undef OP
#define OP(_O2, _T)                                                            \
  bcast = _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0)); \
  t(_O2, _T) = _mm512_fmadd_ps(w(_O2, 0), bcast, t(_O2, _T))
        MATRIX_DEF(8, 1);
        // load weights0, load data (bcast), fma
#undef OP
#define OP(_O2, _P)                                                            \
  w(_O2, 0) = _mm512_load_ps(&md6(aweights, _O2, _ic3, _I2, _V + 1, 0, 0));
        MATRIX_DEF(8, 1);
#undef OP
#define OP(_O2, _T)                                                            \
  bcast = _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 1)); \
  t(_O2, _T) = _mm512_fmadd_ps(w(_O2, 1), bcast, t(_O2, _T))
        MATRIX_DEF(8, 1);
      }
    }
#undef OP
#define OP(_O2, _T) _mm512_store_ps(&md4(aoutput, _O2, 0, _T, 0), t(_O2, _T))
    MATRIX_DEF(8, 1);
  }
}

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

  // 8, T:
  //    T=2:     bcast: 1, kernel: 8, output: 16
  for_each (_ic3, xc.ic3) {
#undef t
#define t(_O2,_T) t##_O2##_T
#undef OP
#define OP(_O2,_T) __m512 t(_O2,_T)
    MATRIX_DEF(8, 2);
#undef w
#define w(_O2, _P) w##_O2##_P
#undef OP
#define OP(_O2, _P) __m512 w(_O2, _P)
    MATRIX_DEF(8, 1);

    __m512 bcast;

    if (_ic3 == 0) {
      if (with_bias) {
#undef OP
#define OP(_O2,_T) t(_O2,_T) = _mm512_load_ps(&md2(abias, _O2, 0))
        MATRIX_DEF(8, 2);
      } else {
#undef OP
#define OP(_O2,_T) t(_O2,_T) = _mm512_setzero_ps()
        MATRIX_DEF(8, 2);
      }
    } else {
#undef OP
#define OP(_O2,_T) t(_O2,_T) = _mm512_load_ps(&md4(aoutput, _O2, 0, _T, 0))
      MATRIX_DEF(8, 2);
    }
    for_each (_I2, xc.I2) {
      for_each (_V, V) {
#undef OP
#define OP(_O2,_P) w(_O2,_P) = _mm512_load_ps(&md5(aweights, _O2, _ic3, _I2, _V, 0))
        MATRIX_DEF(8, 1);

#undef OP
#define OP(_O2, _T)                                                            \
  bcast                                                                        \
      = _mm512_broadcastss_ps(*(__m128 *)&md5(ainput, _ic3, _I2, 0, _T, _V));  \
  t(_O2, _T) = _mm512_fmadd_ps(w(_O2, 0), bcast, t(_O2, _T))
        MATRIX_DEF(8, 2);
      }
    }
#undef OP
#define OP(_O2, _T) _mm512_store_ps(&md4(aoutput, _O2, 0, _T, 0), t(_O2, _T))
    MATRIX_DEF(8, 2);
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
