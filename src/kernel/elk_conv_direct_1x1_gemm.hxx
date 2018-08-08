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


#define _zmm_out(O_, T_) zmm_out##O_##T_ // output
#define _zmm_wei(O_, P_) zmm_wei##O_##P_ // weights
#define zmm_out(O_, T_) _zmm_out(O_, T_)
#define zmm_wei(O_, P_) _zmm_wei(O_, P_)

#define DEF_OUTPUT(O_, T_) __m512 zmm_out(O_, T_)
#define DEF_WEIGHTS(O_, P_) __m512 zmm_wei(O_, P_)
#define _0(tuple) BOOST_PP_TUPLE_ELEM(2, 0, tuple)
#define _1(tuple) BOOST_PP_TUPLE_ELEM(2, 1, tuple)

// Load bias
// 
// for_each (_O1, O1)
//   tmp = _mm512_load_ps(&md2(abias, _O1, 0));
//   for_each (_T, T)
//     zmm_out(_O1, _T) = tmp;
#define MM_LOAD_BIAS(O_, T_) BOOST_PP_REPEAT(O_, _MM_LOAD_BIAS, T_);
#define _MM_LOAD_BIAS(z, O_, T_)                                               \
  tmp = _mm512_load_ps(&md2(abias2, O_, 0));                                   \
  BOOST_PP_REPEAT(T_, __MM_LOAD_BIAS, O_);
#define __MM_LOAD_BIAS(z, T_, O_) zmm_out(O_, T_) = tmp;

// Clear zmm output
#define _MM_CLEAR_OUTPUT(O_, T_) zmm_out(O_, T_) = _mm512_setzero_ps()
#define MM_CLEAR_OUTPUT(O_, T_) MATRIX_OP(_MM_CLEAR_OUTPUT, O_, T_)

// Load output
//
// for_each (O_, O)
//   MD3(float, aoutput3, &md2(aoutput, _O1, 0), xc.t2, T_, V);
//   for_each (T_, T)
//     zmm_out(O_, _T) = _mm512_load_ps(&md3(aoutput3, 0, T_, 0));
#define MM_LOAD_OUTPUT(O_, T_) BOOST_PP_REPEAT(O_, _MM_LOAD_OUTPUT, T_);
#define _MM_LOAD_OUTPUT(z, O_, T_)                                             \
  MD3(float, aoutput3##O_, &md2(aoutput2, O_, 0), xc.t2, T_, V);               \
  BOOST_PP_REPEAT(T_, __MM_LOAD_OUTPUT, O_);
#define __MM_LOAD_OUTPUT(z, T_, O_)                                            \
  zmm_out(O_, T_) = _mm512_load_ps(&md3(aoutput3##O_, 0, T_, 0));

// Load weights
#define _MM_LOAD_WEIGHTS_P1(O_, P_)                                            \
  zmm_wei(O_, P_) = _mm512_load_ps(&md5(aweights5, O_, 0, _I2, _V, 0))
#define MM_LOAD_WEIGHTS_P1(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P1, O_, 1);

// Compute output
//
// for_each (_T, T)
//   bcast = _mm512_broadcastss_ps(*(__m128 *)&md3(ainput3, 0, T_, _V));
//   for_each (O_, O)
//     zmm_out(O_, _T)
//         = _mm512_fmadd_ps(zmm_wei(O_, 0), bcast, zmm_out(O_, T_));
#define MM_COMPUTE_OUTPUT_P1(O_, T_)                                           \
  BOOST_PP_REPEAT(T_, _MM_COMPUTE_OUTPUT_P1, O_);
#define _MM_COMPUTE_OUTPUT_P1(z, T_, O_)                                       \
  bcast = _mm512_broadcastss_ps(*(__m128 *)&md3(ainput3, 0, T_, _V));          \
  BOOST_PP_REPEAT(O_, __MM_COMPUTE_OUTPUT_P1, T_);
#define __MM_COMPUTE_OUTPUT_P1(z, O_, T_)                                      \
  zmm_out(O_, T_) = _mm512_fmadd_ps(zmm_wei(O_, 0), bcast, zmm_out(O_, T_));

// Store output
//
// for_each (O_, O)
//   MD3(float, aoutput3, &md2(aoutput, O_, 0), xc.t2, T, V);
//   for_each (_T, T)
//     _mm512_store_ps(&md3(aoutput3, 0, _T, 0), zmm_out[_O1][_T]);
#define MM_STORE_OUTPUT(O_, T_) BOOST_PP_REPEAT(O_, _MM_STORE_OUTPUT, T_);
#define _MM_STORE_OUTPUT(z, O_, T_)                                            \
  MD3(float, aoutput3##O_, &md2(aoutput2, O_, 0), xc.t2, T_, V);               \
  BOOST_PP_REPEAT(T_, __MM_STORE_OUTPUT, O_);
#define __MM_STORE_OUTPUT(z, T_, O_)                                           \
  _mm512_store_ps(&md3(aoutput3##O_, 0, T_, 0), zmm_out(O_, T_));

#define K_GEMM_FMA_P1(O_, T_, Os_)                                             \
  {                                                                            \
    __m512 bcast;                                                              \
    MATRIX_OP(DEF_OUTPUT, O_, T_);                                             \
    MATRIX_OP(DEF_WEIGHTS, O_, 1);                                             \
                                                                               \
    MD2(float, aoutput2, &md3(aoutput, _oc3, Os_, 0), O_, xc.oh *xc.ow *V);    \
    MD5(float, aweights5, &md4(aweights, _oc3, Os_, _ic3, 0), O_, xc.ic3,      \
        xc.I2, V, V);                                                          \
    MD2(float, abias2, &md3(abias, _oc3, Os_, 0), O_, V);                      \
                                                                               \
    if (_ic3 == 0) {                                                           \
      if (with_bias) {                                                         \
        __m512 tmp;                                                            \
        MM_LOAD_BIAS(O_, T_);                                                  \
      } else {                                                                 \
        MM_CLEAR_OUTPUT(O_, T_);                                               \
      }                                                                        \
    } else {                                                                   \
      MM_LOAD_OUTPUT(O_, T_);                                                  \
    }                                                                          \
    for_each (_I2, xc.I2) {                                                    \
      MD3(float, ainput3, &md3(ainput, _ic3, _I2, 0), xc.t2, T_, V);           \
      pragma_unroll for_each (_V, V)                                           \
      {                                                                        \
        MM_LOAD_WEIGHTS_P1(O_);                                                \
        MM_COMPUTE_OUTPUT_P1(O_, T_);                                          \
      }                                                                        \
    }                                                                          \
    MM_STORE_OUTPUT(O_, T_);                                                   \
  }

#define _MM_PRELOAD_WEIGHTS_P2(O_, nil)                                        \
  zmm_wei(O_, 0) = _mm512_load_ps(&md6(aweights6, O_, 0, 0, 0, 0, 0));
#define MM_PRELOAD_WEIGHTS_P2(O_) MATRIX_OP(_MM_PRELOAD_WEIGHTS_P2, O_, 1)

#define _MM_LOAD_WEIGHTS_P2_1(O_, nil)                                         \
  zmm_wei(O_, 1) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V, 1, 0));
#define MM_LOAD_WEIGHTS_P2_1(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P2_1, O_, 1);

#define _MM_LOAD_WEIGHTS_P2_0(O_, nil)                                         \
  zmm_wei(O_, 0) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V + 1, 0, 0));
#define MM_LOAD_WEIGHTS_P2_0(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P2_0, O_, 1);

#define MM_COMPUTE_OUTPUT_P(O_, T_, P_)                                        \
  BOOST_PP_REPEAT(T_, _MM_COMPUTE_OUTPUT_P, (O_, P_));
#define _MM_COMPUTE_OUTPUT_P(z, T_, OP_)                                       \
  bcast = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, 0, T_, _V, _1(OP_))); \
  BOOST_PP_REPEAT(_0(OP_), __MM_COMPUTE_OUTPUT_P, (T_, _1(OP_)));
#define __MM_COMPUTE_OUTPUT_P(z, O_, TP_)                                      \
  zmm_out(O_, _0(TP_))                                                         \
      = _mm512_fmadd_ps(zmm_wei(O_, _1(TP_)), bcast, zmm_out(O_, _0(TP_)));

#define K_GEMM_FMA_P2(O_, T_, Os_)                                             \
  {                                                                            \
    __m512 bcast;                                                              \
    MATRIX_OP(DEF_OUTPUT, O_, T_);                                             \
    MATRIX_OP(DEF_WEIGHTS, O_, 2);                                             \
                                                                               \
    MD2(float, aoutput2, &md3(aoutput, _oc3, Os_, 0), O_, xc.oh *xc.ow *V);    \
    MD6(float, aweights6, &md4(aweights, _oc3, Os_, _ic3, 0), O_, xc.ic3,      \
        xc.I2, V / 2, 2, V);                                                   \
    MD2(float, abias2, &md3(abias, _oc3, Os_, 0), O_, V);                      \
                                                                               \
    MM_PRELOAD_WEIGHTS_P2(O_);                                                 \
                                                                               \
    if (_ic3 == 0) {                                                           \
      if (with_bias) {                                                         \
        __m512 tmp;                                                            \
        MM_LOAD_BIAS(O_, T_);                                                  \
      } else {                                                                 \
        MM_CLEAR_OUTPUT(O_, T_);                                               \
      }                                                                        \
    } else {                                                                   \
      MM_LOAD_OUTPUT(O_, T_);                                                  \
    }                                                                          \
    for_each (_I2, xc.I2) {                                                    \
      MD4(Type, ainput4, &md3(ainput, _ic3, _I2, 0), xc.t2, T_, V / 2, 2);     \
      pragma_unroll for_each (_V, V / 2)                                       \
      {                                                                        \
        MM_LOAD_WEIGHTS_P2_1(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 0);                                        \
                                                                               \
        MM_LOAD_WEIGHTS_P2_0(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 1);                                        \
      }                                                                        \
    }                                                                          \
    MM_STORE_OUTPUT(O_, T_);                                                   \
  }

#define _MM_PRELOAD_WEIGHTS_P4(O_, nil)                                        \
  zmm_wei(O_, 0) = _mm512_load_ps(&md6(aweights6, O_, 0, 0, 0, 0, 0));         \
  zmm_wei(O_, 1) = _mm512_load_ps(&md6(aweights6, O_, 0, 0, 0, 1, 0));
#define MM_PRELOAD_WEIGHTS_P4(O_) MATRIX_OP(_MM_PRELOAD_WEIGHTS_P4, O_, 1)

#define _MM_LOAD_WEIGHTS_P4_2(O_, nil)                                         \
  zmm_wei(O_, 2) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V, 2, 0));
#define MM_LOAD_WEIGHTS_P4_2(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P4_2, O_, 1);

#define _MM_LOAD_WEIGHTS_P4_3(O_, nil)                                         \
  zmm_wei(O_, 3) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V, 3, 0));
#define MM_LOAD_WEIGHTS_P4_3(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P4_3, O_, 1);

#define _MM_LOAD_WEIGHTS_P4_0(O_, nil)                                         \
  zmm_wei(O_, 0) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V + 1, 0, 0));
#define MM_LOAD_WEIGHTS_P4_0(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P4_0, O_, 1);

#define _MM_LOAD_WEIGHTS_P4_1(O_, nil)                                         \
  zmm_wei(O_, 1) = _mm512_load_ps(&md6(aweights6, O_, 0, _I2, _V + 1, 1, 0));
#define MM_LOAD_WEIGHTS_P4_1(O_) MATRIX_OP(_MM_LOAD_WEIGHTS_P4_1, O_, 1);

#define K_GEMM_FMA_P4(O_, T_, Os_)                                             \
  {                                                                            \
    __m512 bcast;                                                              \
    MATRIX_OP(DEF_OUTPUT, O_, T_);                                             \
    MATRIX_OP(DEF_WEIGHTS, O_, 4);                                             \
                                                                               \
    MD2(float, aoutput2, &md3(aoutput, _oc3, Os_, 0), O_, xc.oh *xc.ow *V);    \
    MD6(float, aweights6, &md4(aweights, _oc3, Os_, _ic3, 0), O_, xc.ic3,      \
        xc.I2, V / 4, 4, V);                                                   \
    MD2(float, abias2, &md3(abias, _oc3, Os_, 0), O_, V);                      \
                                                                               \
    MM_PRELOAD_WEIGHTS_P4(O_);                                                 \
                                                                               \
    if (_ic3 == 0) {                                                           \
      if (with_bias) {                                                         \
        __m512 tmp;                                                            \
        MM_LOAD_BIAS(O_, T_);                                                  \
      } else {                                                                 \
        MM_CLEAR_OUTPUT(O_, T_);                                               \
      }                                                                        \
    } else {                                                                   \
      MM_LOAD_OUTPUT(O_, T_);                                                  \
    }                                                                          \
    for_each (_I2, xc.I2) {                                                    \
      MD4(Type, ainput4, &md3(ainput, _ic3, _I2, 0), xc.t2, T_, V / 4, 4);     \
      pragma_unroll for_each (_V, V / 4)                                       \
      {                                                                        \
        MM_LOAD_WEIGHTS_P4_2(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 0);                                        \
                                                                               \
        MM_LOAD_WEIGHTS_P4_3(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 1);                                        \
                                                                               \
        MM_LOAD_WEIGHTS_P4_0(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 2);                                        \
                                                                               \
        MM_LOAD_WEIGHTS_P4_1(O_);                                              \
        MM_COMPUTE_OUTPUT_P(O_, T_, 3);                                        \
      }                                                                        \
    }                                                                          \
    MM_STORE_OUTPUT(O_, T_);                                                   \
  }

#if 0
if (first_ic3) {
  if (with_bias) { // load bias
    __m512 tmp;
#define LOAD_BIAS(_O1) _mm512_load_ps(&md2(abias, _O1, 0))
        MM_LOAD_BIAS(O1, T);
      } else { // clear output
        for_each (_O1, O1) {
          for_each (_T, T) {
            zmm_out[_O1][_T] = _mm512_setzero_ps();
          }
        }
      }
    } else { // load output
      for_each (_O1, O1) {
        MD3(float, aoutput3, &md2(aoutput, _O1, 0), xc.t2, T, V);
        for_each (_T, T) {
          zmm_out[_O1][_T] = _mm512_load_ps(&md3(aoutput3, 0, _T, 0));
        }
      }
    }

    for_each (_I2, xc.I2) {
      MD3(float, ainput3, &md3(ainput, _ic3, _I2, 0), xc.t2, T, V);
      for_each (_V, V) {
        for_each (_O1, O1) { // load weights
          for_each (_P, 1) {
            zmm_wei[_O1][_P]
                = _mm512_load_ps(&md5(aweights, _O1, 0, _I2, _V, 0));
          }
        }

        for_each (_T, T) { // fma
          bcast = _mm512_broadcastss_ps(*(__m128 *)&md3(ainput3, 0, _T, _V));
          for_each (_O1, O1) {
            zmm_out[_O1][_T]
                = _mm512_fmadd_ps(zmm_wei[_O1][0], bcast, zmm_out[_O1][_T]);
          }
        }
      }
    }

    for_each (_O1, O1) {
      MD3(float, aoutput3, &md2(aoutput, _O1, 0), xc.t2, T, V);
      for_each (_T, T)
        _mm512_store_ps(&md3(aoutput3, 0, _T, 0), zmm_out[_O1][_T]);
    }
  }
};
#endif

//#define K(P_, O_, T_, Os_)                                                     \
//  else if (O2 == O_ && T == T_) { K_GEMM_FMA_P##P_(O_, T_, Os_); }

#define K(P_, O2_, T_, Os_)                                                    \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O2_, T_, V, I, true, with_bias,     \
      with_relu, with_sum>::gemm(elx_conv_t<Type> &xc, Type *output,           \
      Type *input, Type *weights, Type *bias)                                  \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD3(Type, aoutput, output, xc.oc3, O2_, xc.oh *xc.ow *V);                  \
    MD3(Type, ainput, input, xc.ic3, xc.I2, xc.ih *xc.iw *V);                  \
    MD4(Type, aweights, weights, xc.oc3, O2_, xc.ic3, xc.I2 *V *V);            \
    MD3(Type, abias, bias, xc.oc3, O2_, V);                                    \
                                                                               \
    for_each (_ic3, xc.ic3) {                                                  \
      for_each (_oc3, xc.oc3) {                                                \
        K_GEMM_FMA_P##P_(O2_, T_, Os_);                                        \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512, TR(true),  \
      BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *, \
      float *, float *, float *);                                              \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512, TR(true),  \
      BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,         \
      float *, float *, float *, float *);


/////////////////////

template <typename Type, const int V, const int I, const bool with_bias,
    const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel<Type, 1, 1, V, I, true, with_bias,
    with_relu, with_sum>::gemm(elx_conv_t<Type> &xc, Type *output, Type *input,
    Type *weights, Type *bias)
{
  ENABLE_AVX512F();

  MD3(Type, aoutput, output, xc.oc3, 1, xc.oh * xc.ow * V);
  MD3(Type, ainput, input, xc.ic3, xc.I2, xc.ih * xc.iw * V);
  MD4(Type, aweights, weights, xc.oc3, 1, xc.ic3, xc.I2 * V * V);
  MD3(Type, abias, bias, xc.oc3, 1, V);

  for_each (_ic3, xc.ic3) {
    for_each (_oc3, xc.oc3) {
      {
        __m512 bcast;
        MATRIX_OP(DEF_OUTPUT, 1, 1);
        MATRIX_OP(DEF_WEIGHTS, 1, 1);

        MD2(float, aoutput2, &md3(aoutput, _oc3, 0, 0), 1, xc.oh *xc.ow *V);
        MD5(float, aweights5, &md4(aweights, _oc3, 0, _ic3, 0), 1, xc.ic3,
            xc.I2, V, V);
        MD2(float, abias2, &md3(abias, _oc3, 0, 0), 1, V);

        if (_ic3 == 0) {
          if (with_bias) {
            __m512 tmp;
            MM_LOAD_BIAS(1, 1);
          } else {
            MM_CLEAR_OUTPUT(1, 1);
          }
        } else {
          MM_LOAD_OUTPUT(1, 1);
        }
        for_each (_I2, xc.I2) {
          MD3(float, ainput3, &md3(ainput, _ic3, _I2, 0), xc.t2, 1, V);
          pragma_unroll for_each (_V, V)
          {
            MM_LOAD_WEIGHTS_P1(1);
            MM_COMPUTE_OUTPUT_P1(1, 1);
          }
        }
        MM_STORE_OUTPUT(1, 1);
      }
    }
  }
}
template void convolution_direct_1x1_kernel<float, 1, 1, 16, ISA_SKX_AVX512,
    TR(true), BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,
    float *, float *, float *, float *);
template void convolution_direct_1x1_kernel<float, 1, 1, 16, ISA_SKX_AVX512,
    TR(true), BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,
    float *, float *, float *, float *);

//////////////////////




// O=1, T:
//    T=31..:  kernel: 1, output: 31..
//    T=29,30: kernel: 2, output: 29 - 30 (pipeline: 2)
//    T=1..28: kenrel: 4, output: 1 - 28  (pipeline: 4)
//K(1, 1, 1, 0) // XXX
K(4, 1, 2, 0)
K(4, 1, 3, 0)
K(4, 1, 4, 0)
K(4, 1, 5, 0)
K(4, 1, 6, 0)
K(4, 1, 7, 0)
K(4, 1, 8, 0)
K(4, 1, 9, 0)
K(4, 1, 10, 0)
K(4, 1, 11, 0)
K(4, 1, 12, 0)
K(4, 1, 13, 0)
K(4, 1, 14, 0)
K(4, 1, 15, 0)
K(4, 1, 16, 0)
K(4, 1, 17, 0)
K(4, 1, 18, 0)
K(4, 1, 19, 0)
K(4, 1, 20, 0)
K(4, 1, 21, 0)
K(4, 1, 22, 0)
K(4, 1, 23, 0)
K(4, 1, 24, 0)
K(4, 1, 25, 0)
K(4, 1, 26, 0)
K(4, 1, 27, 0)
K(4, 1, 28, 0)
K(2, 1, 29, 0)
K(2, 1, 30, 0)
K(1, 1, 31, 0)
K(1, 1, 32, 0)

// O=2, T:
//    T=14:    bcast: 1, kernel: 2, output: 28
//    T=12,13: bcast: 1, kernel: 4, output: 24,26 (pipeline: 2)
//    T=1..11: bcast: 1, kernel: 8, output: 2..22 (pipeline: 4)
K(4, 2, 1, 0)
K(4, 2, 2, 0)
K(4, 2, 3, 0)
K(4, 2, 4, 0)
K(4, 2, 5, 0)
K(4, 2, 6, 0)
K(4, 2, 7, 0)
K(4, 2, 8, 0)
K(4, 2, 9, 0)
K(4, 2, 10, 0)
K(4, 2, 11, 0)
K(2, 2, 12, 0)
K(2, 2, 13, 0)
K(1, 2, 14, 0)

// O=3, T:
//    T=8:     bcast: 1, kernel 3, output: 24
//    T=7:     bcast: 1, kernel 6, output: 21 (pipeline: 2)
//    T=1..6:  bcast: 1, kernel 12, output: 3..18 (pipeline: 4)
K(4, 3, 1, 0)
K(4, 3, 2, 0)
K(4, 3, 3, 0)
K(4, 3, 4, 0)
K(4, 3, 5, 0)
K(4, 3, 6, 0)
K(2, 3, 7, 0)
K(1, 3, 8, 0)

// O=4, T:
//    T=6:     bcast: 1, kernel: 4, outupt: 24
//    T=1..5:  bcast: 1, kernel: 8, outupt: 4..20 (pipeline: 2)
K(2, 4, 1, 0)
K(2, 4, 2, 0)
K(2, 4, 3, 0)
K(2, 4, 4, 0)
K(2, 4, 5, 0)
K(1, 4, 6, 0)
K(1, 4, 7, 0)

// O=5, T:
//    T=5:     bcast: 1, kernel: 5, output: 25
//    T=3,4:   bcast: 1, kernel: 10, output: 15,20 (pipeline: 2)
//    T=1,2:   bcast: 1, kernel: 20, output: 5,10 (pipeline: 2)
K(4, 5, 1, 0)
K(4, 5, 2, 0)
K(2, 5, 3, 0)
K(2, 5, 4, 0)
K(1, 5, 5, 0)

// O=6, T:
//    T=4:     bcast: 1, kenrel: 6, output: 24
//    T=2,3:   bcast: 1, kernel: 12, output: 12,18 (pipeline: 2)
//    T=1:     bcast: 1, kernel: 24, output: 6 (pipeline: 4)
K(4, 6, 1, 0)
K(2, 6, 2, 0)
K(2, 6, 3, 0)
K(1, 6, 4, 0)

// O=7, T:
//    T=3:     bcast: 1, kernel: 7, output: 21
//    T=1,2:   bcast: 1, kernel: 14, output: 7,14 (pipeline: 2)
K(2, 7, 1, 0)
K(2, 7, 2, 0)
K(1, 7, 3, 0)

// O=8, T:
//    T=2:     bcast: 1, kernel: 8, output: 16
//    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)
K(1, 8, 2, 0)
K(2, 8, 1, 0)


#if 0
#define CONV1X1_GEMM(O_, T_)                                                   \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O_, T_, 16, ISA_SKX_AVX512, TR(true),   \
      BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *, \
      float *, float *, float *);                                              \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O_, T_, 16, ISA_SKX_AVX512, TR(true),   \
      BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,         \
      float *, float *, float *, float *);

CONV1X1_GEMM(1, 1);
CONV1X1_GEMM(1, 2);
CONV1X1_GEMM(1, 3);
CONV1X1_GEMM(1, 4);
CONV1X1_GEMM(1, 5);
CONV1X1_GEMM(1, 6);
CONV1X1_GEMM(1, 7);
CONV1X1_GEMM(1, 8);
CONV1X1_GEMM(1, 9);
CONV1X1_GEMM(1, 10);
CONV1X1_GEMM(1, 11);
CONV1X1_GEMM(1, 12);
CONV1X1_GEMM(1, 13);
CONV1X1_GEMM(1, 14);
CONV1X1_GEMM(1, 15);
CONV1X1_GEMM(1, 16);
CONV1X1_GEMM(1, 17);
CONV1X1_GEMM(1, 18);
CONV1X1_GEMM(1, 19);
CONV1X1_GEMM(1, 20);
CONV1X1_GEMM(1, 21);
CONV1X1_GEMM(1, 22);
CONV1X1_GEMM(1, 23);
CONV1X1_GEMM(1, 24);
CONV1X1_GEMM(1, 25);
CONV1X1_GEMM(1, 26);
CONV1X1_GEMM(1, 27);
CONV1X1_GEMM(1, 28);
CONV1X1_GEMM(1, 29);
CONV1X1_GEMM(1, 30);
CONV1X1_GEMM(1, 31);
CONV1X1_GEMM(1, 32);
CONV1X1_GEMM(1, 33);
CONV1X1_GEMM(1, 34);
CONV1X1_GEMM(1, 35);
CONV1X1_GEMM(2, 1);
CONV1X1_GEMM(2, 2);
CONV1X1_GEMM(2, 3);
CONV1X1_GEMM(2, 4);
CONV1X1_GEMM(2, 5);
CONV1X1_GEMM(2, 6);
CONV1X1_GEMM(2, 7);
CONV1X1_GEMM(2, 8);
CONV1X1_GEMM(2, 9);
CONV1X1_GEMM(2, 10);
CONV1X1_GEMM(2, 11);
CONV1X1_GEMM(2, 12);
CONV1X1_GEMM(2, 13);
CONV1X1_GEMM(2, 14);
CONV1X1_GEMM(3, 1);
CONV1X1_GEMM(3, 2);
CONV1X1_GEMM(3, 3);
CONV1X1_GEMM(3, 4);
CONV1X1_GEMM(3, 5);
CONV1X1_GEMM(3, 6);
CONV1X1_GEMM(3, 7);
CONV1X1_GEMM(3, 8);
CONV1X1_GEMM(4, 1);
CONV1X1_GEMM(4, 2);
CONV1X1_GEMM(4, 3);
CONV1X1_GEMM(4, 4);
CONV1X1_GEMM(4, 5);
CONV1X1_GEMM(4, 6);
CONV1X1_GEMM(4, 7);
CONV1X1_GEMM(5, 1);
CONV1X1_GEMM(5, 2);
CONV1X1_GEMM(5, 3);
CONV1X1_GEMM(5, 4);
CONV1X1_GEMM(5, 5);
CONV1X1_GEMM(6, 1);
CONV1X1_GEMM(6, 2);
CONV1X1_GEMM(6, 3);
CONV1X1_GEMM(6, 4);
CONV1X1_GEMM(7, 1);
CONV1X1_GEMM(7, 2);
CONV1X1_GEMM(7, 3);
CONV1X1_GEMM(8, 2);
CONV1X1_GEMM(8, 1);
#endif
}
