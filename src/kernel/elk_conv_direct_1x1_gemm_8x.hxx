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

#define LOAD_INPUT(_T)                                                         \
  _mm512_broadcastss_ps(*(__m128 *)&md5(ainput, _ic3, _I2, 0, _T, _V))
#define P_LOAD_INPUT_0(_T)                                                     \
  _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0))
#define P_LOAD_INPUT_1(_T)                                                     \
  _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 1))
#define P_LOAD_INPUT_2(_T)                                                     \
  _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 2))
#define P_LOAD_INPUT_3(_T)                                                     \
  _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 3))

// Pipeline = 1 (No pipeline version)
#define LOAD_WEIGHTS(_O, _P)                                                   \
  zmm_wei(_O, _P) = _mm512_load_ps(&md5(aweights, _O, _ic3, _I2, _V, 0))

// for_eachi (_T, T) {
//   bcast = _mm512_broadcastss_ps(*(__m128 *)&md5(ainput, _ic3, _I2, 0, _T, _V));
//   for_each (_O, O) {
//     zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T));
//   }
// }

#define COMPUTE_OUTPUT_T_O(z, _O, _T)                                          \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T));
#define COMPUTE_OUTPUT_T(z, _T, _O)                                            \
  bcast = LOAD_INPUT(_T);                                                      \
  BOOST_PP_REPEAT(_O, COMPUTE_OUTPUT_T_O, _T);
#define COMPUTE_OUTPUT(_O, _T) BOOST_PP_REPEAT(_T, COMPUTE_OUTPUT_T, _O);

// Pipeline = 2
#define P2_PRELOAD_WEIGHTS_0(_O, _P)                                           \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, 0, 0, 0, 0));
#define P2_LOAD_WEIGHTS_1(_O, _P)                                              \
  zmm_wei(_O, 1) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V, 1, 0));
#define P2_LOAD_WEIGHTS_0(_O, _P)                                              \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V + 1, 0, 0));

// for_eachi (_T, T) {
//   bcast = _mm512_broadcastss_ps(*(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0));
//   for_each (_O, O) {
//     zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T));
//   }
// }

#define P2_COMPUTE_OUTPUT_0_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T));
#define P2_COMPUTE_OUTPUT_0_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_0(_T);                                                  \
  BOOST_PP_REPEAT(_O, P2_COMPUTE_OUTPUT_0_T_O, _T);
#define P2_COMPUTE_OUTPUT_0(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P2_COMPUTE_OUTPUT_0_T, _O);

#define P2_COMPUTE_OUTPUT_1_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 1), bcast, zmm_out(_O, _T));
#define P2_COMPUTE_OUTPUT_1_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_1(_T);                                                  \
  BOOST_PP_REPEAT(_O, P2_COMPUTE_OUTPUT_1_T_O, _T);
#define P2_COMPUTE_OUTPUT_1(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P2_COMPUTE_OUTPUT_1_T, _O);

/*
#define P2_COMPUTE_OUTPUT_0(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T))
#define P2_COMPUTE_OUTPUT_1(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 1));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 1), bcast, zmm_out(_O, _T))
*/

// Pipeline = 4
#define P4_PRELOAD_WEIGHTS_0_1(_O, _P)                                         \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, 0, 0, 0, 0));       \
  zmm_wei(_O, 1) = _mm512_load_ps(&md6(aweights, _O, _ic3, 0, 0, 1, 0))
#define P4_LOAD_WEIGHTS_2(_O, _P)                                              \
  zmm_wei(_O, 2) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V, 2, 0));
#define P4_LOAD_WEIGHTS_3(_O, _P)                                              \
  zmm_wei(_O, 3) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V, 3, 0));
#define P4_LOAD_WEIGHTS_0(_O, _P)                                              \
  zmm_wei(_O, 0) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V + 1, 0, 0));
#define P4_LOAD_WEIGHTS_1(_O, _P)                                              \
  zmm_wei(_O, 1) = _mm512_load_ps(&md6(aweights, _O, _ic3, _I2, _V + 1, 1, 0));

#define P4_COMPUTE_OUTPUT_0_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T));
#define P4_COMPUTE_OUTPUT_0_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_0(_T);                                                  \
  BOOST_PP_REPEAT(_O, P4_COMPUTE_OUTPUT_0_T_O, _T);
#define P4_COMPUTE_OUTPUT_0(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P4_COMPUTE_OUTPUT_0_T, _O);

#define P4_COMPUTE_OUTPUT_1_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 1), bcast, zmm_out(_O, _T));
#define P4_COMPUTE_OUTPUT_1_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_1(_T);                                                  \
  BOOST_PP_REPEAT(_O, P4_COMPUTE_OUTPUT_1_T_O, _T);
#define P4_COMPUTE_OUTPUT_1(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P4_COMPUTE_OUTPUT_1_T, _O);

#define P4_COMPUTE_OUTPUT_2_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 2), bcast, zmm_out(_O, _T));
#define P4_COMPUTE_OUTPUT_2_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_2(_T);                                                  \
  BOOST_PP_REPEAT(_O, P4_COMPUTE_OUTPUT_2_T_O, _T);
#define P4_COMPUTE_OUTPUT_2(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P4_COMPUTE_OUTPUT_2_T, _O);

#define P4_COMPUTE_OUTPUT_3_T_O(z, _O, _T)                                     \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 3), bcast, zmm_out(_O, _T));
#define P4_COMPUTE_OUTPUT_3_T(z, _T, _O)                                       \
  bcast = P_LOAD_INPUT_3(_T);                                                  \
  BOOST_PP_REPEAT(_O, P4_COMPUTE_OUTPUT_3_T_O, _T);
#define P4_COMPUTE_OUTPUT_3(_O, _T)                                            \
  BOOST_PP_REPEAT(_T, P4_COMPUTE_OUTPUT_3_T, _O);

/*
#define P4_COMPUTE_OUTPUT_0(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 0));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 0), bcast, zmm_out(_O, _T))

#define P4_COMPUTE_OUTPUT_1(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 1));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 1), bcast, zmm_out(_O, _T))
#define P4_COMPUTE_OUTPUT_2(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 2));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 2), bcast, zmm_out(_O, _T))
#define P4_COMPUTE_OUTPUT_3(_O, _T)                                            \
  bcast = _mm512_broadcastss_ps(                                               \
      *(__m128 *)&md6(ainput, _ic3, _I2, 0, _T, _V, 3));                       \
  zmm_out(_O, _T) = _mm512_fmadd_ps(zmm_wei(_O, 3), bcast, zmm_out(_O, _T))
*/

// P = 1
#define CONV1X1_GEMM_P1(O, T)                                                  \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O, T, V, I, TR(false), JAM(false),  \
      with_bias, with_relu, with_sum>::gemm(elx_conv_t<Type> &xc,              \
      Type *output, Type *input, Type *weights, Type *bias, bool)              \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD4(Type, aoutput, output, O, xc.t2, T, V);                                \
    MD5(Type, ainput, input, xc.ic3, xc.I2, xc.t2, T, V);                      \
    MD5(Type, aweights, weights, O, xc.ic3, xc.I2, V, V);                      \
    MD2(Type, abias, bias, O, V);                                              \
                                                                               \
    for_each (_ic3, xc.ic3) {                                                  \
      __m512 bcast;                                                            \
      MATRIX_OP(DEF_OUTPUT, O, T);                                             \
      MATRIX_OP(DEF_WEIGHTS, O, 1);                                            \
      if (_ic3 == 0) {                                                         \
        if (with_bias) {                                                       \
          MATRIX_OP(LOAD_BIAS, O, T);                                          \
        } else {                                                               \
          MATRIX_OP(CLEAR_OUTPUT, O, T);                                       \
        }                                                                      \
      } else {                                                                 \
        MATRIX_OP(LOAD_OUTPUT, O, T);                                          \
      }                                                                        \
      for_each (_I2, xc.I2) {                                                  \
        pragma_unroll for_each (_V, V)                                         \
        {                                                                      \
          MATRIX_OP(LOAD_WEIGHTS, O, 1);                                       \
          COMPUTE_OUTPUT(O, T);                                                \
        }                                                                      \
      }                                                                        \
      MATRIX_OP(STORE_OUTPUT, O, T);                                           \
    }                                                                          \
  }                                                                            \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(false), RELU(false),                         \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);                                                          \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(true), RELU(false),                          \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);

// P = 2
#define CONV1X1_GEMM_P2(O, T)                                                  \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O, T, V, I, TR(false), JAM(false),  \
      with_bias, with_relu, with_sum>::gemm(elx_conv_t<Type> &xc,              \
      Type *output, Type *input, Type *weights, Type *bias, bool)              \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD4(Type, aoutput, output, O, xc.t2, T, V);                                \
    MD6(Type, ainput, input, xc.ic3, xc.I2, xc.t2, T, V / 2, 2);               \
    MD6(Type, aweights, weights, O, xc.ic3, xc.I2, V / 2, 2, V);               \
    MD2(Type, abias, bias, O, V);                                              \
                                                                               \
    for_each (_ic3, xc.ic3) {                                                  \
      __m512 bcast;                                                            \
      MATRIX_OP(DEF_OUTPUT, O, T);                                             \
      MATRIX_OP(DEF_WEIGHTS, O, 2);                                            \
      MATRIX_OP(P2_PRELOAD_WEIGHTS_0, O, 1);                                   \
      if (_ic3 == 0) {                                                         \
        if (with_bias) {                                                       \
          MATRIX_OP(LOAD_BIAS, O, T);                                          \
        } else {                                                               \
          MATRIX_OP(CLEAR_OUTPUT, O, T);                                       \
        }                                                                      \
      } else {                                                                 \
        MATRIX_OP(LOAD_OUTPUT, O, T);                                          \
      }                                                                        \
      for_each (_I2, xc.I2) {                                                  \
        pragma_unroll for_each (_V, V / 2)                                     \
        {                                                                      \
          MATRIX_OP(P2_LOAD_WEIGHTS_1, O, 1);                                  \
          P2_COMPUTE_OUTPUT_0(O, T);                                           \
                                                                               \
          MATRIX_OP(P2_LOAD_WEIGHTS_0, O, 1);                                  \
          P2_COMPUTE_OUTPUT_1(O, T);                                           \
        }                                                                      \
      }                                                                        \
      MATRIX_OP(STORE_OUTPUT, O, T);                                           \
    }                                                                          \
  }                                                                            \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(false), RELU(false),                         \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);                                                          \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(true), RELU(false),                          \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);

// P = 4
#define CONV1X1_GEMM_P4(O, T)                                                  \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O, T, V, I, TR(false), JAM(false),  \
      with_bias, with_relu, with_sum>::gemm(elx_conv_t<Type> &xc,              \
      Type *output, Type *input, Type *weights, Type *bias, bool)              \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD4(Type, aoutput, output, O, xc.t2, T, V);                                \
    MD6(Type, ainput, input, xc.ic3, xc.I2, xc.t2, T, V / 4, 4);               \
    MD6(Type, aweights, weights, O, xc.ic3, xc.I2, V / 4, 4, V);               \
    MD2(Type, abias, bias, O, V);                                              \
                                                                               \
    for_each (_ic3, xc.ic3) {                                                  \
      __m512 bcast;                                                            \
      MATRIX_OP(DEF_OUTPUT, O, T);                                             \
      MATRIX_OP(DEF_WEIGHTS, O, 4);                                            \
      MATRIX_OP(P4_PRELOAD_WEIGHTS_0_1, O, 1);                                 \
      if (_ic3 == 0) {                                                         \
        if (with_bias) {                                                       \
          MATRIX_OP(LOAD_BIAS, O, T);                                          \
        } else {                                                               \
          MATRIX_OP(CLEAR_OUTPUT, O, T);                                       \
        }                                                                      \
      } else {                                                                 \
        MATRIX_OP(LOAD_OUTPUT, O, T);                                          \
      }                                                                        \
      for_each (_I2, xc.I2) {                                                  \
        pragma_unroll for_each (_V, V / 4)                                     \
        {                                                                      \
          MATRIX_OP(P4_LOAD_WEIGHTS_2, O, 1);                                  \
          P4_COMPUTE_OUTPUT_0(O, T);                                           \
                                                                               \
          MATRIX_OP(P4_LOAD_WEIGHTS_3, O, 1);                                  \
          P4_COMPUTE_OUTPUT_1(O, T);                                           \
                                                                               \
          MATRIX_OP(P4_LOAD_WEIGHTS_0, O, 1);                                  \
          P4_COMPUTE_OUTPUT_2(O, T);                                           \
                                                                               \
          MATRIX_OP(P4_LOAD_WEIGHTS_1, O, 1);                                  \
          P4_COMPUTE_OUTPUT_3(O, T);                                           \
        }                                                                      \
      }                                                                        \
      MATRIX_OP(STORE_OUTPUT, O, T);                                           \
    }                                                                          \
  }                                                                            \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(false), RELU(false),                         \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);                                                          \
  template void convolution_direct_1x1_kernel<float, O, T, 16, ISA_SKX_AVX512, \
      TR(false), JAM(false), BIAS(true), RELU(false),                          \
      SUM(false)>::gemm(elx_conv_t<float> &, float *, float *, float *,        \
      float *, bool);

// O=1, T:
//    T=31..:  kernel: 1, output: 31..
//    T=29,30: kernel: 2, output: 29 - 30 (pipeline: 2)
//    T=1..28: kenrel: 4, output: 1 - 28  (pipeline: 4)
CONV1X1_GEMM_P4(1, 1);
CONV1X1_GEMM_P4(1, 2);
CONV1X1_GEMM_P4(1, 3);
CONV1X1_GEMM_P4(1, 4);
CONV1X1_GEMM_P4(1, 5);
CONV1X1_GEMM_P4(1, 6);
CONV1X1_GEMM_P4(1, 7);
CONV1X1_GEMM_P4(1, 8);
CONV1X1_GEMM_P4(1, 9);
CONV1X1_GEMM_P4(1, 10);
CONV1X1_GEMM_P4(1, 11);
CONV1X1_GEMM_P4(1, 12);
CONV1X1_GEMM_P4(1, 13);
CONV1X1_GEMM_P4(1, 14);
CONV1X1_GEMM_P4(1, 15);
CONV1X1_GEMM_P4(1, 16);
CONV1X1_GEMM_P4(1, 17);
CONV1X1_GEMM_P4(1, 18);
CONV1X1_GEMM_P4(1, 19);
CONV1X1_GEMM_P4(1, 20);
CONV1X1_GEMM_P4(1, 21);
CONV1X1_GEMM_P4(1, 22);
CONV1X1_GEMM_P4(1, 23);
CONV1X1_GEMM_P4(1, 24);
CONV1X1_GEMM_P4(1, 25);
CONV1X1_GEMM_P4(1, 26);
CONV1X1_GEMM_P4(1, 27);
CONV1X1_GEMM_P4(1, 28);
CONV1X1_GEMM_P2(1, 29);
CONV1X1_GEMM_P2(1, 30);
CONV1X1_GEMM_P1(1, 31);
CONV1X1_GEMM_P1(1, 32);
CONV1X1_GEMM_P1(1, 33);
CONV1X1_GEMM_P1(1, 34);
CONV1X1_GEMM_P1(1, 35);

// O=2, T:
//    T=14:    bcast: 1, kernel: 2, output: 28
//    T=12,13: bcast: 1, kernel: 4, output: 24,26 (pipeline: 2)
//    T=1..11: bcast: 1, kernel: 8, output: 2..22 (pipeline: 4)
CONV1X1_GEMM_P4(2, 1);
CONV1X1_GEMM_P4(2, 2);
CONV1X1_GEMM_P4(2, 3);
CONV1X1_GEMM_P4(2, 4);
CONV1X1_GEMM_P4(2, 5);
CONV1X1_GEMM_P4(2, 6);
CONV1X1_GEMM_P4(2, 7);
CONV1X1_GEMM_P4(2, 8);
CONV1X1_GEMM_P4(2, 9);
CONV1X1_GEMM_P4(2, 10);
CONV1X1_GEMM_P4(2, 11);
CONV1X1_GEMM_P2(2, 12);
CONV1X1_GEMM_P2(2, 13);
CONV1X1_GEMM_P1(2, 14);

// O=3, T:
//    T=8:     bcast: 1, kernel 3, output: 24
//    T=7:     bcast: 1, kernel 6, output: 21 (pipeline: 2)
//    T=1..6:  bcast: 1, kernel 12, output: 3..18 (pipeline: 4)
CONV1X1_GEMM_P4(3, 1);
CONV1X1_GEMM_P4(3, 2);
CONV1X1_GEMM_P4(3, 3);
CONV1X1_GEMM_P4(3, 4);
CONV1X1_GEMM_P4(3, 5);
CONV1X1_GEMM_P4(3, 6);
CONV1X1_GEMM_P2(3, 7);
CONV1X1_GEMM_P1(3, 8);
CONV1X1_GEMM_P1(3, 9); // spill
CONV1X1_GEMM_P1(3, 10);
CONV1X1_GEMM_P1(3, 11);
CONV1X1_GEMM_P1(3, 12);
CONV1X1_GEMM_P1(3, 13);
CONV1X1_GEMM_P1(3, 14);

// O=4, T:
//    T=6:     bcast: 1, kernel: 4, outupt: 24
//    T=4..5:  bcast: 1, kernel: 8, outupt: 16..20 (pipeline: 2)
//    T=1..3:  bcast: 1, kernel: 16, outupt: 4..12 (pipeline: 4)
CONV1X1_GEMM_P4(4, 1);
CONV1X1_GEMM_P4(4, 2);
CONV1X1_GEMM_P4(4, 3);
CONV1X1_GEMM_P2(4, 4);
CONV1X1_GEMM_P2(4, 5);
CONV1X1_GEMM_P1(4, 6);
CONV1X1_GEMM_P1(4, 7);
// spill
CONV1X1_GEMM_P1(4, 8);
CONV1X1_GEMM_P1(4, 9);
CONV1X1_GEMM_P1(4, 10);
CONV1X1_GEMM_P1(4, 11);
CONV1X1_GEMM_P1(4, 12);
CONV1X1_GEMM_P1(4, 13);
CONV1X1_GEMM_P1(4, 14);

// O=5, T:
//    T=5:     bcast: 1, kernel: 5, output: 25
//    T=3,4:   bcast: 1, kernel: 10, output: 15,20 (pipeline: 2)
//    T=1,2:   bcast: 1, kernel: 20, output: 5,10 (pipeline: 2)
CONV1X1_GEMM_P4(5, 1);
CONV1X1_GEMM_P4(5, 2);
CONV1X1_GEMM_P2(5, 3);
CONV1X1_GEMM_P2(5, 4);
CONV1X1_GEMM_P1(5, 5);

// O=6, T:
//    T=4:     bcast: 1, kenrel: 6, output: 24
//    T=2,3:   bcast: 1, kernel: 12, output: 12,18 (pipeline: 2)
//    T=1:     bcast: 1, kernel: 24, output: 6 (pipeline: 4)
CONV1X1_GEMM_P4(6, 1);
CONV1X1_GEMM_P2(6, 2);
CONV1X1_GEMM_P2(6, 3);
CONV1X1_GEMM_P1(6, 4);

// O=7, T:
//    T=3:     bcast: 1, kernel: 7, output: 21
//    T=1,2:   bcast: 1, kernel: 14, output: 7,14 (pipeline: 2)
CONV1X1_GEMM_P2(7, 1);
CONV1X1_GEMM_P2(7, 2);
CONV1X1_GEMM_P1(7, 3);

// O=8, T:
//    T=2:     bcast: 1, kernel: 8, output: 16
//    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)
CONV1X1_GEMM_P2(8, 1);
CONV1X1_GEMM_P1(8, 2);
// spill
CONV1X1_GEMM_P1(8, 3);
CONV1X1_GEMM_P1(8, 4);
CONV1X1_GEMM_P1(8, 5);
CONV1X1_GEMM_P1(8, 6);
CONV1X1_GEMM_P1(8, 7);
CONV1X1_GEMM_P1(8, 8);

}
