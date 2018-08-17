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
#undef DEF_OUTPUT
#undef DEF_WEIGHTS
#undef K_GEMM_FMA_P1
#undef K_GEMM_FMA_P2
#undef K_GEMM_FMA_P4
#undef K1
#undef K2
#undef K3

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
    MD2(float, aoutput2, &md2(aoutput, Os_, 0), O_, xc.oh *xc.ow *V);          \
    MD5(float, aweights5, &md3(aweights, Os_, 0, 0), O_, xc.ic34,              \
        xc.I2, V, V);                                                          \
    MD2(float, abias2, &md2(abias, Os_, 0), O_, V);                            \
                                                                               \
    if (reset_out) {                                                           \
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
      MD3(float, ainput3, &md2(ainput, _I2, 0), xc.t2, T_, V);                 \
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
    MD2(float, aoutput2, &md2(aoutput, Os_, 0), O_, xc.oh *xc.ow *V);          \
    MD6(float, aweights6, &md3(aweights, Os_, 0, 0), O_, xc.ic34,              \
        xc.I2, V / 2, 2, V);                                                   \
    MD2(float, abias2, &md2(abias, Os_, 0), O_, V);                            \
                                                                               \
    MM_PRELOAD_WEIGHTS_P2(O_);                                                 \
                                                                               \
    if (reset_out) {                                                           \
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
      MD4(Type, ainput4, &md2(ainput, _I2, 0), xc.t2, T_, V / 2, 2);           \
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
    MD2(float, aoutput2, &md2(aoutput, Os_, 0), O_, xc.oh *xc.ow *V);          \
    MD6(float, aweights6, &md3(aweights, Os_, 0, 0), O_, xc.ic34,              \
        xc.I2, V / 4, 4, V);                                                   \
    MD2(float, abias2, &md2(abias, Os_, 0), O_, V);                            \
                                                                               \
    MM_PRELOAD_WEIGHTS_P4(O_);                                                 \
                                                                               \
    if (reset_out) {                                                           \
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
      MD4(Type, ainput4, &md2(ainput, _I2, 0), xc.t2, T_, V / 4, 4);           \
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

#define K1(O2_, T_, P_)                                                        \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O2_, T_, V, I, with_bias,           \
      with_relu, with_sum>::gemm(elx_conv_t<Type> &xc, Type *output,           \
      Type *input, Type *weights, Type *bias, bool reset_out)                  \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD2(Type, aoutput, output, O2_, xc.oh *xc.ow *V);                          \
    MD2(Type, ainput, input, xc.I2, xc.ih *xc.iw *V);                          \
    MD3(Type, aweights, weights, O2_, xc.ic34, xc.I2 *V *V);                   \
    MD2(Type, abias, bias, O2_, V);                                            \
                                                                               \
    K_GEMM_FMA_P##P_(O2_, T_, 0);                                              \
  }                                                                            \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *, \
      float *, float *, float *, bool);                                        \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,         \
      float *, float *, float *, float *, bool);

#define K2(O2_, T_, o0_, o1_, P0_, P1_)                                        \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O2_, T_, V, I, with_bias,           \
      with_relu, with_sum>::gemm(elx_conv_t<Type> &xc, Type *output,           \
      Type *input, Type *weights, Type *bias, bool reset_out)                  \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD2(Type, aoutput, output, O2_, xc.oh *xc.ow *V);                          \
    MD2(Type, ainput, input, xc.I2, xc.ih *xc.iw *V);                          \
    MD3(Type, aweights, weights, O2_, xc.ic34, xc.I2 *V *V);                   \
    MD2(Type, abias, bias, O2_, V);                                            \
                                                                               \
    K_GEMM_FMA_P##P0_(o0_, T_, 0);                                             \
    K_GEMM_FMA_P##P1_(o1_, T_, o0_);                                           \
  }                                                                            \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *, \
      float *, float *, float *, bool);                                        \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,         \
      float *, float *, float *, float *, bool);

#define K3(O2_, T_, o0_, o1_, o2_, P0_, P1_, P2_)                              \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel<Type, O2_, T_, V, I, with_bias,           \
      with_relu, with_sum>::gemm(elx_conv_t<Type> &xc, Type *output,           \
      Type *input, Type *weights, Type *bias, bool reset_out)                  \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD2(Type, aoutput, output, O2_, xc.oh *xc.ow *V);                          \
    MD2(Type, ainput, input, xc.I2, xc.ih *xc.iw *V);                          \
    MD3(Type, aweights, weights, O2_, xc.ic34, xc.I2 *V *V);                   \
    MD2(Type, abias, bias, O2_, V);                                            \
                                                                               \
    K_GEMM_FMA_P##P0_(o0_, T_, 0);                                             \
    K_GEMM_FMA_P##P1_(o1_, T_, o0_);                                           \
    K_GEMM_FMA_P##P2_(o2_, T_, o0_ + o1_);                                     \
  }                                                                            \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(true), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &, float *, \
      float *, float *, float *, bool);                                        \
  template void                                                                \
  convolution_direct_1x1_kernel<float, O2_, T_, 16, ISA_SKX_AVX512,            \
      BIAS(false), RELU(false), SUM(false)>::gemm(elx_conv_t<float> &,         \
      float *, float *, float *, float *, bool);

// O=1, T:
//    T=31..:  kernel: 1, output: 31..
//    T=29,30: kernel: 2, output: 29 - 30 (pipeline: 2)
//    T=1..28: kenrel: 4, output: 1 - 28  (pipeline: 4)
K1(1, 1,  4)
K1(1, 2,  4)
K1(1, 3,  4)
K1(1, 4,  4)
K1(1, 5,  4)
K1(1, 6,  4)
K1(1, 7,  4)
K1(1, 8,  4)
K1(1, 9,  4)
K1(1, 10, 4)
K1(1, 11, 4)
K1(1, 12, 4)
K1(1, 13, 4)
K1(1, 14, 4)
K1(1, 15, 4)
K1(1, 16, 4)
K1(1, 17, 4)
K1(1, 18, 4)
K1(1, 19, 4)
K1(1, 20, 4)
K1(1, 21, 4)
K1(1, 22, 4)
K1(1, 23, 4)
K1(1, 24, 4)
K1(1, 25, 4)
K1(1, 26, 4)
K1(1, 27, 4)
K1(1, 28, 4)
K1(1, 29, 2)
K1(1, 30, 2)
K1(1, 31, 1)
K1(1, 32, 1)

// O=2, T:
//    T=14:    bcast: 1, kernel: 2, output: 28
//    T=12,13: bcast: 1, kernel: 4, output: 24,26 (pipeline: 2)
//    T=1..11: bcast: 1, kernel: 8, output: 2..22 (pipeline: 4)
K1(2, 1,  4)
K1(2, 2,  4)
K1(2, 3,  4)
K1(2, 4,  4)
K1(2, 5,  4)
K1(2, 6,  4)
K1(2, 7,  4)
K1(2, 8,  4)
K1(2, 9,  4)
K1(2, 10, 4)
K1(2, 11, 4)
K1(2, 12, 2)
K1(2, 13, 2)
K1(2, 14, 1)

// O=3, T:
//    T=8:     bcast: 1, kernel 3, output: 24
//    T=7:     bcast: 1, kernel 6, output: 21 (pipeline: 2)
//    T=1..6:  bcast: 1, kernel 12, output: 3..18 (pipeline: 4)
K1(3, 1, 4)
K1(3, 2, 4)
K1(3, 3, 4)
K1(3, 4, 4)
K1(3, 5, 4)
K1(3, 6, 4)
K1(3, 7, 2)
K1(3, 8, 1)
K2(3, 9, 2, 1, 4, 4)
K2(3, 10, 2, 1, 4, 4)
K2(3, 11, 2, 1, 4, 4)
K2(3, 12, 2, 1, 2, 4)
K2(3, 13, 2, 1, 2, 4)
K2(3, 14, 2, 1, 1, 4)

// O=4, T:
//    T=6:     bcast: 1, kernel: 4, outupt: 24
//    T=4..5:  bcast: 1, kernel: 8, outupt: 16..20 (pipeline: 2)
//    T=1..3:  bcast: 1, kernel: 16, output: 4..12 (pipeline: 4)
K1(4, 1, 4)
K1(4, 2, 4)
K1(4, 3, 4)
K1(4, 4, 2)
K1(4, 5, 2)
K1(4, 6, 1)
//K2(4, 7, 3, 1, 2, 4)
K2(4, 7, 2, 2, 4, 4)
K2(4, 8, 3, 1, 1, 4)
K2(4, 9, 2, 2, 4, 4)
K2(4, 10, 2, 2, 4, 4)
K2(4, 11, 2, 2, 4, 4)
K2(4, 12, 2, 2, 2, 2)
K2(4, 13, 2, 2, 2, 2)
K2(4, 14, 2, 2, 1, 1)

// O=5, T:
//    T=5:     bcast: 1, kernel: 5, output: 25
//    T=3,4:   bcast: 1, kernel: 10, output: 15,20 (pipeline: 2)
//    T=1,2:   bcast: 1, kernel: 20, output: 5,10 (pipeline: 4)
K1(5, 1, 4)
K1(5, 2, 4)
K1(5, 3, 2)
K1(5, 4, 2)
K1(5, 5, 1)

// O=6, T:
//    T=4:     bcast: 1, kenrel: 6, output: 24
//    T=2,3:   bcast: 1, kernel: 12, output: 12,18 (pipeline: 2)
//    T=1:     bcast: 1, kernel: 24, output: 6 (pipeline: 4)
K1(6, 1, 4)
K1(6, 2, 2)
K1(6, 3, 2)
K1(6, 4, 1)

// O=7, T:
//    T=3:     bcast: 1, kernel: 7, output: 21
//    T=1,2:   bcast: 1, kernel: 14, output: 7,14 (pipeline: 2)
K1(7, 1, 2)
K1(7, 2, 2)
K1(7, 3, 1)

// O=8, T:
//    T=2:     bcast: 1, kernel: 8, output: 16
//    T=1:     bcast: 1, kernel: 16, output: 8 (pipeline: 2)
K1(8, 1, 2)
K1(8, 2, 1)
K2(8, 3, 5, 3, 2, 4)
K2(8, 4, 5, 3, 2, 4)
K2(8, 5, 5, 3, 1, 4)
K2(8, 6, 4, 4, 1, 1)
K3(8, 7, 3, 3, 2, 2, 2, 4)
K3(8, 8, 3, 3, 2, 1, 1, 4)

}
