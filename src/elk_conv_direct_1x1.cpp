#include <assert.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

#include <x86intrin.h>

namespace euler {

#define AVX512_DEF(z, n, nil) __m512 t##n;
#define AVX512_ZERO(z, n, nil) t##n = _mm512_setzero_ps();
#define AVX512_LOAD(z, n, nil)                                                 \
  t##n = _mm512_load_ps(&md4(aoutput, _O2, 0, n, 0));
#define AVX512_LOAD_BIAS(z, n, nil) t##n = _mm512_load_ps(&md2(abias, _O2, 0));
#define AVX512_FMA(z, n, nil)                                                  \
  x = _mm512_set1_ps(md5(ainput, _ic3, _I2, 0, n, _V));                        \
  t##n = _mm512_fmadd_ps(w, x, t##n);
#define AVX512_STORE(z, n, nil)                                                \
  _mm512_store_ps(&md4(aoutput, _O2, 0, n, 0), t##n);

#define DEF_function_gemm31(z, T, nil)                                         \
  template <typename Type, const int V, const int I, const bool with_bias,     \
      const bool with_relu, const bool with_sum>                               \
  void convolution_direct_1x1_kernel::gemm##T(elx_conv_t<Type> &xc,            \
      Type *output, Type *input, Type *weights, Type *bias)                    \
  {                                                                            \
    ENABLE_AVX512F();                                                          \
                                                                               \
    MD4(Type, aoutput, output, xc.O2, xc.t2, T, V);                            \
    MD5(Type, ainput, input, xc.ic3, xc.I2, xc.t2, T, V);                      \
    MD5(Type, aweights, weights, xc.O2, xc.ic3, xc.I2, V, V);                  \
    MD2(Type, abias, bias, xc.O2, V);                                          \
                                                                               \
    for_each (_ic3, xc.ic3) {                                                  \
      for_each (_O2, xc.O2) {                                                  \
        BOOST_PP_REPEAT(T, AVX512_DEF, nil);                                   \
        if (_ic3 == 0) {                                                       \
          if (with_bias) {                                                     \
            BOOST_PP_REPEAT(T, AVX512_LOAD_BIAS, nil);                         \
          } else {                                                             \
            BOOST_PP_REPEAT(T, AVX512_ZERO, nil);                              \
          }                                                                    \
        } else {                                                               \
          BOOST_PP_REPEAT(T, AVX512_LOAD, nil);                                \
        }                                                                      \
        for_each (_I2, xc.I2) {                                                \
          for_each (_V, V) {                                                   \
            __m512 x;                                                          \
            __m512 w = _mm512_load_ps(&md5(aweights, _O2, _ic3, _I2, _V, 0));  \
            BOOST_PP_REPEAT(T, AVX512_FMA, nil);                               \
          }                                                                    \
        }                                                                      \
        BOOST_PP_REPEAT(T, AVX512_STORE, nil);                                 \
      }                                                                        \
    }                                                                          \
  }
BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, DEF_function_gemm31, nil);

#define INST_V_gemm(z, T, nil)                                                 \
  template void convolution_direct_1x1_kernel::gemm##T<float, 16,              \
      ISA_SKX_AVX512, BIAS(false), RELU(false), SUM(false)>(                   \
      elx_conv_t<float> &, float *, float *, float *, float *);                \
  template void convolution_direct_1x1_kernel::gemm##T<float, 16,              \
      ISA_SKX_AVX512, BIAS(true), RELU(false), SUM(false)>(                    \
      elx_conv_t<float> &, float *, float *, float *, float *);

BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, INST_V_gemm, nil);

}
