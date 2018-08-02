#include <assert.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_direct_1x1.hpp"

#include <x86intrin.h>

namespace euler {

#define AVX512_ZERO(z, n, nil) __m512 t##n = _mm512_setzero_ps();
#define AVX512_FMA(z, n, nil)                                                  \
  x = _mm512_set1_ps(md5(ainput, _ic3, _I2, 0, n, _V));                        \
  t##n = _mm512_fmadd_ps(w, x, t##n);
#define AVX512_ADD(z, n, nil)                                                  \
  t##n = _mm512_add_ps(t##n, *(__m512 *)&md4(aoutput, _O2, 0, n, 0));
#define AVX512_STORE(z, n, nil)                                                \
  _mm512_store_ps(&md4(aoutput, _O2, 0, n, 0), t##n);

template <typename Type, const int V, const int I,
    const bool with_bias, const bool with_relu, const bool with_sum>
void convolution_direct_1x1_kernel::gemm28(
    elx_conv_t<Type> &xc, Type *output, Type *input, Type *weights, Type *blias)
{
  ENABLE_AVX512F();

  MD4(Type, aoutput, output, xc.O2, xc.t2, 28, V);
  MD5(Type, ainput, input, xc.ic3, xc.I2, xc.t2, 28, V);
  MD5(Type, aweights, weights, xc.O2, xc.ic3, xc.I2, V, V);

  for_each (_ic3, xc.ic3) {
    for_each (_O2, xc.O2) {
      BOOST_PP_REPEAT(28, AVX512_ZERO, nil);

      for_each (_I2, xc.I2) {
        for_each (_V, V) {
          __m512 x;
          __m512 w = _mm512_load_ps(&md5(aweights, _O2, _ic3, _I2, _V, 0));
          BOOST_PP_REPEAT(28, AVX512_FMA, nil);
        }
      }
      if (_ic3 > 0) {
        BOOST_PP_REPEAT(28, AVX512_ADD, nil);
      }
      BOOST_PP_REPEAT(28, AVX512_STORE, nil);
    }
  }
}

template void convolution_direct_1x1_kernel::gemm28<float, 16, ISA_SKX_AVX512,
    false, false, false>(
    elx_conv_t<float> &, float *, float *, float *, float *);
}
