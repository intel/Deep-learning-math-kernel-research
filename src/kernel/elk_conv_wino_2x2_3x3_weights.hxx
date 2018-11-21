#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_2x2_3x3_input.hxx"

namespace euler {

// float atweights[A][A][V][V] <- float aweights[K][K][V][V])
template <typename InputType, typename WeightsType,
     typename OutputType, typename BiasType, typename TarrayType, int V>
 inline void convolution_winograd_kernel_base<InputType, WeightsType, OutputType,
     BiasType, TarrayType, ISA_SKX_AVX512, V, 4, 3>::
__trans_weights(TarrayType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]) {
  ENABLE_AVX512F();

  // Constants
  __m<V> r4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m<V> r2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));

  // Inputs
  __m<V> f00, f10, f20, f01, f11, f21, f02, f12, f22;
  // Cache
  __m<V> c10, c11, c12, c20, c21, c22;
  // Outputs
  __m<V> /* t00, */t10, t20, /* t30, */t01, t11, t21, t31, t02, t12, t22,
      t32, /* t03, */t13, t23/*, t33 */;
#undef F
#undef T
#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n)                                                   \
  if(std::is_same<WeightsType, float>::value)                     \
    f(m,n) = _mm<V>::load_ps(F(m, n));                            \
  else {                                                          \
    auto f16 = _mm<V>::load_si256((__m256i *)F(m, n));            \
    f(m,n) = _mm<V>::cvtph_ps(f16);                               \
  }
  for (int _V = 0; _V < 16; ++_V) {
    MATRIX_DEF(3, 3);

    _mm<V>::store_ps(T(0, 0), f00);
    t10 = MUL(r2, ADD(SUB(f00, f10), f20));
    _mm<V>::store_ps(T(1, 0), t10);
    t20 = MUL(r2, ADD(ADD(f00, f10), f20));
    _mm<V>::store_ps(T(2, 0), t20);
    _mm<V>::store_ps(T(3, 0), f20);

    c10 = MUL(r4, ADD(SUB(f00, f01), f02));
    c11 = MUL(r4, ADD(SUB(f10, f11), f12));
    c12 = MUL(r4, ADD(SUB(f20, f21), f22));
    t01 = MUL(z2, c10);
    _mm<V>::store_ps(T(0, 1), t01);
    t11 = ADD(SUB(c10, c11), c12);
    _mm<V>::store_ps(T(1, 1), t11);
    t21 = ADD(ADD(c10, c11), c12);
    _mm<V>::store_ps(T(2, 1), t21);
    t31 = MUL(z2, c12);
    _mm<V>::store_ps(T(3, 1), t31);

    c20 = MUL(r4, ADD(ADD(f00, f01), f02));
    c21 = MUL(r4, ADD(ADD(f10, f11), f12));
    c22 = MUL(r4, ADD(ADD(f20, f21), f22));
    t02 = MUL(z2, c20);
    _mm<V>::store_ps(T(0, 2), t02);
    t12 = ADD(SUB(c20, c21), c22);
    _mm<V>::store_ps(T(1, 2), t12);
    t22 = ADD(ADD(c20, c21), c22);
    _mm<V>::store_ps(T(2, 2), t22);
    t32 = MUL(z2, c22);
    _mm<V>::store_ps(T(3, 2), t32);

    _mm<V>::store_ps(T(0, 3), f02);
    t13 = MUL(r2, ADD(SUB(f02, f12), f22));
    _mm<V>::store_ps(T(1, 3), t13);
    t23 = MUL(r2, ADD(ADD(f02, f12), f22));
    _mm<V>::store_ps(T(2, 3), t23);
    _mm<V>::store_ps(T(3, 3), f22);
  }
}


} // namespace euler
