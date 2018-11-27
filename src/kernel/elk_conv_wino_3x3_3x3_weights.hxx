#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_3x3_3x3_input.hxx"

namespace euler {

template <typename UserTypes, typename TrOpType, int V>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_weights(TrOpType atweights[A][A][V][V],
    WeightsType aweights[K][K][V][V])
{
  // Constants
  __m<V> r12 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 12.0f));
  __m<V> r6 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 6.0f));
  __m<V> r4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m<V> r_4 = _mm<V>::set_ps(IMM_BCAST16(-1.0f / 4.0f));
  __m<V> r3 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 3.0f));
  __m<V> r2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m<V> r_2 = _mm<V>::set_ps(IMM_BCAST16(-1.0f / 2.0f));
  __m<V> r2_3 = _mm<V>::set_ps(IMM_BCAST16(2.0f / 3.0f));

  // Inputs
  __m<V> f00, f10, f20, f01, f11, f21, f02, f12, f22;
  // Cache
  __m<V> c10, c11, c12, c20, c21, c22, c30, c31, c32;
  // Outputs
  __m<V> t00, t10, t20, t30, t40, t01, t11, t21, t31 /*, t41 */, t02, t12, t22,
      t32 /*, t42 */, t03, t13, t23, t33 /*, t43 */, t04, t14, t24,
      t34 /*, t44 */;
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
  for (int _V = 0; _V < V; ++_V) {
    MATRIX_DEF(3, 3);

    c10 = MUL(r6, SUB(f01, ADD(f00, f02)));
    c11 = MUL(r6, SUB(f11, ADD(f10, f12)));
    c12 = MUL(r6, SUB(f21, ADD(f20, f22)));
    c20 = MUL(r_2, ADD(ADD(f00, f01), f02));
    c21 = MUL(r_2, ADD(ADD(f10, f11), f12));
    c22 = MUL(r_2, ADD(ADD(f20, f21), f22));

    t00 = MUL(r4, f00);
    _mm<V>::store_ps(T(0, 0), t00);
    t10 = MUL(r12, SUB(f10, ADD(f00, f20)));
    _mm<V>::store_ps(T(1, 0), t10);
    t20 = MUL(r_4, ADD(f10, ADD(f00, f20)));
    _mm<V>::store_ps(T(2, 0), t20);
    t30 = FMADD(r12, f00, FMADD(r6, f10, MUL(r3, f20)));
    _mm<V>::store_ps(T(3, 0), t30);
    t40 = MUL(r2, f20);
    _mm<V>::store_ps(T(4, 0), t40);

    t01 = MUL(r2, c10);
    _mm<V>::store_ps(T(0, 1), t01);
    t11 = MUL(r6, SUB(c11, ADD(c10, c12)));
    _mm<V>::store_ps(T(1, 1), t11);
    t21 = MUL(r_2, ADD(c11, ADD(c10, c12)));
    _mm<V>::store_ps(T(2, 1), t21);
    t31 = FMADD(r6, c10, FMADD(r3, c11, MUL(r2_3, c12)));
    _mm<V>::store_ps(T(3, 1), t31);
    _mm<V>::store_ps(T(4, 1), c12);

    t02 = MUL(r2, c20);
    _mm<V>::store_ps(T(0, 2), t02);
    t12 = MUL(r6, SUB(c21, ADD(c20, c22)));
    _mm<V>::store_ps(T(1, 2), t12);
    t22 = MUL(r_2, ADD(c21, ADD(c20, c22)));
    _mm<V>::store_ps(T(2, 2), t22);
    t32 = FMADD(r6, c20, FMADD(r3, c21, MUL(r2_3, c22)));
    _mm<V>::store_ps(T(3, 2), t32);
    _mm<V>::store_ps(T(4, 2), c22);

    c30 = FMADD(r6, f00, FMADD(r3, f01, MUL(r2_3, f02)));
    c31 = FMADD(r6, f10, FMADD(r3, f11, MUL(r2_3, f12)));
    c32 = FMADD(r6, f20, FMADD(r3, f21, MUL(r2_3, f22)));

    t03 = MUL(r2, c30);
    _mm<V>::store_ps(T(0, 3), t03);
    t13 = MUL(r6, SUB(c31, ADD(c30, c32)));
    _mm<V>::store_ps(T(1, 3), t13);
    t23 = MUL(r_2, ADD(c31, ADD(c30, c32)));
    _mm<V>::store_ps(T(2, 3), t23);
    t33 = FMADD(r6, c30, FMADD(r3, c31, MUL(r2_3, c32)));
    _mm<V>::store_ps(T(3, 3), t33);
    _mm<V>::store_ps(T(4, 3), c32);

    t04 = MUL(r2, f02);
    _mm<V>::store_ps(T(0, 4), t04);
    t14 = MUL(r6, SUB(f12, ADD(f02, f22)));
    _mm<V>::store_ps(T(1, 4), t14);
    t24 = MUL(r_2, ADD(f12, ADD(f02, f22)));
    _mm<V>::store_ps(T(2, 4), t24);
    t34 = FMADD(r6, f02, FMADD(r3, f12, MUL(r2_3, f22)));
    _mm<V>::store_ps(T(3, 4), t34);
    _mm<V>::store_ps(T(4, 4), f22);
  }
}
} // namespace euler
