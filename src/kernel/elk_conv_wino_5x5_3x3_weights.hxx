#pragma once

#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

#define AVX512_CALCULATE_W_5_0(z, n, nil)                                      \
  c##n = (r4_81 * (f##n##0 + f##n##1 + f##n##2));
#define AVX512_CALCULATE_W_5_1(z, n, nil)                                      \
  c##n = (r4_81 * (f##n##0 - f##n##1 + f##n##2));
#define AVX512_CALCULATE_W_5_2(z, n, nil)                                      \
  c##n = (r2_405 * f##n##0 + (r4_405 * f##n##1 + (r8_405 * f##n##2)));
#define AVX512_CALCULATE_W_5_3(z, n, nil)                                      \
  c##n = (r4_405 * f##n##1 - (r2_405 * f##n##0 + (r8_405 * f##n##2)));
#define AVX512_CALCULATE_W_5_4(z, n, nil)                                      \
  c##n = (r32_405 * f##n##0 + (r16_405 * f##n##1 + (r8_405 * f##n##2)));
#define AVX512_CALCULATE_W_5_5(z, n, nil)                                      \
  c##n = (r16_405 * f##n##1 - (r32_405 * f##n##0 + (r8_405 * f##n##2)));

#define AVX512_CALCULATE_W_5(n)                                                \
  t0##n = (c0 + c1 + c2);                                                      \
  _mm<V>::store_ps(T(0, n), t0##n);                                            \
  t1##n = (c0 - c1 + c2);                                                      \
  _mm<V>::store_ps(T(1, n), t1##n);                                            \
  t2##n = (r1_10 * c0 + (r1_5 * c1 + (r2_5 * c2)));                            \
  _mm<V>::store_ps(T(2, n), t2##n);                                            \
  t3##n = (r1_5 * c1 - (r1_10 * c0 + (r2_5 * c2)));                            \
  _mm<V>::store_ps(T(3, n), t3##n);                                            \
  t4##n = (r8_5 * c0 + (r4_5 * c1 + (r2_5 * c2)));                             \
  _mm<V>::store_ps(T(4, n), t4##n);                                            \
  t5##n = (r4_5 * c1 - (r8_5 * c0 + (r2_5 * c2)));                             \
  _mm<V>::store_ps(T(5, n), t5##n);                                            \
  t6##n = (r9_2 * c2);                                                         \
  _mm<V>::store_ps(T(6, n), t6##n);

template <typename WeightsType, int V>
struct elk_conv_wino_trans_weights<float, WeightsType, ISA_SKX_AVX512,
    7, 3, V> {
  constexpr static int A = 7;
  constexpr static int K = 3;
  constexpr static int I = ISA_SKX_AVX512;

  static void execute(
      float atweights[A][A][V][V], WeightsType aweights[K][K][V][V])
  {
    ENABLE_AVX512F();

    __m<V> r1_5 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 5.0f));
    __m<V> r1_10 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 10.0f));
    __m<V> r2_5 = _mm<V>::set_ps(IMM_BCAST16(2.0f / 5.0f));
    __m<V> r4_5 = _mm<V>::set_ps(IMM_BCAST16(4.0f / 5.0f));
    __m<V> r8_5 = _mm<V>::set_ps(IMM_BCAST16(8.0f / 5.0f));
    __m<V> r4_81 = _mm<V>::set_ps(IMM_BCAST16(4.0f / 81.0f));
    __m<V> r9_2 = _mm<V>::set_ps(IMM_BCAST16(9.0f / 2.0f));

    __m<V> f00, f10, f20, f01, f11, f21, f02, f12, f22;
    // Cache
    __m<V> c0, c1, c2;
    // Outputs
    __m<V> t00, t01, t02, t03, t04, t05, t06, t10, t11, t12, t13, t14, t15, t16,
        t20, t21, t22, t23, t24, t25, t26, t30, t31, t32, t33, t34, t35, t36,
        t40, t41, t42, t43, t44, t45, t46, t50, t51, t52, t53, t54, t55, t56,
        t60, t61, t62, t63, t64, t65, t66;

#undef F
#undef T
#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[h][w][_V]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m, n)                                                               \
  if (std::is_same<WeightsType, float>::value)                                 \
    f(m, n) = _mm<V>::load_ps(F(m, n));                                        \
  else {                                                                       \
    auto f16 = _mm<V / 2>::load_si256((__m256i *)F(m, n));                     \
    f(m, n) = _mm<V>::cvtph_ps(f16);                                           \
  }

    for (int _V = 0; _V < V; ++_V) {
      MATRIX_DEF(3, 3);

      // col 1
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_0, nil)
      AVX512_CALCULATE_W_5(0)

      // col 2
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_1, nil)
      AVX512_CALCULATE_W_5(1)

      // col 3
      __m<V> r2_405 = _mm<V>::set_ps(IMM_BCAST16(2.0f / 405.0f));
      __m<V> r4_405 = _mm<V>::set_ps(IMM_BCAST16(4.0f / 405.0f));
      __m<V> r8_405 = _mm<V>::set_ps(IMM_BCAST16(8.0f / 405.0f));
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_2, nil)
      AVX512_CALCULATE_W_5(2)

      // col 4
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_3, nil)
      AVX512_CALCULATE_W_5(3)

      // col 5
      __m<V> r16_405 = _mm<V>::set_ps(IMM_BCAST16(16.0f / 405.0f));
      __m<V> r32_405 = _mm<V>::set_ps(IMM_BCAST16(32.0f / 405.0f));
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_4, nil)
      AVX512_CALCULATE_W_5(4)

      // col 6
      BOOST_PP_REPEAT(3, AVX512_CALCULATE_W_5_5, nil)
      AVX512_CALCULATE_W_5(5)

      // col 7
      __m<V> r2_9 = _mm<V>::set_ps(IMM_BCAST16(2.0f / 9.0f));
      __m<V> r1_45 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 45.0f));
      __m<V> r2_45 = _mm<V>::set_ps(IMM_BCAST16(2.0f / 45.0f));
      __m<V> r4_45 = _mm<V>::set_ps(IMM_BCAST16(4.0f / 45.0f));
      __m<V> r8_45 = _mm<V>::set_ps(IMM_BCAST16(8.0f / 45.0f));
      __m<V> r16_45 = _mm<V>::set_ps(IMM_BCAST16(16.0f / 45.0f));

      c0 = (r2_9 * (f02 + f22));
      c1 = (r1_45 * f02 + r4_45 * f22);
      c2 = (r16_45 * f02 + r4_45 * f22);
      t06 = (r2_9 * f12 + c0);
      _mm<V>::store_ps(T(0, 6), t06);
      t16 = -(r2_9 * f12 - c0);
      _mm<V>::store_ps(T(1, 6), t16);
      t26 = (r2_45 * f12 + c1);
      _mm<V>::store_ps(T(2, 6), t26);
      t36 = (r2_45 * f12 - c1);
      _mm<V>::store_ps(T(3, 6), t36);
      t46 = (r8_45 * f12 + c2);
      _mm<V>::store_ps(T(4, 6), t46);
      t56 = (r8_45 * f12 - c2);
      _mm<V>::store_ps(T(5, 6), t56);
      t66 = f22;
      _mm<V>::store_ps(T(6, 6), t66);
    }
  }
}; // elk_conv_wino_trans_weights
} // namespace euler
