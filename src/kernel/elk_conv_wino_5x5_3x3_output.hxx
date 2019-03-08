#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
#undef FUSE_BIAS
#define FUSE_BIAS(p)                                                           \
  if (std::is_same<BiasType, float>::value) {                                  \
    p = ADD(p, *(__m<V>*)bias);                                                \
  } else {                                                                     \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);                         \
    p = ADD(p, _mm<V>::cvtph_ps(f16v));                                        \
  }

#define AVX512_CALCULATE_O_0(z, n, nil)                                        \
  c##n = ADD(ADD(ADD(ADD(ADD(t##n##0, t##n##1), t##n##2), t##n##3), t##n##4),  \
         t##n##5);
#define AVX512_CALCULATE_O_1(z, n, nil)                                        \
  c##n = ADD(FMADD(z2, SUB(t##n##2, t##n##3), t##n##0), FMSUB(z1_2,            \
         SUB(t##n##4, t##n##5), t##n##1));
#define AVX512_CALCULATE_O_2(z, n, nil)                                        \
  c##n = ADD(FMADD(z4, ADD(t##n##2, t##n##3), t##n##0), FMADD(z1_4,            \
         ADD(t##n##4, t##n##5), t##n##1));
#define AVX512_CALCULATE_O_3(z, n, nil)                                        \
  c##n = ADD(FMADD(z8, SUB(t##n##2, t##n##3), t##n##0), FMSUB(z1_8,            \
         SUB(t##n##4, t##n##5), t##n##1));
#define AVX512_CALCULATE_O_4(z, n, nil)                                        \
  c##n = ADD(FMADD(z16, ADD(t##n##2, t##n##3), ADD(t##n##0, t##n##1)),         \
         FMADD(z1_16, ADD(t##n##4, t##n##5), t##n##6));

#undef STORE
#define STORE(i, j)                                                            \
  if (std::is_same<OutputType, float>::value)                                  \
    _mm<V>::store_ps(P(i, j), p##i##j);                                        \
  else {                                                                       \
    auto f16 = _mm<V>::cvtps_ph(p##i##j,                                       \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                        \
    _mm<V/2>::store_si256((__m256i *)P(i, j), f16);                            \
  }

#define AVX512_CALCULATE_O(n)                                                  \
  __m<V> p0##n = ADD(ADD(ADD(ADD(ADD(c0, c1), c2), c3), c4), c5);              \
  if (fuse_bias) {FUSE_BIAS(p0##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p0##n = ADD(p0##n, *(__m<V>*)P(0, n));                                     \
  if (fuse_relu) {                                                             \
    zero = XOR(zero, zero);                                                    \
    p0##n = MAX(p0##n, zero);                                                  \
  }                                                                            \
  STORE(0, n)                                                                  \
  __m<V> p1##n = ADD(FMADD(z2, SUB(c2, c3), c0), FMSUB(z1_2, SUB(c4, c5), c1));\
  if (fuse_bias) {FUSE_BIAS(p1##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p1##n = ADD(p1##n, *(__m<V>*)P(1, n));                                     \
  if (fuse_relu)                                                               \
    p1##n = MAX(p1##n, zero);                                                  \
  STORE(1, n)                                                                  \
  __m<V> p2##n = ADD(FMADD(z4, ADD(c2, c3), c0), FMADD(z1_4, ADD(c4, c5), c1));\
  if (fuse_bias) {FUSE_BIAS(p2##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p2##n = ADD(p2##n, *(__m<V>*)P(2, n));                                     \
  if (fuse_relu)                                                               \
    p2##n = MAX(p2##n, zero);                                                  \
  STORE(2, n)                                                                  \
  __m<V> p3##n = ADD(FMADD(z8, SUB(c2, c3), c0), FMSUB(z1_8, SUB(c4, c5), c1));\
  if (fuse_bias) {FUSE_BIAS(p3##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p3##n = ADD(p3##n, *(__m<V>*)P(3, n));                                     \
  if (fuse_relu)                                                               \
    p3##n = MAX(p3##n, zero);                                                  \
  STORE(3, n)                                                                  \
  __m<V> p4##n = ADD(FMADD(z16, ADD(c2, c3), c0), FMADD(z1_16, ADD(c4, c5), c1));

#define AVX512_ADD_B(n);                                                       \
  if (fuse_bias) {FUSE_BIAS(p4##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p4##n = ADD(p4##n, *(__m<V>*)P(4, n));                                     \
  if (fuse_relu)                                                               \
    p4##n = MAX(p4##n, zero);                                                  \
  STORE(4, n)

template <typename OutputType, typename BiasType,
    int format, bool is_border, bool with_bias, bool with_relu,
    bool with_ip_sum, int V>
struct elk_conv_wino_trans_output<float, OutputType, BiasType, format,
    is_border, with_bias, with_relu, with_ip_sum, ISA_SKX_AVX512, 7, 3, V> {
  constexpr static int A = 7;
  constexpr static int K = 3;

  static void execute(elx_conv_params_t &xc, OutputType *output,
      float atoutput[A][A][V], BiasType *bias, int hOA_end, int wOA_end)
  {
    ENABLE_AVX512F();
    bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
    bool fuse_bias = with_bias && (bias != nullptr);
    bool fuse_relu = with_relu && (bias != nullptr);

    alignas(64) OutputType dummy[16];
    auto p_cb = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
        return &md3(aoutput, _h, _w, 0);
      } else if (format == TKF_BLOCKED) {
        MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      } else {
        MD3(OutputType, aoutput, output, xc.oh, xc.ow, xc.oc);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      }
    };

#undef P
#undef T
#define T(_h, _w) atoutput[_h][_w]
#define P(_h, _w) p_cb(_h, _w)

    __m<V> c0, c1, c2, c3, c4, c5, zero;

    __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
    __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));
    __m<V> z8 = _mm<V>::set_ps(IMM_BCAST16(8.0f));
    __m<V> z16 = _mm<V>::set_ps(IMM_BCAST16(16.0f));
    __m<V> z1_2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
    __m<V> z1_4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
    __m<V> z1_8 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 8.0f));
    __m<V> z1_16 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 16.0f));

#undef t
#undef OP
#define t(m, n) t##m##n
#define OP(m,n) __m<V> t(m,n) = _mm<V>::load_ps(T(m, n))
    MATRIX_DEF(7, 6);

    BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_0, nil)
    AVX512_CALCULATE_O(0);
    p40 = ADD(ADD(ADD(ADD(ADD(ADD(p40, t60), t61), t62), t63), t64), t65);
    AVX512_ADD_B(0)

    BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_1, nil)
    AVX512_CALCULATE_O(1);
    p41 = ADD(p41,
        FMADD(z2, SUB(t62, t63), FMADD(z1_2, SUB(t64, t65), SUB(t60, t61))));
    AVX512_ADD_B(1)

    BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_2, nil)
    AVX512_CALCULATE_O(2);
    p42 = ADD(p42,
        FMADD(z4, ADD(t62, t63), FMADD(z1_4, ADD(t64, t65), ADD(t60, t61))));
    AVX512_ADD_B(2)

    BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_3, nil)
    AVX512_CALCULATE_O(3);
    p43 = ADD(p43,
        FMADD(z8, SUB(t62, t63), FMADD(z1_8, SUB(t64, t65), SUB(t60, t61))));
    AVX512_ADD_B(3)

    __m<V> t06 = _mm<V>::load_ps(T(0, 6));
    __m<V> t16 = _mm<V>::load_ps(T(1, 6));
    __m<V> t26 = _mm<V>::load_ps(T(2, 6));
    __m<V> t36 = _mm<V>::load_ps(T(3, 6));
    __m<V> t46 = _mm<V>::load_ps(T(4, 6));
    __m<V> t56 = _mm<V>::load_ps(T(5, 6));
    __m<V> t66 = _mm<V>::load_ps(T(6, 6));
    BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_4, nil)
    AVX512_CALCULATE_O(4);
    p44 = ADD(p44,
        FMADD(z16, ADD(t62, t63),
            FMADD(z1_16, ADD(t64, t65), ADD(ADD(t60, t61), t66))));
    AVX512_ADD_B(4)
  }
}; // elk_conv_wino_trans_output
} // namespace euler
