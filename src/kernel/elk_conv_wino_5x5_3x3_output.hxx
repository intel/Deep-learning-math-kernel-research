#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_5x5_3x3_input.hxx"

namespace euler {
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

#define AVX512_CALCULATE_O(n)                                                  \
  __m<V> p0##n = ADD(ADD(ADD(ADD(ADD(c0, c1), c2), c3), c4), c5);              \
  if (fuse_bias)                                                               \
    p0##n = ADD(p0##n, *(__m<V>*)bias);                                        \
  if (fuse_ip_sum)                                                             \
    p0##n = ADD(p0##n, *(__m<V>*)P(0, n));                                     \
  if (fuse_relu) {                                                             \
    zero = XOR(zero, zero);                                                    \
    p0##n = MAX(p0##n, zero);                                                  \
  }                                                                            \
  _mm<V>::store_ps(P(0, n), p0##n);                                            \
  __m<V> p1##n = ADD(FMADD(z2, SUB(c2, c3), c0), FMSUB(z1_2, SUB(c4, c5), c1));\
  if (fuse_bias)                                                               \
    p1##n = ADD(p1##n, *(__m<V>*)bias);                                        \
  if (fuse_ip_sum)                                                             \
    p1##n = ADD(p1##n, *(__m<V>*)P(1, n));                                     \
  if (fuse_relu)                                                               \
    p1##n = MAX(p1##n, zero);                                                  \
  _mm<V>::store_ps(P(1, n), p1##n);                                            \
  __m<V> p2##n = ADD(FMADD(z4, ADD(c2, c3), c0), FMADD(z1_4, ADD(c4, c5), c1));\
  if (fuse_bias)                                                               \
    p2##n = ADD(p2##n, *(__m<V>*)bias);                                        \
  if (fuse_ip_sum)                                                             \
    p2##n = ADD(p2##n, *(__m<V>*)P(2, n));                                     \
  if (fuse_relu)                                                               \
    p2##n = MAX(p2##n, zero);                                                  \
  _mm<V>::store_ps(P(2, n), p2##n);                                            \
  __m<V> p3##n = ADD(FMADD(z8, SUB(c2, c3), c0), FMSUB(z1_8, SUB(c4, c5), c1));\
  if (fuse_bias)                                                               \
    p3##n = ADD(p3##n, *(__m<V>*)bias);                                        \
  if (fuse_ip_sum)                                                             \
    p3##n = ADD(p3##n, *(__m<V>*)P(3, n));                                     \
  if (fuse_relu)                                                               \
    p3##n = MAX(p3##n, zero);                                                  \
  _mm<V>::store_ps(P(3, n), p3##n);                                            \
  __m<V> p4##n = ADD(FMADD(z16, ADD(c2, c3), c0), FMADD(z1_16, ADD(c4, c5), c1));

#define AVX512_ADD_B(n);                                                       \
  if (fuse_bias)                                                               \
    p4##n = ADD(p4##n, *(__m<V>*)bias);                                        \
  if (fuse_ip_sum)                                                             \
    p4##n = ADD(p4##n, *(__m<V>*)P(4, n));                                     \
  if (fuse_relu)                                                               \
    p4##n = MAX(p4##n, zero);                                                  \
  _mm<V>::store_ps(P(4, n), p4##n);

// template <const bool is_border_, const bool with_bias>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 7, 3>::
__trans_output(elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
    float *bias, int hOA_end, int wOA_end) {
  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
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
  p40 = ADD(ADD(ADD(ADD(ADD(ADD(p40, t60), t61),
      t62), t63), t64), t65);
  AVX512_ADD_B(0)

  BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_1, nil)
  AVX512_CALCULATE_O(1);
  p41 = ADD(p41, FMADD(z2, SUB(t62, t63),
      FMADD(z1_2, SUB(t64, t65), SUB(t60, t61))));
  AVX512_ADD_B(1)

  BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_2, nil)
  AVX512_CALCULATE_O(2);
  p42 = ADD(p42, FMADD(z4, ADD(t62, t63),
      FMADD(z1_4, ADD(t64, t65), ADD(t60, t61))));
  AVX512_ADD_B(2)

  BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_3, nil)
  AVX512_CALCULATE_O(3);
  p43 = ADD(p43, FMADD(z8, SUB(t62, t63),
      FMADD(z1_8, SUB(t64, t65), SUB(t60, t61))));
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
  p44 = ADD(p44, FMADD(z16, ADD(t62, t63), FMADD(z1_16,
      ADD(t64, t65), ADD(ADD(t60, t61), t66))));
  AVX512_ADD_B(4)
}

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 7, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {
  ENABLE_AVX512F();

  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) &md4(atoutput, _h, 0, 0, 0)
#define P(_h) &md2(atoutputa, _h, 0)

  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));
  __m<V> z8 = _mm<V>::set_ps(IMM_BCAST16(8.0f));
  __m<V> z16 = _mm<V>::set_ps(IMM_BCAST16(16.0f));
  __m<V> z1_2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m<V> z1_4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m<V> z1_8 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 8.0f));
  __m<V> z1_16 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 16.0f));

  __m<V> t0 = _mm<V>::load_ps(T(0));
  __m<V> t1 = _mm<V>::load_ps(T(1));
  __m<V> t2 = _mm<V>::load_ps(T(2));
  __m<V> t3 = _mm<V>::load_ps(T(3));
  __m<V> t4 = _mm<V>::load_ps(T(4));
  __m<V> t5 = _mm<V>::load_ps(T(5));
  __m<V> t6 = _mm<V>::load_ps(T(6));

  __m<V> p0 = ADD(ADD(ADD(ADD(ADD(t0, t1), t2), t3), t4), t5);
  __m<V> p1 = ADD(FMADD(z2, SUB(t2, t3), t0), FMSUB(z1_2, SUB(t4, t5), t1));
  __m<V> p2 = ADD(FMADD(z4, ADD(t2, t3), t0), FMADD(z1_4, ADD(t4, t5), t1));
  __m<V> p3 = ADD(FMADD(z8, SUB(t2, t3), t0), FMSUB(z1_8, SUB(t4, t5), t1));
  __m<V> p4 = ADD(FMADD(z16, ADD(t2, t3), ADD(t0, t1)), FMADD(z1_16, ADD(t4, t5), t6));

  if (stream_out) {
    _mm<V>::stream_ps(P(0), p0);
    _mm<V>::stream_ps(P(1), p1);
    _mm<V>::stream_ps(P(2), p2);
    _mm<V>::stream_ps(P(3), p3);
    _mm<V>::stream_ps(P(4), p4);
  } else {
    _mm<V>::store_ps(P(0), p0);
    _mm<V>::store_ps(P(1), p1);
    _mm<V>::store_ps(P(2), p2);
    _mm<V>::store_ps(P(3), p3);
    _mm<V>::store_ps(P(4), p4);
  }
}

#define AVX512_BH_CALCULATE_TILE_7(z, n, nil)                        \
  t0 = _mm<V>::load_ps(T(n, 0));                                     \
  t1 = _mm<V>::load_ps(T(n, 1));                                     \
  t2 = _mm<V>::load_ps(T(n, 2));                                     \
  t3 = _mm<V>::load_ps(T(n, 3));                                     \
  t4 = _mm<V>::load_ps(T(n, 4));                                     \
  t5 = _mm<V>::load_ps(T(n, 5));                                     \
  t6 = _mm<V>::load_ps(T(n, 6));                                     \
                                                                     \
  p0 = ADD(ADD(ADD(ADD(ADD(t0, t1), t2), t3), t4), t5);              \
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);                       \
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(n, 0));                  \
  if (fuse_relu) { zero = XOR(zero, zero); p0 = MAX(p0, zero); }     \
  p1 = ADD(FMADD(z2, SUB(t2, t3), t0), FMSUB(z1_2, SUB(t4, t5), t1));\
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);                       \
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(n, 1));                  \
  if (fuse_relu) p1 = MAX(p1, zero);                                 \
  p2 = ADD(FMADD(z4, ADD(t2, t3), t0), FMADD(z1_4, ADD(t4, t5), t1));\
  if (fuse_bias) p2 = ADD(p2, *(__m<V>*)bias);                       \
  if (fuse_ip_sum) p2 = ADD(p2, *(__m<V>*)P(n, 2));                  \
  if (fuse_relu) p2 = MAX(p2, zero);                                 \
  p3 = ADD(FMADD(z8, SUB(t2, t3), t0), FMSUB(z1_8, SUB(t4, t5), t1));\
  if (fuse_bias) p3 = ADD(p3, *(__m<V>*)bias);                       \
  if (fuse_ip_sum) p3 = ADD(p3, *(__m<V>*)P(n, 3));                  \
  if (fuse_relu) p3 = MAX(p3, zero);                                 \
  p4 = ADD(FMADD(z16, ADD(t2, t3), ADD(t0, t1)), FMADD(z1_16, ADD(t4, t5), t6)); \
  if (fuse_bias) p4 = ADD(p4, *(__m<V>*)bias);                       \
  if (fuse_ip_sum) p4 = ADD(p4, *(__m<V>*)P(n, 4));                  \
  if (fuse_relu) p4 = MAX(p4, zero);                                 \
                                                                     \
  _mm<V>::store_ps(P(n,0), p0);                                      \
  _mm<V>::store_ps(P(n,1), p1);                                      \
  _mm<V>::store_ps(P(n,2), p2);                                      \
  _mm<V>::store_ps(P(n,3), p3);                                      \
  _mm<V>::store_ps(P(n,4), p4);

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 7, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) float dummy[V];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));
  __m<V> z8 = _mm<V>::set_ps(IMM_BCAST16(8.0f));
  __m<V> z16 = _mm<V>::set_ps(IMM_BCAST16(16.0f));
  __m<V> z1_2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m<V> z1_4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m<V> z1_8 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 8.0f));
  __m<V> z1_16 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 16.0f));

  __m<V> t0, t1, t2, t3, t4, t5, t6, p0, p1, p2, p3, p4, zero;

  BOOST_PP_REPEAT(5, AVX512_BH_CALCULATE_TILE_7, nil)
}
} // namespace euler
