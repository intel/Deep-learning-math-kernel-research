#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_3x3_3x3_input.hxx"

// #ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
// #error "Don't include this file directly"
// #endif
namespace euler {
#undef FUSE_BIAS
#define FUSE_BIAS(p)                                                           \
  if (std::is_same<BiasType, float>::value) {                                  \
    p = ADD(p, *(__m<V>*)bias);                                                \
  } else {                                                                     \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);                         \
    p = ADD(p, _mm<V>::cvtph_ps(f16v));                                        \
  }

#define AVX512_CALCULATE_C_0(z, n, nil)                                        \
  c##n = ADD(ADD(ADD(t##n##0, t##n##1), t##n##2), t##n##3);
#define AVX512_CALCULATE_C_1(z, n, nil)                                        \
  c##n = FMADD(z2, t##n##3, SUB(t##n##2, t##n##1));
#define AVX512_CALCULATE_C_2(z, n, nil)                                        \
  c##n = FMADD(z4, t##n##3, ADD(ADD(t##n##1, t##n##2), t##n##4));
#define AVX512_CALCULATE_P(n)                                                  \
  __m<V> p0##n = ADD(ADD(ADD(c0, c1), c2), c3);                                \
  if (fuse_bias) {FUSE_BIAS(p0##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p0##n = ADD(p0##n, *(__m<V>*)P(0, n));                                     \
  if (fuse_relu) {                                                             \
    zero = XOR(zero, zero);                                                    \
    p0##n = MAX(p0##n, zero);                                                  \
  }                                                                            \
  __m<V> p1##n = FMADD(z2, c3, SUB(c2, c1));                                   \
  if (fuse_bias) {FUSE_BIAS(p1##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p1##n = ADD(p1##n, *(__m<V>*)P(1, n));                                     \
  if (fuse_relu)                                                               \
    p1##n = MAX(p1##n, zero);                                                  \
  __m<V> p2##n = FMADD(z4, c3, ADD(ADD(c1, c2), c4));                          \
  if (fuse_bias) {FUSE_BIAS(p2##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p2##n = ADD(p2##n, *(__m<V>*)P(2, n));                                     \
  if (fuse_relu)                                                               \
    p2##n = MAX(p2##n, zero);

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_output(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A][V], BiasType *bias, TrOpType *shift,
    int hOA_end, int wOA_end)
{
  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) OutputType dummy[V];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#undef t
#undef OP
#undef STORE
#undef ISTORE
#undef FUSE_BIAS

#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)
#define t(m, n) t##m##n
#define OP(m,n) t(m,n) = _mm<V>::load_ps(T(m, n))

#define ISTORE(i, j)                                              \
  if (std::is_same<OutputType, float>::value)                     \
    _mm<V>::store_ps(P(i, j), p##i##j);                           \
  else {                                                          \
    auto f16 = _mm<V>::cvtps_ph(p##i##j,                          \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
    _mm<V/2>::store_si256((__m256i *)P(i, j), f16);               \
  }

#define FUSE_BIAS(p)                                              \
  if (std::is_same<BiasType, float>::value) {                     \
    p = ADD(p, *(__m<V>*)bias);                                   \
  } else {                                                        \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);            \
    p = ADD(p, _mm<V>::cvtph_ps(f16v));                           \
  }

#define BIAS                                                      \
  std::is_same<BiasType, float>::value                            \
  ? *(__m<V>*)bias                                                \
  : _mm<V>::cvtph_ps(_mm<V/2>::load_si256((__m256i *)bias))

#define STORE(i, j)                                               \
  if (std::is_same<OutputType, float>::value)                     \
    _mm<V>::store_ps(P(i, j), p##j);                              \
  else {                                                          \
    auto f16 = _mm<V>::cvtps_ph(p##j,                             \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
    _mm<V/2>::store_si256((__m256i *)P(i, j), f16);               \
  }

  __m<V> M[3][5];

  auto z0 = _mm<V>::set1_ps(0.3333333333333333f);
  auto z1 = _mm<V>::set1_ps(0.6666666666666666f);
  auto z2 = _mm<V>::set1_ps(1.3333333333333333f);
  __m<V> z = XOR(z, z);
  auto mshift = shift == nullptr ? _mm<V>::setzero_ps() : *(__m<V>*)shift;

#pragma unroll
  for (int i = 0; i < 5; i++) {
    auto f0 = _mm<V>::load_ps(T(0, i));
    auto f1 = _mm<V>::load_ps(T(1, i));
    auto f2 = _mm<V>::load_ps(T(2, i));
    auto f3 = _mm<V>::load_ps(T(3, i));
    auto f4 = _mm<V>::load_ps(T(4, i));

    auto t0 = f2 * z0;
    auto t1 = t0 + f1;

    M[0][i] = f3 * z2 + f0 + t1;
    M[1][i] = t0 - f1 - f3 * z1;
    M[2][i] = f3 * z0 + f4 + t1;
  }

#pragma unroll
  for (int i = 0; i < 3; i++) {
    auto f0 = M[i][0];
    auto f1 = M[i][1];
    auto f2 = M[i][2];
    auto f3 = M[i][3];
    auto f4 = M[i][4];

    auto t0 = f2 * z0;
    auto t1 = t0 + f1;

    auto p0 = f3 * z2 + f0 + t1 + mshift;
    auto p1 = t0 - f1 - f3 * z1 + mshift;
    auto p2 = f3 * z0 + f4 + t1 + mshift;

    if (fuse_bias) {
      p0 += BIAS;
      p1 += BIAS;
      p2 += BIAS;
    }
    if (fuse_ip_sum) {
      p0 += *(__m<V> *)P(i, 0);
      p1 += *(__m<V> *)P(i, 1);
      p2 += *(__m<V> *)P(i, 2);
    }
    if (fuse_relu) {
      p0 = MAX(p0, z);
      p1 = MAX(p1, z);
      p2 = MAX(p2, z);
    }
    STORE(i, 0)
    STORE(i, 1)
    STORE(i, 2)
  }

#if 0
#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m<V> c0, c1, c2, c3, c4, zero;
  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));

#undef t
#undef OP
#define t(m, n) t##m##n
#define OP(m,n) __m<V> t(m,n) = _mm<V>::load_ps(T(m, n))
  MATRIX_DEF(5, 5);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_0, nil);
  AVX512_CALCULATE_P(0);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_1, nil);
  AVX512_CALCULATE_P(1);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_2, nil);
  AVX512_CALCULATE_P(2);

#undef OP
#define p(m, n) p##m##n
//#define OP(m,n) _mm<V>::store_ps(P(m, n), p(m, n))
#define OP(m,n)                                         \
  if (std::is_same<OutputType, float>::value)           \
    _mm<V>::store_ps(P(m, n), p(m, n));                 \
  else {                                                \
    auto f16 = _mm<V>::cvtps_ph(p(m, n),                \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); \
    _mm<V/2>::store_si256((__m256i *)P(m, n), f16);     \
  }
  MATRIX_DEF(3, 3);
#endif
}

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_outputa_th(elx_conv_t<UserTypes> &xc,
    TrOpType *toutputa, TrOpType *toutput, int Tz, bool stream_out)
{
  ENABLE_AVX512F();

  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) &md4(atoutput, _h, 0, 0, 0)
#define P(_h) &md2(atoutputa, _h, 0)

  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));

  __m<V> t0 = _mm<V>::load_ps(T(0));
  __m<V> t1 = _mm<V>::load_ps(T(1));
  __m<V> t2 = _mm<V>::load_ps(T(2));
  __m<V> t3 = _mm<V>::load_ps(T(3));
  __m<V> t4 = _mm<V>::load_ps(T(4));

  __m<V> p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  __m<V> p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  __m<V> p2 = ADD(ADD(ADD(MUL(z4, t3), t2), t1), t4);

  if (stream_out) {
    _mm<V>::stream_ps(P(0), p0);
    _mm<V>::stream_ps(P(1), p1);
    _mm<V>::stream_ps(P(2), p2);
  } else {
    _mm<V>::store_ps(P(0), p0);
    _mm<V>::store_ps(P(1), p1);
    _mm<V>::store_ps(P(2), p2);
  }
}

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_outputa_bh(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A - K + 1][V], BiasType *bias,
    int hOA_end, int wOA_end)
{

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

  // __m<V> c0, c1, c2, c3, c4;
  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));

  __m<V> t0, t1, t2, t3, t4, p0, p1, p2, zero;
  t0 = _mm<V>::load_ps(T(0,0));
  t1 = _mm<V>::load_ps(T(0,1));
  t2 = _mm<V>::load_ps(T(0,2));
  t3 = _mm<V>::load_ps(T(0,3));
  t4 = _mm<V>::load_ps(T(0,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(0, 0));
  if (fuse_relu) { zero = XOR(zero, zero); p0 = MAX(p0, zero); }
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(0, 1));
  if (fuse_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (fuse_bias) p2 = ADD(p2, *(__m<V>*)bias);
  if (fuse_ip_sum) p2 = ADD(p2, *(__m<V>*)P(0, 2));
  if (fuse_relu) p2 = MAX(p2, zero);

  _mm<V>::store_ps(P(0,0), p0);
  _mm<V>::store_ps(P(0,1), p1);
  _mm<V>::store_ps(P(0,2), p2);

  t0 = _mm<V>::load_ps(T(1,0));
  t1 = _mm<V>::load_ps(T(1,1));
  t2 = _mm<V>::load_ps(T(1,2));
  t3 = _mm<V>::load_ps(T(1,3));
  t4 = _mm<V>::load_ps(T(1,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(1, 0));
  if (fuse_relu) p0 = MAX(p0, zero);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(1, 1));
  if (fuse_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (fuse_bias) p2 = ADD(p2, *(__m<V>*)bias);
  if (fuse_ip_sum) p2 = ADD(p2, *(__m<V>*)P(1, 2));
  if (fuse_relu) p2 = MAX(p2, zero);

  _mm<V>::store_ps(P(1,0), p0);
  _mm<V>::store_ps(P(1,1), p1);
  _mm<V>::store_ps(P(1,2), p2);

  t0 = _mm<V>::load_ps(T(2,0));
  t1 = _mm<V>::load_ps(T(2,1));
  t2 = _mm<V>::load_ps(T(2,2));
  t3 = _mm<V>::load_ps(T(2,3));
  t4 = _mm<V>::load_ps(T(2,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(2, 0));
  if (fuse_relu) p0 = MAX(p0, zero);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(2, 1));
  if (fuse_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (fuse_bias) p2 = ADD(p2, *(__m<V>*)bias);
  if (fuse_ip_sum) p2 = ADD(p2, *(__m<V>*)P(2, 2));
  if (fuse_relu) p2 = MAX(p2, zero);

  _mm<V>::store_ps(P(2,0), p0);
  _mm<V>::store_ps(P(2,1), p1);
  _mm<V>::store_ps(P(2,2), p2);
}
} // namespace euler
