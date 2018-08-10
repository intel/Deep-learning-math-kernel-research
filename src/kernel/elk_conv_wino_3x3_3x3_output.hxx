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
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::
__trans_output(elx_conv_t<float> &xc, float *output,
      float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end) {
  float dummy[16];
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

  auto p_cb = [&](int _h, int _w, int _V) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, _V);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return &dummy[_V];
      else
        return &md3(aoutput, _h, _w, _V);
    }
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]
  float c0[V], c1[V], c2[V], c3[V], c4[V];
#pragma omp simd
  for (int _V = 0; _V < V; ++_V) {
    C(0) = T(0, 0) + T(0, 1) + T(0, 2) + T(0, 3);
    C(1) = T(1, 0) + T(1, 1) + T(1, 2) + T(1, 3);
    C(2) = T(2, 0) + T(2, 1) + T(2, 2) + T(2, 3);
    C(3) = T(3, 0) + T(3, 1) + T(3, 2) + T(3, 3);
    C(4) = T(4, 0) + T(4, 1) + T(4, 2) + T(4, 3);
    P(0, 0) = C(0) + C(1) + C(2) + C(3);
    if (with_bias) P(0, 0) += B;
    if (with_relu) P(0, 0) = P(0, 0) > 0 ? P(0, 0) : 0;
    P(1, 0) = C(2) - C(1) + 2 * C(3);
    if (with_bias) P(1, 0) += B;
    if (with_relu) P(1, 0) = P(1, 0) > 0 ? P(1, 0) : 0;
    P(2, 0) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias) P(2, 0) += B;
    if (with_relu) P(2, 0) = P(2, 0) > 0 ? P(2, 0) : 0;

    C(0) = T(0, 2) - T(0, 1) + 2 * T(0, 3);
    C(1) = T(1, 2) - T(1, 1) + 2 * T(1, 3);
    C(2) = T(2, 2) - T(2, 1) + 2 * T(2, 3);
    C(3) = T(3, 2) - T(3, 1) + 2 * T(3, 3);
    C(4) = T(4, 2) - T(4, 1) + 2 * T(4, 3);
    P(0, 1) = C(0) + C(1) + C(2) + C(3);
    if (with_bias) P(0, 1) += B;
    if (with_relu) P(0, 1) = P(0, 1) > 0 ? P(0, 1) : 0;
    P(1, 1) = C(2) - C(1) + 2 * C(3);
    if (with_bias) P(1, 1) += B;
    if (with_relu) P(1, 1) = P(1, 1) > 0 ? P(1, 1) : 0;
    P(2, 1) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias) P(2, 1) += B;
    if (with_relu) P(2, 1) = P(2, 1) > 0 ? P(2, 1) : 0;

    C(0) = T(0, 1) + T(0, 2) + 4 * T(0, 3) + T(0, 4);
    C(1) = T(1, 1) + T(1, 2) + 4 * T(1, 3) + T(1, 4);
    C(2) = T(2, 1) + T(2, 2) + 4 * T(2, 3) + T(2, 4);
    C(3) = T(3, 1) + T(3, 2) + 4 * T(3, 3) + T(3, 4);
    C(4) = T(4, 1) + T(4, 2) + 4 * T(4, 3) + T(4, 4);
    P(0, 2) = C(0) + C(1) + C(2) + C(3);
    if (with_bias) P(0, 2) += B;
    if (with_relu) P(0, 2) = P(0, 2) > 0 ? P(0, 2) : 0;
    P(1, 2) = C(2) - C(1) + 2 * C(3);
    if (with_bias) P(1, 2) += B;
    if (with_relu) P(1, 2) = P(1, 2) > 0 ? P(1, 2) : 0;
    P(2, 2) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias) P(2, 2) += B;
    if (with_relu) P(2, 2) = P(2, 2) > 0 ? P(2, 2) : 0;
  }
}

#define AVX512_CALCULATE_C_0(z, n, nil)                                        \
  c##n = ADD(ADD(ADD(t##n##0, t##n##1), t##n##2), t##n##3);
#define AVX512_CALCULATE_C_1(z, n, nil)                                        \
  c##n = FMADD(z2, t##n##3, SUB(t##n##2, t##n##1));
#define AVX512_CALCULATE_C_2(z, n, nil)                                        \
  c##n = FMADD(z4, t##n##3, ADD(ADD(t##n##1, t##n##2), t##n##4));
#define AVX512_CALCULATE_P(n)                                                  \
  __m512 p0##n = ADD(ADD(ADD(c0, c1), c2), c3);                                \
  if (with_bias)                                                              \
    p0##n = ADD(p0##n, *(__m512*)bias);                                        \
  if (with_relu) {                                                            \
    zero = XOR(zero, zero);                                                    \
    p0##n = MAX(p0##n, zero);                                                  \
  }                                                                            \
  __m512 p1##n = FMADD(z2, c3, SUB(c2, c1));                                   \
  if (with_bias)                                                              \
    p1##n = ADD(p1##n, *(__m512*)bias);                                        \
  if (with_relu)                                                              \
    p1##n = MAX(p1##n, zero);                                                  \
  __m512 p2##n = FMADD(z4, c3, ADD(ADD(c1, c2), c4));                          \
  if (with_bias)                                                              \
    p2##n = ADD(p2##n, *(__m512*)bias);                                        \
  if (with_relu)                                                              \
    p2##n = MAX(p2##n, zero);                                                  \

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3>::
__trans_output(elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
    float *bias, int hOA_end, int wOA_end)
{
  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

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

  __m512 c0, c1, c2, c3, c4, zero;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

#undef t
#undef OP
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
  MATRIX_DEF(5, 5);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_0, nil);
  AVX512_CALCULATE_P(0);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_1, nil);
  AVX512_CALCULATE_P(1);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_2, nil);
  AVX512_CALCULATE_P(2);

#undef OP
#define p(m, n) p##m##n
#define OP(m,n) _mm512_store_ps(P(m, n), p(m, n))
  MATRIX_DEF(3, 3);
}

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {
  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) md4(atoutput, _h, 0, 0, _V)
#define P(_h) md2(atoutputa, _h, _V)

#pragma omp simd
  for (int _V = 0; _V < V; ++_V) {
    P(0) = T(0) + T(1) + T(2) + T(3);
    P(1) = - T(1) + T(2) + 2.0f * T(3);
    P(2) = T(1) + T(2) + 4.0f * T(3) + T(4);
  }
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {
  ENABLE_AVX512F();

  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) &md4(atoutput, _h, 0, 0, 0)
#define P(_h) &md2(atoutputa, _h, 0)

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  __m512 t0 = _mm512_load_ps(T(0));
  __m512 t1 = _mm512_load_ps(T(1));
  __m512 t2 = _mm512_load_ps(T(2));
  __m512 t3 = _mm512_load_ps(T(3));
  __m512 t4 = _mm512_load_ps(T(4));

  __m512 p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  __m512 p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  __m512 p2 = ADD(ADD(ADD(MUL(z4, t3), t2), t1), t4);

  if (stream_out) {
    _mm512_stream_ps(P(0), p0);
    _mm512_stream_ps(P(1), p1);
    _mm512_stream_ps(P(2), p2);
  } else {
    _mm512_store_ps(P(0), p0);
    _mm512_store_ps(P(1), p1);
    _mm512_store_ps(P(2), p2);
  }
}

#define GENERIC_CALCULATE_TILE_5(z, n, nil)                    \
  P(n, 0) = T(n, 0) + T(n, 1) + T(n, 2) + T(n, 3);             \
  if (with_bias) P(n, 0) += B;                                \
  if (with_relu) P(n, 0) = P(n, 0) > 0 ? P(n, 0) : 0;         \
  P(n, 1) = T(n, 2) - T(n, 1) + z2 * T(n, 3);                  \
  if (with_bias) P(n, 1) += B;                                \
  if (with_relu) P(n, 1) = P(n, 1) > 0 ? P(n, 1) : 0;         \
  P(n, 2) = T(n, 1) + T(n, 2) + z4 * T(n, 3) + T(n, 4);        \
  if (with_bias) P(n, 2) += B;                                \
  if (with_relu) P(n, 2) = P(n, 2) > 0 ? P(n, 2) : 0;

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

  float dummy[V];
  auto p_cb = [&](int _h, int _w, int _V) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, _V);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return &dummy[_V];
      else
        return &md3(aoutput, _h, _w, _V);
    }
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]

#pragma omp simd
  for (int _V = 0; _V < V; ++_V) {
    BOOST_PP_REPEAT(3, GENERIC_CALCULATE_TILE_5, nil)
  }
}

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {

  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

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

  // __m512 c0, c1, c2, c3, c4;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  __m512 t0, t1, t2, t3, t4, p0, p1, p2, zero;
  t0 = _mm512_load_ps(T(0,0));
  t1 = _mm512_load_ps(T(0,1));
  t2 = _mm512_load_ps(T(0,2));
  t3 = _mm512_load_ps(T(0,3));
  t4 = _mm512_load_ps(T(0,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias) p0 = ADD(p0, *(__m512*)bias);
  if (with_relu) { zero = XOR(zero, zero); p0 = MAX(p0, zero); }
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias) p1 = ADD(p1, *(__m512*)bias);
  if (with_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias) p2 = ADD(p2, *(__m512*)bias);
  if (with_relu) p2 = MAX(p2, zero);

  _mm512_store_ps(P(0,0), p0);
  _mm512_store_ps(P(0,1), p1);
  _mm512_store_ps(P(0,2), p2);

  t0 = _mm512_load_ps(T(1,0));
  t1 = _mm512_load_ps(T(1,1));
  t2 = _mm512_load_ps(T(1,2));
  t3 = _mm512_load_ps(T(1,3));
  t4 = _mm512_load_ps(T(1,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias) p0 = ADD(p0, *(__m512*)bias);
  if (with_relu) p0 = MAX(p0, zero);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias) p1 = ADD(p1, *(__m512*)bias);
  if (with_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias) p2 = ADD(p2, *(__m512*)bias);
  if (with_relu) p2 = MAX(p2, zero);

  _mm512_store_ps(P(1,0), p0);
  _mm512_store_ps(P(1,1), p1);
  _mm512_store_ps(P(1,2), p2);

  t0 = _mm512_load_ps(T(2,0));
  t1 = _mm512_load_ps(T(2,1));
  t2 = _mm512_load_ps(T(2,2));
  t3 = _mm512_load_ps(T(2,3));
  t4 = _mm512_load_ps(T(2,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias) p0 = ADD(p0, *(__m512*)bias);
  if (with_relu) p0 = MAX(p0, zero);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias) p1 = ADD(p1, *(__m512*)bias);
  if (with_relu) p1 = MAX(p1, zero);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias) p2 = ADD(p2, *(__m512*)bias);
  if (with_relu) p2 = MAX(p2, zero);

  _mm512_store_ps(P(2,0), p0);
  _mm512_store_ps(P(2,1), p1);
  _mm512_store_ps(P(2,2), p2);
}
} // namespace euler
