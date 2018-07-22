#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

#ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
#error "Don't include this file directly"
#endif
namespace euler {

#define AVX2_CALCULATE_O_0(z, n, nil)                                        \
  C(n) = T(n, 0) + T(n, 1) + T(n, 2) + T(n, 3)                               \
      + T(n, 4) + T(n, 5);
#define AVX2_CALCULATE_O_1(z, n, nil)                                        \
  C(n) = T(n, 0) - T(n, 1) + a2 * (T(n, 2) - T(n, 3))                        \
      + a1_2 * (T(n, 4) - T(n, 5));
#define AVX2_CALCULATE_O_2(z, n, nil)                                        \
  C(n) = T(n, 0) + T(n, 1) + a4 * (T(n, 2) + T(n, 3))                        \
      + a1_4* (T(n, 4) + T(n, 5));
#define AVX2_CALCULATE_O_3(z, n, nil)                                        \
  C(n) = T(n, 0) - T(n, 1) + a8 * (T(n, 2) - T(n, 3))                        \
      + a1_8* (T(n, 4) - T(n, 5));
#define AVX2_CALCULATE_O_4(z, n, nil)                                        \
  C(n) = T(n, 0) + T(n, 1) + a16 * (T(n, 2) + T(n, 3))                       \
      + a1_16* (T(n, 4) + T(n, 5)) + T(n, 6);

#define AVX2_CALCULATE_O(n)                                                  \
  P(0, n) = C(0) + C(1) + C(2) + C(3) + C(4) + C(5);                         \
  if (with_bias_) P(0, n) += B;                                              \
  P(1, n) = C(0) - C(1) + a2 * (C(2) - C(3)) + a1_2 * (C(4) - C(5));         \
  if (with_bias_) P(1, n) += B;                                              \
  P(2, n) = C(0) + C(1) + a4 * (C(2) + C(3)) + a1_4 * (C(4) + C(5));         \
  if (with_bias_) P(2, n) += B;                                              \
  P(3, n) = C(0) - C(1) + a8 * (C(2) - C(3)) + a1_8 * (C(4) - C(5));         \
  if (with_bias_) P(3, n) += B;                                              \
  P(4, n) = C(0) + C(1) + a16 * (C(2) + C(3)) + a1_16 * (C(4) + C(5));

#define AVX2_ADD_TAIL_0(n, z)                                           \
  P(4, n) += T(6, 0) - T(6, 1) + a##z * (T(6, 2) - T(6, 3))             \
      + a1_##z * (T(6, 4) - T(6, 5));
#define AVX2_ADD_TAIL_1(n, z)                                           \
  P(4, n) += T(6, 0) + T(6, 1) + a##z * (T(6, 2) + T(6, 3))             \
      + a1_##z * (T(6, 4) + T(6, 5));

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 7, 3, 16, ISA_GENERIC)
{
  const float a2 = 2.0f;
  const float a4 = 4.0f;
  const float a8 = 8.0f;
  const float a16 = 16.0f;
  const float a1_2 = 1.0f / 2.0f;
  const float a1_4 = 1.0f / 4.0f;
  const float a1_8 = 1.0f / 8.0f;
  const float a1_16 = 1.0f / 16.0f;

  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    if (_wOA_end == -1) {
      MD(float, aoutput, [A - K + 1][A - K + 1][16], output);
      return &aoutput[_h][_w][_V];
    } else {
      MD(float, aoutput, [xc.oh][xc.ow][16], output);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
        return &dummy[_V];
      else
        return &aoutput[_h][_w][_V];
    }
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define C(n) C##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]

  float C0[16], C1[16], C2[16], C3[16], C4[16], C5[16];

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    BOOST_PP_REPEAT(6, AVX2_CALCULATE_O_0, nil)
    AVX2_CALCULATE_O(0)
    P(4, 0) += T(6, 0) + T(6, 1) + T(6, 2) + T(6, 3) + T(6, 4) + T(6, 5);
    if (with_bias_) P(4, 0) += B;


    BOOST_PP_REPEAT(6, AVX2_CALCULATE_O_1, nil)
    AVX2_CALCULATE_O(1)
    AVX2_ADD_TAIL_0(1, 2)
    if (with_bias_) P(4, 1) += B;


    BOOST_PP_REPEAT(6, AVX2_CALCULATE_O_2, nil)
    AVX2_CALCULATE_O(2)
    AVX2_ADD_TAIL_1(2, 4)
    if (with_bias_) P(4, 2) += B;


    BOOST_PP_REPEAT(6, AVX2_CALCULATE_O_3, nil)
    AVX2_CALCULATE_O(3)
    AVX2_ADD_TAIL_0(3, 8)
    if (with_bias_) P(4, 3) += B;


    BOOST_PP_REPEAT(6, AVX2_CALCULATE_O_4, nil)
    AVX2_CALCULATE_O(4)
    AVX2_ADD_TAIL_1(4, 16)
    P(4, 4) += T(6, 6);
    if (with_bias_) P(4, 4) += B;
  }
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

#define AVX512_CALCULATE_O(n)                                                  \
  __m512 p0##n = ADD(ADD(ADD(ADD(ADD(c0, c1), c2), c3), c4), c5);              \
  if (with_bias_)                                                              \
    p0##n = ADD(p0##n, *(__m512*)bias);                                        \
  _mm512_store_ps(P(0, n), p0##n);                                             \
  __m512 p1##n = ADD(FMADD(z2, SUB(c2, c3), c0), FMSUB(z1_2, SUB(c4, c5), c1));\
  if (with_bias_)                                                              \
    p1##n = ADD(p1##n, *(__m512*)bias);                                        \
  _mm512_store_ps(P(1, n), p1##n);                                             \
  __m512 p2##n = ADD(FMADD(z4, ADD(c2, c3), c0), FMADD(z1_4, ADD(c4, c5), c1));\
  if (with_bias_)                                                              \
    p2##n = ADD(p2##n, *(__m512*)bias);                                        \
  _mm512_store_ps(P(2, n), p2##n);                                             \
  __m512 p3##n = ADD(FMADD(z8, SUB(c2, c3), c0), FMSUB(z1_8, SUB(c4, c5), c1));\
  if (with_bias_)                                                              \
    p3##n = ADD(p3##n, *(__m512*)bias);                                        \
  _mm512_store_ps(P(3, n), p3##n);                                             \
  __m512 p4##n = ADD(FMADD(z16, ADD(c2, c3), c0), FMADD(z1_16, ADD(c4, c5), c1));

#define AVX512_ADD_B(n);                                                       \
  if (with_bias_)                                                              \
    p4##n = ADD(p4##n, *(__m512*)bias);                                        \
  _mm512_store_ps(P(4, n), p4##n);

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 7, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (_wOA_end == -1) {
      MD(float, aoutput, [A - K + 1][A - K + 1][16], output);
      return aoutput[_h][_w];
    } else {
      MD(float, aoutput, [xc.oh][xc.ow][16], output);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
        return dummy;
      else
        return aoutput[_h][_w];
    }
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3, c4, c5;

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z8 = _mm512_set_ps(IMM_BCAST16(8.0f));
  __m512 z16 = _mm512_set_ps(IMM_BCAST16(16.0f));
  __m512 z1_2 = _mm512_set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m512 z1_4 = _mm512_set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m512 z1_8 = _mm512_set_ps(IMM_BCAST16(1.0f / 8.0f));
  __m512 z1_16 = _mm512_set_ps(IMM_BCAST16(1.0f / 16.0f));

#undef t
#undef OP
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
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

  __m512 t06 = _mm512_load_ps(T(0, 6));
  __m512 t16 = _mm512_load_ps(T(1, 6));
  __m512 t26 = _mm512_load_ps(T(2, 6));
  __m512 t36 = _mm512_load_ps(T(3, 6));
  __m512 t46 = _mm512_load_ps(T(4, 6));
  __m512 t56 = _mm512_load_ps(T(5, 6));
  __m512 t66 = _mm512_load_ps(T(6, 6));
  BOOST_PP_REPEAT(6, AVX512_CALCULATE_O_4, nil)
  AVX512_CALCULATE_O(4);
  p44 = ADD(p44, FMADD(z16, ADD(t62, t63), FMADD(z1_16,
      ADD(t64, t65), ADD(ADD(t60, t61), t66))));
  AVX512_ADD_B(4)
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 7, 3, 16, ISA_GENERIC)
{
  MD(float, atoutput, [A][xc.oc3 * xc.O2][Tz][V], toutput);
  MD(float, atoutputa, [A - K + 1][V], toutputa);

#undef P
#undef T
#define T(_h) atoutput[_h][0][0][_V]
#define P(_h) atoutputa[_h][_V]

  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z8 = 8.0f;
  const float z16 = 16.0f;
  const float z1_2 = 1.0f / 2.0f;
  const float z1_4 = 1.0f / 4.0f;
  const float z1_8 = 1.0f / 8.0f;
  const float z1_16 = 1.0f / 16.0f;
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    P(0) = T(0) + T(1) + T(2) + T(3) + T(4) + T(5);
    P(1) = T(0) - T(1) + z2 * (T(2) - T(3)) + z1_2 * (T(4) - T(5));
    P(2) = T(0) + T(1) + z4 * (T(2) + T(3)) + z1_4 * (T(4) + T(5));
    P(3) = T(0) - T(1) + z8 * (T(2) - T(3)) + z1_8 * (T(4) - T(5));
    P(4) = T(0) + T(1) + z16 * (T(2) + T(3)) + z1_16 * (T(4) + T(5)) + T(6);
  }
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 7, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  MD(float, atoutput, [A][xc.oc3 * xc.O2][Tz][V], toutput);
  MD(float, atoutputa, [A - K + 1][V], toutputa);

#undef P
#undef T
#define T(_h) atoutput[_h][0][0]
#define P(_h) atoutputa[_h]

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z8 = _mm512_set_ps(IMM_BCAST16(8.0f));
  __m512 z16 = _mm512_set_ps(IMM_BCAST16(16.0f));
  __m512 z1_2 = _mm512_set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m512 z1_4 = _mm512_set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m512 z1_8 = _mm512_set_ps(IMM_BCAST16(1.0f / 8.0f));
  __m512 z1_16 = _mm512_set_ps(IMM_BCAST16(1.0f / 16.0f));

  __m512 t0 = _mm512_load_ps(T(0));
  __m512 t1 = _mm512_load_ps(T(1));
  __m512 t2 = _mm512_load_ps(T(2));
  __m512 t3 = _mm512_load_ps(T(3));
  __m512 t4 = _mm512_load_ps(T(4));
  __m512 t5 = _mm512_load_ps(T(5));
  __m512 t6 = _mm512_load_ps(T(6));

  __m512 p0 = ADD(ADD(ADD(ADD(ADD(t0, t1), t2), t3), t4), t5);
  __m512 p1 = ADD(FMADD(z2, SUB(t2, t3), t0), FMSUB(z1_2, SUB(t4, t5), t1));
  __m512 p2 = ADD(FMADD(z4, ADD(t2, t3), t0), FMADD(z1_4, ADD(t4, t5), t1));
  __m512 p3 = ADD(FMADD(z8, SUB(t2, t3), t0), FMSUB(z1_8, SUB(t4, t5), t1));
  __m512 p4 = ADD(FMADD(z16, ADD(t2, t3), ADD(t0, t1)), FMADD(z1_16, ADD(t4, t5), t6));

  if (stream_out) {
    _mm512_stream_ps(P(0), p0);
    _mm512_stream_ps(P(1), p1);
    _mm512_stream_ps(P(2), p2);
    _mm512_stream_ps(P(3), p3);
    _mm512_stream_ps(P(4), p4);
  } else {
    _mm512_store_ps(P(0), p0);
    _mm512_store_ps(P(1), p1);
    _mm512_store_ps(P(2), p2);
    _mm512_store_ps(P(3), p3);
    _mm512_store_ps(P(4), p4);
  }
}

#define GENERIC_CALCULATE_TILE_7(z, n, nil)                       \
  P(n, 0) = T(n, 0) + T(n, 1) + T(n, 2) + T(n, 3) + T(n, 4)       \
      + T(n, 5);                                                  \
  if (with_bias_) P(n, 0) += B;                                   \
  P(n, 1) = T(n, 0) - T(n, 1) + z2 * (T(n, 2) - T(n, 3))          \
      + z1_2 * (T(n, 4) - T(n,5));                                \
  if (with_bias_) P(n, 1) += B;                                   \
  P(n, 2) = T(n, 0) + T(n, 1) + z4 * (T(n, 2) + T(n, 3))          \
      + z1_4 * (T(n, 4) + T(n,5));                                \
  if (with_bias_) P(n, 2) += B;                                   \
  P(n, 3) = T(n, 0) - T(n, 1) + z8 * (T(n, 2) - T(n, 3))          \
      + z1_8 * (T(n, 4) - T(n,5));                                \
  if (with_bias_) P(n, 3) += B;                                   \
  P(n, 4) = T(n, 0) + T(n, 1) + z16 * (T(n, 2) + T(n, 3))         \
      + z1_16 * (T(n, 4) + T(n,5)) + T(n, 6);                     \
  if (with_bias_) P(n, 4) += B;

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 7, 3, 16, ISA_GENERIC)
{
  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    if (_wOA_end == -1) {
      MD(float, aoutput, [A - K + 1][A - K + 1][16], output);
      return &aoutput[_h][_w][_V];
    } else {
      MD(float, aoutput, [xc.oh][xc.ow][16], output);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
        return &dummy[_V];
      else
        return &aoutput[_h][_w][_V];
    }
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]

  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z8 = 8.0f;
  const float z16 = 16.0f;
  const float z1_2 = 1.0f / 2.0f;
  const float z1_4 = 1.0f / 4.0f;
  const float z1_8 = 1.0f / 8.0f;
  const float z1_16 = 1.0f / 16.0f;

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    BOOST_PP_REPEAT(5, GENERIC_CALCULATE_TILE_7, nil)
  }
}

#define AVX512_BH_CALCULATE_TILE_7(z, n, nil)                       \
  t0 = _mm512_load_ps(T(n, 0));                                     \
  t1 = _mm512_load_ps(T(n, 1));                                     \
  t2 = _mm512_load_ps(T(n, 2));                                     \
  t3 = _mm512_load_ps(T(n, 3));                                     \
  t4 = _mm512_load_ps(T(n, 4));                                     \
  t5 = _mm512_load_ps(T(n, 5));                                     \
  t6 = _mm512_load_ps(T(n, 6));                                     \
                                                                    \
  p0 = ADD(ADD(ADD(ADD(ADD(t0, t1), t2), t3), t4), t5);             \
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);                     \
  p1 = ADD(FMADD(z2, SUB(t2, t3), t0), FMSUB(z1_2, SUB(t4, t5), t1)); \
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);                       \
  p2 = ADD(FMADD(z4, ADD(t2, t3), t0), FMADD(z1_4, ADD(t4, t5), t1)); \
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);                       \
  p3 = ADD(FMADD(z8, SUB(t2, t3), t0), FMSUB(z1_8, SUB(t4, t5), t1)); \
  if (with_bias_) p3 = ADD(p3, *(__m512*)bias);                       \
  p4 = ADD(FMADD(z16, ADD(t2, t3), ADD(t0, t1)), FMADD(z1_16, ADD(t4, t5), t6)); \
  if (with_bias_) p4 = ADD(p4, *(__m512*)bias);                       \
                                                                      \
  _mm512_store_ps(P(n,0), p0);                                        \
  _mm512_store_ps(P(n,1), p1);                                        \
  _mm512_store_ps(P(n,2), p2);                                        \
  _mm512_store_ps(P(n,3), p3);                                        \
  _mm512_store_ps(P(n,4), p4);

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 7, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (_wOA_end == -1) {
      MD(float, aoutput, [A - K + 1][A - K + 1][16], output);
      return aoutput[_h][_w];
    } else {
      MD(float, aoutput, [xc.oh][xc.ow][16], output);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
        return dummy;
      else
        return aoutput[_h][_w];
    }
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z8 = _mm512_set_ps(IMM_BCAST16(8.0f));
  __m512 z16 = _mm512_set_ps(IMM_BCAST16(16.0f));
  __m512 z1_2 = _mm512_set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m512 z1_4 = _mm512_set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m512 z1_8 = _mm512_set_ps(IMM_BCAST16(1.0f / 8.0f));
  __m512 z1_16 = _mm512_set_ps(IMM_BCAST16(1.0f / 16.0f));

  __m512 t0, t1, t2, t3, t4, t5, t6, p0, p1, p2, p3, p4;

  BOOST_PP_REPEAT(5, AVX512_BH_CALCULATE_TILE_7, nil)
}


TRANS_OUPUT(float, 7, 3, 16, ISA_GENERIC);
TRANS_OUPUT(float, 7, 3, 16, ISA_SKX_AVX512);

TRANS_OUTPUTA_TH(float, 7, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_TH(float, 7, 3, 16, ISA_SKX_AVX512);

TRANS_OUTPUTA_BH(float, 7, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_BH(float, 7, 3, 16, ISA_SKX_AVX512);

} // namespace euler
