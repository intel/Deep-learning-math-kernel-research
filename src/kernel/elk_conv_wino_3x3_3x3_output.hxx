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


// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 5, 3, 16, ISA_GENERIC)
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
#define C(n) c##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]
  float c0[16], c1[16], c2[16], c3[16], c4[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(0, 0) + T(0, 1) + T(0, 2) + T(0, 3);
    C(1) = T(1, 0) + T(1, 1) + T(1, 2) + T(1, 3);
    C(2) = T(2, 0) + T(2, 1) + T(2, 2) + T(2, 3);
    C(3) = T(3, 0) + T(3, 1) + T(3, 2) + T(3, 3);
    C(4) = T(4, 0) + T(4, 1) + T(4, 2) + T(4, 3);
    P(0, 0) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 0) += B;
    P(1, 0) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 0) += B;
    P(2, 0) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 0) += B;

    C(0) = T(0, 2) - T(0, 1) + 2 * T(0, 3);
    C(1) = T(1, 2) - T(1, 1) + 2 * T(1, 3);
    C(2) = T(2, 2) - T(2, 1) + 2 * T(2, 3);
    C(3) = T(3, 2) - T(3, 1) + 2 * T(3, 3);
    C(4) = T(4, 2) - T(4, 1) + 2 * T(4, 3);
    P(0, 1) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 1) += B;
    P(1, 1) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 1) += B;
    P(2, 1) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 1) += B;

    C(0) = T(0, 1) + T(0, 2) + 4 * T(0, 3) + T(0, 4);
    C(1) = T(1, 1) + T(1, 2) + 4 * T(1, 3) + T(1, 4);
    C(2) = T(2, 1) + T(2, 2) + 4 * T(2, 3) + T(2, 4);
    C(3) = T(3, 1) + T(3, 2) + 4 * T(3, 3) + T(3, 4);
    C(4) = T(4, 1) + T(4, 2) + 4 * T(4, 3) + T(4, 4);
    P(0, 2) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 2) += B;
    P(1, 2) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 2) += B;
    P(2, 2) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 2) += B;
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
  if (with_bias_)                                                              \
    p0##n = ADD(p0##n, *(__m512*)bias);                                        \
  __m512 p1##n = FMADD(z2, c3, SUB(c2, c1));                                   \
  if (with_bias_)                                                              \
    p1##n = ADD(p1##n, *(__m512*)bias);                                        \
  __m512 p2##n = FMADD(z4, c3, ADD(ADD(c1, c2), c4));                          \
  if (with_bias_)                                                              \
    p2##n = ADD(p2##n, *(__m512*)bias);

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512)

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

  __m512 c0, c1, c2, c3, c4;
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



// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 5, 3, 16, ISA_GENERIC)
{
  // TODO
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 5, 3, 16, ISA_SKX_AVX512)
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

// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_GENERIC)
{
  // TODO
}

// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_SKX_AVX512)
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

  // __m512 c0, c1, c2, c3, c4;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  __m512 t0, t1, t2, t3, t4, p0, p1, p2;
  t0 = _mm512_load_ps(T(0,0));
  t1 = _mm512_load_ps(T(0,1));
  t2 = _mm512_load_ps(T(0,2));
  t3 = _mm512_load_ps(T(0,3));
  t4 = _mm512_load_ps(T(0,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(0,0), p0);
  _mm512_store_ps(P(0,1), p1);
  _mm512_store_ps(P(0,2), p2);

  t0 = _mm512_load_ps(T(1,0));
  t1 = _mm512_load_ps(T(1,1));
  t2 = _mm512_load_ps(T(1,2));
  t3 = _mm512_load_ps(T(1,3));
  t4 = _mm512_load_ps(T(1,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(1,0), p0);
  _mm512_store_ps(P(1,1), p1);
  _mm512_store_ps(P(1,2), p2);

  t0 = _mm512_load_ps(T(2,0));
  t1 = _mm512_load_ps(T(2,1));
  t2 = _mm512_load_ps(T(2,2));
  t3 = _mm512_load_ps(T(2,3));
  t4 = _mm512_load_ps(T(2,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(2,0), p0);
  _mm512_store_ps(P(2,1), p1);
  _mm512_store_ps(P(2,2), p2);
}


TRANS_OUPUT(float, 5, 3, 16, ISA_GENERIC);
TRANS_OUPUT(float, 5, 3, 16, ISA_SKX_AVX512);

TRANS_OUTPUTA_TH(float, 5, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_TH(float, 5, 3, 16, ISA_SKX_AVX512);

TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_SKX_AVX512);

} // namespace euler
