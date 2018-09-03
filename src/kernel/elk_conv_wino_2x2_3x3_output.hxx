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

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3>::
__trans_output(elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
    float *bias, int hOA_end, int wOA_end) {

  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, 16);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#undef OP
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3;
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
  MATRIX_DEF(4, 4);

  c0 = ADD(ADD(t10, t11), t12);
  c1 = ADD(ADD(t20, t21), t22);
  c2 = SUB(ADD(t12, t13), t11);
  c3 = SUB(ADD(t22, t23), t21);

  __m512 p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
  if (with_bias) p00 = ADD(p00, *(__m512*)bias);
  __m512 p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
  if (with_bias) p10 = ADD(p10, *(__m512*)bias);
  __m512 p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
  if (with_bias) p01 = ADD(p01, *(__m512*)bias);
  __m512 p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);
  if (with_bias) p11 = ADD(p11, *(__m512*)bias);

#undef OP
#define p_(m, n) p##m##n
#define OP(m,n) _mm512_store_ps(P(m, n), p_(m, n))
  MATRIX_DEF(2, 2);
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {
  ENABLE_AVX512F();

  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) &md4(atoutput, _h, 0, 0, 0)
#define P(_h) &md2(atoutputa, _h, 0)

  __m512 t0 = _mm512_load_ps(T(0));
  __m512 t1 = _mm512_load_ps(T(1));
  __m512 t2 = _mm512_load_ps(T(2));
  __m512 t3 = _mm512_load_ps(T(3));

  __m512 p0 = ADD(ADD(t0, t1), t2);
  __m512 p1 = SUB(ADD(t2, t3), t1);

  if (stream_out) {
    _mm512_stream_ps(P(1), p0);
    _mm512_stream_ps(P(1), p1);
  } else {
    _mm512_store_ps(P(0), p0);
    _mm512_store_ps(P(1), p1);
  }
}



// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {

  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput,_h,_w, 0);
    } else {
    MD3(float, aoutput, output, xc.oh, xc.ow, 16);
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

 // __m512 c0, c1, c2, c3;

  __m512 t0, t1, t2, t3, p0, p1;
  t0 = _mm512_load_ps(T(0,0));
  t1 = _mm512_load_ps(T(0,1));
  t2 = _mm512_load_ps(T(0,2));
  t3 = _mm512_load_ps(T(0,3));

  p0 = ADD(ADD(t0, t1), t2);
  if (with_bias) p0 = ADD(p0, *(__m512*)bias); p1 = SUB(ADD(t2, t3), t1);
  if (with_bias) p1 = ADD(p1, *(__m512*)bias);

  _mm512_store_ps(P(0,0), p0);
  _mm512_store_ps(P(0,1), p1);

  t0 = _mm512_load_ps(T(1,0));
  t1 = _mm512_load_ps(T(1,1));
  t2 = _mm512_load_ps(T(1,2));
  t3 = _mm512_load_ps(T(1,3));

  p0 = ADD(ADD(t0, t1), t2);
  if (with_bias) p0 = ADD(p0, *(__m512*)bias); p1 = SUB(ADD(t2, t3), t1);
  if (with_bias) p1 = ADD(p1, *(__m512*)bias);

  _mm512_store_ps(P(1,0), p0);
  _mm512_store_ps(P(1,1), p1);

}

} // namespace euler
