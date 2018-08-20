#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_4x4_3x3_input.hxx"

namespace euler {
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3>::
__trans_output(elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
    float *bias, int hOA_end, int wOA_end) {
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z8 = _mm512_set_ps(IMM_BCAST16(8.0f));

  // Inputs
  __m512 t00, t01, t02, t03, t04, t05,
         t10, t11, t12, t13, t14, t15,
         t20, t21, t22, t23, t24, t25,
         t30, t31, t32, t33, t34, t35,
         t40, t41, t42, t43, t44, t45,
         t50, t51, t52, t53, t54, t55;
  // Cache
  __m512 c0, c1, c2, c3;
  // Buffer
  __m512 a00, a01, a02, a03;
  __m512 b00, b01, b02, b03;
  __m512 d00, d01, d02, d03,
         d04, d05, d06, d07;
  // Outputs
  __m512 p00, p01, p02, p03,
         p10, p11, p12, p13,
         p20, p21, p22, p23,
         p30, p31, p32, p33;

  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;

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
#undef t
#undef OP
#undef ISTORE

#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)
#define t(m, n) t##m##n
#define OP(m,n) t(m,n) = _mm512_load_ps(T(m, n))
#define ISTORE(i, j) _mm512_store_ps(P(i, j), p##i##j)

  VECTOR_DEF(M6, M5);

  d00 = ADD(t11 ,t12);
  d01 = ADD(t21 ,t22);
  d02 = ADD(t31 ,t32);
  d03 = ADD(t41 ,t42);

  d04 = ADD(t13 ,t14);
  d05 = ADD(t23 ,t24);
  d06 = ADD(t33 ,t34);
  d07 = ADD(t43 ,t44);

  c0 = ADD(t10, ADD(d00, d04));
  c1 = ADD(t20, ADD(d01, d05));
  c2 = ADD(t30, ADD(d02, d06));
  c3 = ADD(t40, ADD(d03, d07));

  a00 = ADD(c0, c1);
  a01 = ADD(c2, c3);
  a02 = SUB(c0, c1);
  a03 = SUB(c2, c3);

  b00 = ADD(t01, t02);
  b01 = ADD(t03, t04);
  b02 = ADD(t51, t52);
  b03 = ADD(t53, t54);

  p00 = ADD(t00, ADD(b00, ADD(b01, ADD(a00, a01))));
  if (with_bias) p00 = ADD(p00, *(__m512*)bias);
  ISTORE(0, 0);
  p10 = FMADD(z2, a03, a02);
  if (with_bias) p10 = ADD(p10, *(__m512*)bias);
  ISTORE(1, 0);
  p20 = FMADD(z4, a01, a00);
  if (with_bias) p20 = ADD(p20, *(__m512*)bias);
  ISTORE(2, 0);
  p30 = FMADD(z8, a03, ADD(a02, ADD(t50, ADD(b02, b03))));
  if (with_bias) p30 = ADD(p30, *(__m512*)bias);
  ISTORE(3, 0);

  c0 = FMADD(z4, d04, d00);
  c1 = FMADD(z4, d05, d01);
  c2 = FMADD(z4, d06, d02);
  c3 = FMADD(z4, d07, d03);

  a00 = ADD(c0, c1);
  a01 = ADD(c2, c3);
  a02 = SUB(c0, c1);
  a03 = SUB(c2, c3);

  p02 = FMADD(z4, b01, ADD(b00, ADD(a00, a01)));
  if (with_bias) p02 = ADD(p02, *(__m512*)bias);
  ISTORE(0, 2);
  p12 = FMADD(z2, a03, a02);
  if (with_bias) p12 = ADD(p12, *(__m512*)bias);
  ISTORE(1, 2);
  p22 = FMADD(z4, a01, a00);
  if (with_bias) p22 = ADD(p22, *(__m512*)bias);
  ISTORE(2, 2);
  p32 = ADD(FMADD(z8, a03, a02), FMADD(z4, b03, b02));
  if (with_bias) p32 = ADD(p32, *(__m512*)bias);
  ISTORE(3, 2);

  d00 = SUB(t11, t12);
  d01 = SUB(t21, t22);
  d02 = SUB(t31, t32);
  d03 = SUB(t41, t42);

  d04 = SUB(t13, t14);
  d05 = SUB(t23, t24);
  d06 = SUB(t33, t34);
  d07 = SUB(t43, t44);

  c0 = FMADD(z2, d04, d00);
  c1 = FMADD(z2, d05, d01);
  c2 = FMADD(z2, d06, d02);
  c3 = FMADD(z2, d07, d03);

  a00 = ADD(c0, c1);
  a01 = ADD(c2, c3);
  a02 = SUB(c0, c1);
  a03 = SUB(c2, c3);

  b00 = SUB(t01, t02);
  b01 = SUB(t03, t04);
  b02 = SUB(t51, t52);
  b03 = SUB(t53, t54);

  p01 = ADD(FMADD(z2, b01, b00), ADD(a00, a01));
  if (with_bias) p01 = ADD(p01, *(__m512*)bias);
  ISTORE(0, 1);
  p11 = FMADD(z2, a03, a02);
  if (with_bias) p11 = ADD(p11, *(__m512*)bias);
  ISTORE(1, 1);
  p21 = FMADD(z4, a01, a00);
  if (with_bias) p21 = ADD(p21, *(__m512*)bias);
  ISTORE(2, 1);
  p31 = ADD(FMADD(z8, a03, a02), FMADD(z2, b03, b02));
  if (with_bias) p31 = ADD(p31, *(__m512*)bias);
  ISTORE(3, 1);

  VECTOR_DEF(M6, (5));

  c0 = ADD(FMADD(z8, d04, d00), t15);
  c1 = ADD(FMADD(z8, d05, d01), t25);
  c2 = ADD(FMADD(z8, d06, d02), t35);
  c3 = ADD(FMADD(z8, d07, d03), t45);

  a00 = ADD(c0, c1);
  a01 = ADD(c2, c3);
  a02 = SUB(c0, c1);
  a03 = SUB(c2, c3);

  p03 = ADD(FMADD(z8, b01, b00), ADD(t05, ADD(a00, a01)));
  if (with_bias) p03 = ADD(p03, *(__m512*)bias);
  ISTORE(0, 3);
  p13 = FMADD(z2, a03, a02);
  if (with_bias) p13 = ADD(p13, *(__m512*)bias);
  ISTORE(1, 3);
  p23 = FMADD(z4, a01, a00);
  if (with_bias) p23 = ADD(p23, *(__m512*)bias);
  ISTORE(2, 3);
  p33 = ADD(FMADD(z8, a03, a02), ADD(FMADD(z8, b03, b02), t55));
  if (with_bias) p33 = ADD(p33, *(__m512*)bias);
  ISTORE(3, 3);
}

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {  // TODO
  el_error("Unimplemented");
}

template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
  // TODO
  el_error("Unimplemented");
}
} // namespace euler
