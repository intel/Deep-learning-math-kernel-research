#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
template <>
class convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3> {
  template <typename Type, int ...configs>
    friend class convolution_winograd_kernel_base;
protected:
  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int V = 16;
  constexpr static int A = 6;
  constexpr static int K = 3;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end);

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int wA, int hA_start, int hA_end, int wA_start,
      int wA_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<float> &xc, float *output,
      float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<float> &xc, float *toutputa,
      float *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<float> &xc, float *output,
      float aoutputa[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]);
};

template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3>::__trans_input( elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {

  // Inputs
  __m512 f00, f01, f02, f03, f04, f05,
         f10, f11, f12, f13, f14, f15,
         f20, f21, f22, f23, f24, f25,
         f30, f31, f32, f33, f34, f35,
         f40, f41, f42, f43, f44, f45,
         f50, f51, f52, f53, f54, f55;
  // Cache
  __m512 c1, c2, c3, c4;
  // Buffer
  __m512 a00, a01, a02, a03;
  __m512 b00, b01, b02, b03, b04, b05, b06, b07;
  __m512 d00, d01, d02, d03;
  // Outputs
  __m512 t00, t01, t02, t03, t04, t05,
         t10, t11, t12, t13, t14, t15,
         t20, t21, t22, t23, t24, t25,
         t30, t31, t32, t33, t34, t35,
         t40, t41, t42, t43, t44, t45,
         t50, t51, t52, t53, t54, t55;

  __m512 z0 = _mm512_setzero_ps();

  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return _mm512_load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else
        return _mm512_load_ps(&md3(ainput, _h, _w, 0));
    }
  };

#undef F
#undef T
#undef f
#undef OP
#undef ISTORE

#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)
#define ISTORE(i, j) _mm512_store_ps(T(i, j), t##i##j)

  VECTOR_DEF(M6, ME3);

  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z5 = _mm512_set_ps(IMM_BCAST16(5.0f));

  c1 = FMSUB(z4, f10, (FMSUB(z5, f12, f14)));
  c2 = FMSUB(z4, f20, (FMSUB(z5, f22, f24)));
  c3 = FMSUB(z4, f30, (FMSUB(z5, f32, f34)));
  c4 = FMSUB(z4, f40, (FMSUB(z5, f42, f44)));

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z16 = _mm512_set_ps(IMM_BCAST16(16.0f));
  __m512 z20 = _mm512_set_ps(IMM_BCAST16(20.0f));

  t00 = FMADD(z16, f00, FNMADD(z20, f02, FMADD(z4, f04, FNMADD(z5, c2, c4))));
  ISTORE(0, 0);
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t10 = SUB(a01, a00);
  ISTORE(1, 0);
  t20 = ADD(a01, a00);
  ISTORE(2, 0);
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t30 = FNMSUB(z2, a02, a03);
  ISTORE(3, 0);
  t40 = FMSUB(z2, a02, a03);
  ISTORE(4, 0);
  t50 = FMADD(z4, ADD(c1, f50), FNMADD(z5, ADD(c3, f52), f54));
  ISTORE(5, 0);

  VECTOR_DEF(M6, MO2);

  b00 = FMSUB(z4, f11, f13);
  b01 = FMSUB(z4, f21, f23);
  b02 = FMSUB(z4, f31, f33);
  b03 = FMSUB(z4, f41, f43);

  b04 = FMSUB(z4, f12, f14);
  b05 = FMSUB(z4, f22, f24);
  b06 = FMSUB(z4, f32, f34);
  b07 = FMSUB(z4, f42, f44);

  c1 = ADD(b00, b04);
  c2 = ADD(b01, b05);
  c3 = ADD(b02, b06);
  c4 = ADD(b03, b07);

  d00 = FMSUB(z4, f01, f03);
  d01 = FMSUB(z4, f02, f04);
  t01 = FNMADD(z4, d00, FNMADD(z4, d01, FMSUB(z5, c2, c4)));
  ISTORE(0, 1);
  a00 = FMSUB(z4, c1, c3);
  a01 = FMSUB(z4, c2, c4);
  t11 = ADD(a01, a00);
  ISTORE(1, 1);
  t21 = SUB(a01, a00);
  ISTORE(2, 1);
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t31 = FMADD(z2, a02, a03);
  ISTORE(3, 1);
  t41 = FNMADD(z2, a02, a03);
  ISTORE(4, 1);
  d02 = FMSUB(z4, f51, f53);
  d03 = FNMADD(z4, f52, f54);
  t51 = FNMADD(z4, c1, FMADD(z5, c3, SUB(d03, d02)));
  ISTORE(5, 1);

  c1 = SUB(b00, b04);
  c2 = SUB(b01, b05);
  c3 = SUB(b02, b06);
  c4 = SUB(b03, b07);

  t02 = FMADD(z4, d00, FNMADD(z4, d01, FNMADD(z5, c2, c4)));
  ISTORE(0, 2);
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t12 = SUB(a01, a00);
  ISTORE(1, 2);
  t22 = ADD(a01, a00);
  ISTORE(2, 2);
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t32 = FNMADD(z2, a02, a03);
  ISTORE(3, 2);
  t42 = FMADD(z2, a02, a03);
  ISTORE(4, 2);
  t52 = FMADD(z4, c1, FNMADD(z5, c3, ADD(d03, d02)));
  ISTORE(5, 2);

  b00 = SUB(f11, f13);
  b01 = SUB(f21, f23);
  b02 = SUB(f31, f33);
  b03 = SUB(f41, f43);

  b04 = SUB(f12, f14);
  b05 = SUB(f22, f24);
  b06 = SUB(f32, f34);
  b07 = SUB(f42, f44);

  c1 = FMADD(z2, b00, b04);
  c2 = FMADD(z2, b01, b05);
  c3 = FMADD(z2, b02, b06);
  c4 = FMADD(z2, b03, b07);

  __m512 z8 = _mm512_set_ps(IMM_BCAST16(8.0f));

  d00 = SUB(f04, f02);
  d01 = SUB(f03, f01);
  t03 = FMADD(z8, d01, FMADD(z4, d00, FMSUB(z5, c2, c4)));
  ISTORE(0, 3);
  a00 = FMSUB(z4, c1, c3);
  a01 = FMSUB(z4, c2, c4);
  t13 = ADD(a01, a00);
  ISTORE(1, 3);
  t23 = SUB(a01, a00);
  ISTORE(2, 3);
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t33 = FMADD(z2, a02, a03);
  ISTORE(3, 3);
  t43 = FNMADD(z2, a02, a03);
  ISTORE(4, 3);
  d02 = SUB(f53, f51);
  d03 = SUB(f54, f52);
  t53 = FNMADD(z4, c1, FMADD(z5, c3, FMADD(z2, d02, d03)));
  ISTORE(5, 3);

  c1 = FMSUB(z2, b00, b04);
  c2 = FMSUB(z2, b01, b05);
  c3 = FMSUB(z2, b02, b06);
  c4 = FMSUB(z2, b03, b07);

  t04 = FNMADD(z8, d01, FMADD(z4, d00, FNMADD(z5, c2, c4)));
  ISTORE(0, 4);
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t14 = SUB(a01, a00);
  ISTORE(1, 4);
  t24 = ADD(a01, a00);
  ISTORE(2, 4);
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t34 = FNMADD(z2, a02, a03);
  ISTORE(3, 4);
  t44 = FMADD(z2, a02, a03);
  ISTORE(4, 4);
  t54 = FMADD(z4, c1, FNMADD(z5, c3, FNMADD(z2, d02, d03)));
  ISTORE(5, 4);

  VECTOR_DEF(M6, (5));

  c1 = FMADD(z4, f11, FNMADD(z5, f13, f15));
  c2 = FMADD(z4, f21, FNMADD(z5, f23, f25));
  c3 = FMADD(z4, f31, FNMADD(z5, f33, f35));
  c4 = FMADD(z4, f41, FNMADD(z5, f43, f45));

  t05 = FMADD(z4, FMADD(z4, f01, f05), FNMADD(z5, FMADD(z4, f03, c2), c4));
  ISTORE(0, 5);
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t15 = SUB(a01, a00);
  ISTORE(1, 5);
  t25 = ADD(a01, a00);
  ISTORE(2, 5);
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t35 = FNMADD(z2, a02, a03);
  ISTORE(3, 5);
  t45 = FMADD(z2, a02, a03);
  ISTORE(4, 5);
  t55 = FMADD(z4, ADD(c1, f51), FNMADD(z5, ADD(c3, f53), f55));
  ISTORE(5, 5);
}

template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 6, 3>::
__trans_inputa(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  // TODO
  el_error("Unimplemented");
}
}
