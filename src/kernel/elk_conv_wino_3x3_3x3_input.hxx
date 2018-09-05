#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

#define GENERIC_CALCULATE_I(n)                                                  \
  T(0, n) = C(1) + z2 * C(2) - C(3);                                            \
  T(1, n) = z3 * C(2) - z2 * C(1) - C(3);                                       \
  T(2, n) = z2 * C(1) + C(2) - C(3);                                            \
  T(3, n) = C(1) - C(3);                                                        \
  T(4, n) = - z2 * C(1) + C(2) + z2 * C(3);

template <>
class convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3> {
  template<typename Type, int ...configs>
    friend class convolution_winograd_kernel_base;
protected:
  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int V = 16;
  constexpr static int A = 5;
  constexpr static int K = 3;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end);

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int wA, int hA_start, int hA_end, int wA_start,
      int _wA_end);

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
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3>::__trans_input(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  ENABLE_AVX512F();

  // Inputs
  __m<V> f00, f01, f02, f03, f04, f10, f11, f12, f13, f14, f20, f21, f22, f23,
      f24, f30, f31, f32, f33, f34, f40, f41, f42, f43, f44;
  // Cache
  __m<V> c1, c2, c3;
  // Outputs
  __m<V> t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23,
      t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;

  __m<V> z0 = _mm<V>::setzero_ps();
  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z3 = _mm<V>::set_ps(IMM_BCAST16(3.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));
  __m<V> z6 = _mm<V>::set_ps(IMM_BCAST16(6.0f));

  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, V);
      return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)
  MATRIX_DEF(5, 4);

  c1 = FMADD(z2, SUB(f12, f10), SUB(f11, f13));
  c2 = FMADD(z2, SUB(f22, f20), SUB(f21, f23));
  c3 = FMADD(z2, SUB(f32, f30), SUB(f31, f33));
  t00 = ADD(
      FMADD(z4, SUB(f00, f02), c1), FMSUB(z2, ADD(SUB(f03, f01), c2), c3));
  _mm<V>::store_ps(T(0, 0), t00);
  t10 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm<V>::store_ps(T(1, 0), t10);
  t20 = FMADD(z2, c1, SUB(c2, c3));
  _mm<V>::store_ps(T(2, 0), t20);
  t30 = SUB(c1, c3);
  _mm<V>::store_ps(T(3, 0), t30);
  t40 = FMADD(z2, ADD(SUB(f40, f42), SUB(c3, c1)), ADD(SUB(f43, f41), c2));
  _mm<V>::store_ps(T(4, 0), t40);

  c1 = FMSUB(z3, f12, FMADD(z2, f11, f13));
  c2 = FMSUB(z3, f22, FMADD(z2, f21, f23));
  c3 = FMSUB(z3, f32, FMADD(z2, f31, f33));
  t01 = ADD(FMSUB(z4, f01, FMADD(z6, f02, c3)), FMADD(z2, ADD(f03, c2), c1));
  _mm<V>::store_ps(T(0, 1), t01);
  t11 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm<V>::store_ps(T(1, 1), t11);
  t21 = FMADD(z2, c1, SUB(c2, c3));
  _mm<V>::store_ps(T(2, 1), t21);
  t31 = SUB(c1, c3);
  _mm<V>::store_ps(T(3, 1), t31);
  t41 = ADD(FMSUB(z2, ADD(f41, SUB(c3, c1)), FMSUB(z3, f42, c2)), f43);
  _mm<V>::store_ps(T(4, 1), t41);

  c1 = FMADD(z2, f11, SUB(f12, f13));
  c2 = FMADD(z2, f21, SUB(f22, f23));
  c3 = FMADD(z2, f31, SUB(f32, f33));
  t02 = ADD(FMSUB(z2, ADD(f03, SUB(c2, f02)), FMADD(z4, f01, c3)), c1);
  _mm<V>::store_ps(T(0, 2), t02);
  t12 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm<V>::store_ps(T(1, 2), t12);
  t22 = FMADD(z2, c1, SUB(c2, c3));
  _mm<V>::store_ps(T(2, 2), t22);
  t32 = SUB(c1, c3);
  _mm<V>::store_ps(T(3, 2), t32);
  t42 = ADD(FMADD(z2, SUB(c3, ADD(f41, c1)), c2), SUB(f43, f42));
  _mm<V>::store_ps(T(4, 2), t42);

  c1 = SUB(f11, f13);
  c2 = SUB(f21, f23);
  c3 = SUB(f31, f33);
  t03 = FMADD(z2, ADD(SUB(f03, f01), c2), SUB(c1, c3));
  _mm<V>::store_ps(T(0, 3), t03);
  t13 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm<V>::store_ps(T(1, 3), t13);
  t23 = FMADD(z2, c1, SUB(c2, c3));
  _mm<V>::store_ps(T(2, 3), t23);
  t33 = SUB(c1, c3);
  _mm<V>::store_ps(T(3, 3), t33);
  t43 = ADD(FMADD(z2, SUB(c3, c1), c2), SUB(f43, f41));
  _mm<V>::store_ps(T(4, 3), t43);

  f04 = F(0, 4);
  f14 = F(1, 4);
  f24 = F(2, 4);
  f34 = F(3, 4);
  f44 = F(4, 4);

  c1 = ADD(f12, FMSUB(z2, SUB(f13, f11), f14));
  c2 = ADD(f22, FMSUB(z2, SUB(f23, f21), f24));
  c3 = ADD(f32, FMSUB(z2, SUB(f33, f31), f34));

  t04 = ADD(
      FMADD(z4, SUB(f01, f03), c1), FMSUB(z2, ADD(SUB(f04, f02), c2), c3));
  _mm<V>::store_ps(T(0, 4), t04);
  t14 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm<V>::store_ps(T(1, 4), t14);
  t24 = FMADD(z2, c1, SUB(c2, c3));
  _mm<V>::store_ps(T(2, 4), t24);
  t34 = SUB(c1, c3);
  _mm<V>::store_ps(T(3, 4), t34);
  t44 = ADD(FMADD(z2, ADD(SUB(f41, f43), SUB(c3, c1)), SUB(f44, f42)), c2);
  _mm<V>::store_ps(T(4, 4), t44);
}

template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 5, 3>::
__trans_inputa(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  ENABLE_AVX512F();

  // Inputs
  __m<V> f00, f01, f02, f03, f04, f10, f11, f12, f13, f14, f20, f21, f22, f23,
      f24, f30, f31, f32, f33, f34, f40, f41, f42, f43, f44;
  // Cache
  __m<V> c1, c2, c3;
  // Outputs
  __m<V> t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23,
      t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;

  __m<V> z0 = _mm<V>::setzero_ps();
  __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));
  __m<V> z3 = _mm<V>::set_ps(IMM_BCAST16(3.0f));
  __m<V> z4 = _mm<V>::set_ps(IMM_BCAST16(4.0f));
  __m<V> z6 = _mm<V>::set_ps(IMM_BCAST16(6.0f));

  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, V);
      return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end || _w > wT_end))
        return z0;
      else
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[h][w]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)
  MATRIX_DEF(5, 4);

  switch (wA) {
  case 0:
    c1 = FMADD(z2, SUB(f12, f10), SUB(f11, f13));
    c2 = FMADD(z2, SUB(f22, f20), SUB(f21, f23));
    c3 = FMADD(z2, SUB(f32, f30), SUB(f31, f33));

    t00 = ADD(
        FMADD(z4, SUB(f00, f02), c1), FMSUB(z2, ADD(SUB(f03, f01), c2), c3));
    _mm<V>::store_ps(T(0, 0), t00);
    t10 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm<V>::store_ps(T(1, 0), t10);
    t20 = FMADD(z2, c1, SUB(c2, c3));
    _mm<V>::store_ps(T(2, 0), t20);
    t30 = SUB(c1, c3);
    _mm<V>::store_ps(T(3, 0), t30);
    t40 = FMADD(z2, ADD(SUB(f40, f42), SUB(c3, c1)), ADD(SUB(f43, f41), c2));
    _mm<V>::store_ps(T(4, 0), t40);

    break;
  case 1:
    c1 = FMSUB(z3, f12, FMADD(z2, f11, f13));
    c2 = FMSUB(z3, f22, FMADD(z2, f21, f23));
    c3 = FMSUB(z3, f32, FMADD(z2, f31, f33));

    t01 = ADD(
        FMSUB(z4, f01, FMADD(z6, f02, c3)), FMADD(z2, ADD(f03, c2), c1));
    _mm<V>::store_ps(T(0, 1), t01);
    t11 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm<V>::store_ps(T(1, 1), t11);
    t21 = FMADD(z2, c1, SUB(c2, c3));
    _mm<V>::store_ps(T(2, 1), t21);
    t31 = SUB(c1, c3);
    _mm<V>::store_ps(T(3, 1), t31);
    t41 = ADD(FMSUB(z2, ADD(f41, SUB(c3, c1)), FMSUB(z3, f42, c2)), f43);
    _mm<V>::store_ps(T(4, 1), t41);

    break;
  case 2:
    c1 = FMADD(z2, f11, SUB(f12, f13));
    c2 = FMADD(z2, f21, SUB(f22, f23));
    c3 = FMADD(z2, f31, SUB(f32, f33));

    t02 = ADD(FMSUB(z2, ADD(f03, SUB(c2, f02)), FMADD(z4, f01, c3)), c1);
    _mm<V>::store_ps(T(0, 2), t02);
    t12 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm<V>::store_ps(T(1, 2), t12);
    t22 = FMADD(z2, c1, SUB(c2, c3));
    _mm<V>::store_ps(T(2, 2), t22);
    t32 = SUB(c1, c3);
    _mm<V>::store_ps(T(3, 2), t32);
    t42 = ADD(FMADD(z2, SUB(c3, ADD(f41, c1)), c2), SUB(f43, f42));
    _mm<V>::store_ps(T(4, 2), t42);

    break;
  case 3:
    c1 = SUB(f11, f13);
    c2 = SUB(f21, f23);
    c3 = SUB(f31, f33);

    t03 = FMADD(z2, ADD(SUB(f03, f01), c2), SUB(c1, c3));
    _mm<V>::store_ps(T(0, 3), t03);
    t13 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm<V>::store_ps(T(1, 3), t13);
    t23 = FMADD(z2, c1, SUB(c2, c3));
    _mm<V>::store_ps(T(2, 3), t23);
    t33 = SUB(c1, c3);
    _mm<V>::store_ps(T(3, 3), t33);
    t43 = ADD(FMADD(z2, SUB(c3, c1), c2), SUB(f43, f41));
    _mm<V>::store_ps(T(4, 3), t43);

    break;
  case 4:
    f04 = F(0, 4);
    f14 = F(1, 4);
    f24 = F(2, 4);
    f34 = F(3, 4);
    f44 = F(4, 4);

    c1 = ADD(f12, FMSUB(z2, SUB(f13, f11), f14));
    c2 = ADD(f22, FMSUB(z2, SUB(f23, f21), f24));
    c3 = ADD(f32, FMSUB(z2, SUB(f33, f31), f34));

    t04 = ADD(
        FMADD(z4, SUB(f01, f03), c1), FMSUB(z2, ADD(SUB(f04, f02), c2), c3));
    _mm<V>::store_ps(T(0, 4), t04);
    t14 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm<V>::store_ps(T(1, 4), t14);
    t24 = FMADD(z2, c1, SUB(c2, c3));
    _mm<V>::store_ps(T(2, 4), t24);
    t34 = SUB(c1, c3);
    _mm<V>::store_ps(T(3, 4), t34);
    t44 = ADD(FMADD(z2, ADD(SUB(f41, f43), SUB(c3, c1)), SUB(f44, f42)), c2);
    _mm<V>::store_ps(T(4, 4), t44);

    break;
  }
}
} // namespace euler
