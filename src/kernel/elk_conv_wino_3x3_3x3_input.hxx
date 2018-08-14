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

// template <const bool is_border_>
// Params:
//    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//    int _hT_start, int _hT_end, int _wT_start, int _wT_end
__TRANS_INPUT(float, 5, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  // Inputs
  __m512 f00, f01, f02, f03, f04, f10, f11, f12, f13, f14, f20, f21, f22, f23,
      f24, f30, f31, f32, f33, f34, f40, f41, f42, f43, f44;
  // Cache
  __m512 c1, c2, c3;
  // Outputs
  __m512 t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23,
      t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;

  __m512 z0 = _mm512_setzero_ps();
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z3 = _mm512_set_ps(IMM_BCAST16(3.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z6 = _mm512_set_ps(IMM_BCAST16(6.0f));

  auto f_cb = [&](int _h, int _w) {
    if (_wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return _mm512_load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border_
          && (_h < _hT_start || _w < _wT_start || _h > _hT_end
                 || _w > _wT_end))
        return z0;
      else
        return _mm512_load_ps(&md3(ainput, _h, _w, 0));
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
  _mm512_store_ps(T(0, 0), t00);
  t10 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm512_store_ps(T(1, 0), t10);
  t20 = FMADD(z2, c1, SUB(c2, c3));
  _mm512_store_ps(T(2, 0), t20);
  t30 = SUB(c1, c3);
  _mm512_store_ps(T(3, 0), t30);
  t40 = FMADD(z2, ADD(SUB(f40, f42), SUB(c3, c1)), ADD(SUB(f43, f41), c2));
  _mm512_store_ps(T(4, 0), t40);

  c1 = FMSUB(z3, f12, FMADD(z2, f11, f13));
  c2 = FMSUB(z3, f22, FMADD(z2, f21, f23));
  c3 = FMSUB(z3, f32, FMADD(z2, f31, f33));
  t01 = ADD(FMSUB(z4, f01, FMADD(z6, f02, c3)), FMADD(z2, ADD(f03, c2), c1));
  _mm512_store_ps(T(0, 1), t01);
  t11 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm512_store_ps(T(1, 1), t11);
  t21 = FMADD(z2, c1, SUB(c2, c3));
  _mm512_store_ps(T(2, 1), t21);
  t31 = SUB(c1, c3);
  _mm512_store_ps(T(3, 1), t31);
  t41 = ADD(FMSUB(z2, ADD(f41, SUB(c3, c1)), FMSUB(z3, f42, c2)), f43);
  _mm512_store_ps(T(4, 1), t41);

  c1 = FMADD(z2, f11, SUB(f12, f13));
  c2 = FMADD(z2, f21, SUB(f22, f23));
  c3 = FMADD(z2, f31, SUB(f32, f33));
  t02 = ADD(FMSUB(z2, ADD(f03, SUB(c2, f02)), FMADD(z4, f01, c3)), c1);
  _mm512_store_ps(T(0, 2), t02);
  t12 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm512_store_ps(T(1, 2), t12);
  t22 = FMADD(z2, c1, SUB(c2, c3));
  _mm512_store_ps(T(2, 2), t22);
  t32 = SUB(c1, c3);
  _mm512_store_ps(T(3, 2), t32);
  t42 = ADD(FMADD(z2, SUB(c3, ADD(f41, c1)), c2), SUB(f43, f42));
  _mm512_store_ps(T(4, 2), t42);

  c1 = SUB(f11, f13);
  c2 = SUB(f21, f23);
  c3 = SUB(f31, f33);
  t03 = FMADD(z2, ADD(SUB(f03, f01), c2), SUB(c1, c3));
  _mm512_store_ps(T(0, 3), t03);
  t13 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm512_store_ps(T(1, 3), t13);
  t23 = FMADD(z2, c1, SUB(c2, c3));
  _mm512_store_ps(T(2, 3), t23);
  t33 = SUB(c1, c3);
  _mm512_store_ps(T(3, 3), t33);
  t43 = ADD(FMADD(z2, SUB(c3, c1), c2), SUB(f43, f41));
  _mm512_store_ps(T(4, 3), t43);

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
  _mm512_store_ps(T(0, 4), t04);
  t14 = FMSUB(z3, c2, FMADD(z2, c1, c3));
  _mm512_store_ps(T(1, 4), t14);
  t24 = FMADD(z2, c1, SUB(c2, c3));
  _mm512_store_ps(T(2, 4), t24);
  t34 = SUB(c1, c3);
  _mm512_store_ps(T(3, 4), t34);
  t44 = ADD(FMADD(z2, ADD(SUB(f41, f43), SUB(c3, c1)), SUB(f44, f42)), c2);
  _mm512_store_ps(T(4, 4), t44);
}

// template <const bool is_border_>
// Params:
//   elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//   int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)
__TRANS_INPUTA(float, 5, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  // Inputs
  __m512 f00, f01, f02, f03, f04, f10, f11, f12, f13, f14, f20, f21, f22, f23,
      f24, f30, f31, f32, f33, f34, f40, f41, f42, f43, f44;
  // Cache
  __m512 c1, c2, c3;
  // Outputs
  __m512 t00, t01, t02, t03, t04, t10, t11, t12, t13, t14, t20, t21, t22, t23,
      t24, t30, t31, t32, t33, t34, t40, t41, t42, t43, t44;

  __m512 z0 = _mm512_setzero_ps();
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z3 = _mm512_set_ps(IMM_BCAST16(3.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  __m512 z6 = _mm512_set_ps(IMM_BCAST16(6.0f));

  auto f_cb = [&](int _h, int _w) {
    if (_wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return _mm512_load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border_
          && (_h < _hT_start || _w < _wT_start || _h > _hT_end || _w > _wT_end))
        return z0;
      else
        return _mm512_load_ps(&md3(ainput, _h, _w, 0));
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

  switch (_wA) {
  case 0:
    c1 = FMADD(z2, SUB(f12, f10), SUB(f11, f13));
    c2 = FMADD(z2, SUB(f22, f20), SUB(f21, f23));
    c3 = FMADD(z2, SUB(f32, f30), SUB(f31, f33));

    t00 = ADD(
        FMADD(z4, SUB(f00, f02), c1), FMSUB(z2, ADD(SUB(f03, f01), c2), c3));
    _mm512_store_ps(T(0, 0), t00);
    t10 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm512_store_ps(T(1, 0), t10);
    t20 = FMADD(z2, c1, SUB(c2, c3));
    _mm512_store_ps(T(2, 0), t20);
    t30 = SUB(c1, c3);
    _mm512_store_ps(T(3, 0), t30);
    t40 = FMADD(z2, ADD(SUB(f40, f42), SUB(c3, c1)), ADD(SUB(f43, f41), c2));
    _mm512_store_ps(T(4, 0), t40);

    break;
  case 1:
    c1 = FMSUB(z3, f12, FMADD(z2, f11, f13));
    c2 = FMSUB(z3, f22, FMADD(z2, f21, f23));
    c3 = FMSUB(z3, f32, FMADD(z2, f31, f33));

    t01 = ADD(
        FMSUB(z4, f01, FMADD(z6, f02, c3)), FMADD(z2, ADD(f03, c2), c1));
    _mm512_store_ps(T(0, 1), t01);
    t11 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm512_store_ps(T(1, 1), t11);
    t21 = FMADD(z2, c1, SUB(c2, c3));
    _mm512_store_ps(T(2, 1), t21);
    t31 = SUB(c1, c3);
    _mm512_store_ps(T(3, 1), t31);
    t41 = ADD(FMSUB(z2, ADD(f41, SUB(c3, c1)), FMSUB(z3, f42, c2)), f43);
    _mm512_store_ps(T(4, 1), t41);

    break;
  case 2:
    c1 = FMADD(z2, f11, SUB(f12, f13));
    c2 = FMADD(z2, f21, SUB(f22, f23));
    c3 = FMADD(z2, f31, SUB(f32, f33));

    t02 = ADD(FMSUB(z2, ADD(f03, SUB(c2, f02)), FMADD(z4, f01, c3)), c1);
    _mm512_store_ps(T(0, 2), t02);
    t12 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm512_store_ps(T(1, 2), t12);
    t22 = FMADD(z2, c1, SUB(c2, c3));
    _mm512_store_ps(T(2, 2), t22);
    t32 = SUB(c1, c3);
    _mm512_store_ps(T(3, 2), t32);
    t42 = ADD(FMADD(z2, SUB(c3, ADD(f41, c1)), c2), SUB(f43, f42));
    _mm512_store_ps(T(4, 2), t42);

    break;
  case 3:
    c1 = SUB(f11, f13);
    c2 = SUB(f21, f23);
    c3 = SUB(f31, f33);

    t03 = FMADD(z2, ADD(SUB(f03, f01), c2), SUB(c1, c3));
    _mm512_store_ps(T(0, 3), t03);
    t13 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm512_store_ps(T(1, 3), t13);
    t23 = FMADD(z2, c1, SUB(c2, c3));
    _mm512_store_ps(T(2, 3), t23);
    t33 = SUB(c1, c3);
    _mm512_store_ps(T(3, 3), t33);
    t43 = ADD(FMADD(z2, SUB(c3, c1), c2), SUB(f43, f41));
    _mm512_store_ps(T(4, 3), t43);

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
    _mm512_store_ps(T(0, 4), t04);
    t14 = FMSUB(z3, c2, FMADD(z2, c1, c3));
    _mm512_store_ps(T(1, 4), t14);
    t24 = FMADD(z2, c1, SUB(c2, c3));
    _mm512_store_ps(T(2, 4), t24);
    t34 = SUB(c1, c3);
    _mm512_store_ps(T(3, 4), t34);
    t44 = ADD(FMADD(z2, ADD(SUB(f41, f43), SUB(c3, c1)), SUB(f44, f42)), c2);
    _mm512_store_ps(T(4, 4), t44);

    break;
  }
}

TRANS_INPUT(float, 5, 3, 16, ISA_SKX_AVX512);
TRANS_INPUTA(float, 5, 3, 16, ISA_SKX_AVX512);

} // namespace euler
