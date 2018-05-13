#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

namespace euler {

template <typename T, const int A, const int K, const int V, const int I>
void elk_trans_input(elx_conv_t<T> &xc, T atinput[A][A][V], T *input) {}

template <typename T, const int A, const int K, const int V, const int I>
void elk_trans_input(elx_conv_t<T> &xc, T atinput[A][A][V], T *input,
                     int _hA_start, int _hA_end, int _wA_start, int _wA_end) {}


template <>
void elk_trans_input<float, 5, 3, 16, ISA_GENERIC>(elx_conv_t<float> &xc,
                                                   float atinput[5][5][16],
                                                   float *input,
                                                   int _hT_start, int _hT_end,
                                                   int _wT_start, int _wT_end) {
  const float z2 = 2.0f;
  const float z3 = 3.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

  auto f = [&](int _h, int _w, int _V) {
    int _i = _h * xc.input_strides[2] + _w * 16 + _V;
    if (_h < _hT_start || _w < _wT_start || _h > _hT_end || _w > _wT_end)
      return 0.0f;
    else
      return *(input + _i);
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f(_h, _w, _V)
#define C(n) C##n[_V]
#define T(_h, _w) atinput[_h][_w][_V]

  float C1[16], C2[16], C3[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1, 1) + z2 * F(1, 2) - z2 * F(1, 0) - F(1, 3);
    C(2) = F(2, 1) + z2 * F(2, 2) - z2 * F(2, 0) - F(2, 3);
    C(3) = F(3, 1) + z2 * F(3, 2) - z2 * F(3, 0) - F(3, 3);
    T(0, 0) = z4 * F(0, 0) - z2 * F(0, 1) - z4 * F(0, 2) + z2 * F(0, 3) + C(1) +
              z2 * C(2) - C(3);
    T(1, 0) = z3 * C(2) - z2 * C(1) - C(3);
    T(2, 0) = z2 * C(1) + C(2) - C(3);
    T(3, 0) = C(1) - C(3);
    T(4, 0) = z2 * F(4, 0) - F(4, 1) - z2 * F(4, 2) + F(4, 3) - z2 * C(1) +
              C(2) + z2 * C(3);

    C(1) = z3 * F(1, 2) - z2 * F(1, 1) - F(1, 3);
    C(2) = z3 * F(2, 2) - z2 * F(2, 1) - F(2, 3);
    C(3) = z3 * F(3, 2) - z2 * F(3, 1) - F(3, 3);
    T(0, 1) =
        z4 * F(0, 1) - z6 * F(0, 2) + z2 * F(0, 3) + C(1) + z2 * C(2) - C(3);
    T(1, 1) = z3 * C(2) - z2 * C(1) - C(3);
    T(2, 1) = z2 * C(1) + C(2) - C(3);
    T(3, 1) = C(1) - C(3);
    T(4, 1) =
        z2 * F(4, 1) - z3 * F(4, 2) + F(4, 3) - z2 * C(1) + C(2) + z2 * C(3);

    C(1) = z2 * F(1, 1) + F(1, 2) - F(1, 3);
    C(2) = z2 * F(2, 1) + F(2, 2) - F(2, 3);
    C(3) = z2 * F(3, 1) + F(3, 2) - F(3, 3);
    T(0, 2) =
        z2 * F(0, 3) - z2 * F(0, 2) - z4 * F(0, 1) + C(1) + z2 * C(2) - C(3);
    T(1, 2) = z3 * C(2) - z2 * C(1) - C(3);
    T(2, 2) = z2 * C(1) + C(2) - C(3);
    T(3, 2) = C(1) - C(3);
    T(4, 2) = F(4, 3) - z2 * F(4, 1) - F(4, 2) - z2 * C(1) + C(2) + z2 * C(3);

    C(1) = F(1, 1) - F(1, 3);
    C(2) = F(2, 1) - F(2, 3);
    C(3) = F(3, 1) - F(3, 3);
    T(0, 3) = z2 * F(0, 3) - z2 * F(0, 1) + C(1) + z2 * C(2) - C(3);
    T(1, 3) = z3 * C(2) - z2 * C(1) - C(3);
    T(2, 3) = z2 * C(1) + C(2) - C(3);
    T(3, 3) = C(1) - C(3);
    T(4, 3) = F(4, 3) - F(4, 1) - z2 * C(1) + C(2) + z2 * C(3);

    C(1) = F(1, 2) + z2 * F(1, 3) - z2 * F(1, 1) - F(1, 4);
    C(2) = F(2, 2) + z2 * F(2, 3) - z2 * F(2, 1) - F(2, 4);
    C(3) = F(3, 2) + z2 * F(3, 3) - z2 * F(3, 1) - F(3, 4);
    T(0, 4) = z4 * F(0, 1) - z2 * F(0, 2) - z4 * F(0, 3) + z2 * F(0, 4) + C(1) +
              z2 * C(2) - C(3);
    T(1, 4) = z3 * C(2) - z2 * C(1) - C(3);
    T(2, 4) = z2 * C(1) + C(2) - C(3);
    T(3, 4) = C(1) - C(3);
    T(4, 4) = z2 * F(4, 1) - F(4, 2) - z2 * F(4, 3) + F(4, 4) - z2 * C(1) +
              C(2) + z2 * C(3);
  }
}

template <>
void elk_trans_input<float, 5, 3, 16, ISA_GENERIC>(elx_conv_t<float> &xc,
                                                   float atinput[5][5][16],
                                                   float *input) {
  elk_trans_input<float, 5, 3, 16, ISA_GENERIC>(xc, atinput, input, 0, 4, 0, 4);
}

template <>
void elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(elx_conv_t<float> &xc,
                                                      float atinput[5][5][16],
                                                      float *input) {
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

  auto f = [&](int _h, int _w) {
    mdarray<float, 3> ainput(input, xc.ih, xc.iw, 16);
    return _mm512_load_ps(&ainput(_h, _w, 0));
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f(_h, _w)
#define T(h, w) atinput[h][w]

  f00 = F(0, 0);
  f01 = F(0, 1);
  f02 = F(0, 2);
  f03 = F(0, 3);
  f10 = F(1, 0);
  f11 = F(1, 1);
  f12 = F(1, 2);
  f13 = F(1, 3);
  f20 = F(2, 0);
  f21 = F(2, 1);
  f22 = F(2, 2);
  f23 = F(2, 3);
  f30 = F(3, 0);
  f31 = F(3, 1);
  f32 = F(3, 2);
  f33 = F(3, 3);
  f40 = F(4, 0);
  f41 = F(4, 1);
  f42 = F(4, 2);
  f43 = F(4, 3);

  c1 = _mm512_sub_ps(f11, f13);
  c1 = _mm512_fmadd_ps(z2, f12, c1);
  c1 = _mm512_fmsub_ps(z2, f10, c1);
  c2 = _mm512_sub_ps(f21, f23);
  c2 = _mm512_fmadd_ps(z2, f22, c2);
  c2 = _mm512_fmsub_ps(z2, f20, c2);
  c3 = _mm512_sub_ps(f31, f33);
  c3 = _mm512_fmadd_ps(z2, f32, c3);
  c3 = _mm512_fmsub_ps(z2, f30, c3);
  t00 = _mm512_sub_ps(c1, c3);
  t00 = _mm512_fmadd_ps(z4, f00, t00);
  t00 = _mm512_fmsub_ps(z2, f01, t00);
  t00 = _mm512_fmsub_ps(z4, f02, t00);
  t00 = _mm512_fmadd_ps(z2, f03, t00);
  t00 = _mm512_fmadd_ps(z2, c2, t00);
  _mm512_store_ps(T(0, 0), t00);
  t10 = _mm512_mul_ps(z3, c2);
  t10 = _mm512_fmsub_ps(z2, c1, t10);
  t10 = _mm512_sub_ps(t10, c3);
  _mm512_store_ps(T(1, 0), t10);
  t20 = _mm512_sub_ps(c2, c3);
  t20 = _mm512_fmadd_ps(z2, c1, t20);
  _mm512_store_ps(T(2, 0), t20);
  t30 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 0), t30);
  t40 = _mm512_sub_ps(f43, f41);
  t40 = _mm512_fmadd_ps(z2, f40, t40);
  t40 = _mm512_fmsub_ps(z2, f42, t40);
  t40 = _mm512_fmsub_ps(z2, c1, t40);
  t40 = _mm512_fmadd_ps(z2, c3, t40);
  t40 = _mm512_add_ps(t40, c2);
  _mm512_store_ps(T(4, 0), t40);

  c1 = _mm512_mul_ps(z3, f12);
  c1 = _mm512_fmsub_ps(z2, f11, c1);
  c1 = _mm512_sub_ps(c1, f13);
  c2 = _mm512_mul_ps(z3, f22);
  c2 = _mm512_fmsub_ps(z2, f21, c2);
  c2 = _mm512_sub_ps(c2, f23);
  c3 = _mm512_mul_ps(z3, f32);
  c3 = _mm512_fmsub_ps(z2, f31, c3);
  c3 = _mm512_sub_ps(c3, f33);
  t01 = _mm512_sub_ps(c1, c3);
  t01 = _mm512_fmadd_ps(z4, f01, t01);
  t01 = _mm512_fmsub_ps(z6, f02, t01);
  t01 = _mm512_fmadd_ps(z2, f03, t01);
  t01 = _mm512_fmadd_ps(z2, c2, t01);
  _mm512_store_ps(T(0, 1), t01);
  t11 = _mm512_mul_ps(z3, c2);
  t11 = _mm512_fmsub_ps(z2, c1, t11);
  t11 = _mm512_sub_ps(t11, c3);
  _mm512_store_ps(T(1, 1), t11);
  t21 = _mm512_sub_ps(c2, c3);
  t21 = _mm512_fmadd_ps(z2, c1, t21);
  _mm512_store_ps(T(2, 1), t21);
  t31 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 1), t31);
  t41 = _mm512_add_ps(f43, c2);
  t41 = _mm512_fmadd_ps(z2, f41, t41);
  t41 = _mm512_fmsub_ps(z3, f42, t41);
  t41 = _mm512_fmsub_ps(z2, c1, t41);
  t41 = _mm512_fmadd_ps(z2, c3, t41);
  _mm512_store_ps(T(4, 1), t41);

  c1 = _mm512_sub_ps(f12, f13);
  c1 = _mm512_fmadd_ps(z2, f11, c1);
  c2 = _mm512_sub_ps(f22, f23);
  c2 = _mm512_fmadd_ps(z2, f21, c2);
  c3 = _mm512_sub_ps(f32, f33);
  c3 = _mm512_fmadd_ps(z2, f31, c3);
  t02 = _mm512_sub_ps(c1, c3);
  t02 = _mm512_fmadd_ps(z2, f03, t02);
  t02 = _mm512_fmsub_ps(z2, f02, t02);
  t02 = _mm512_fmsub_ps(z4, f01, t02);
  t02 = _mm512_fmadd_ps(z2, c2, t02);
  _mm512_store_ps(T(0, 2), t02);
  t12 = _mm512_mul_ps(z3, c2);
  t12 = _mm512_fmsub_ps(z2, c1, t12);
  t12 = _mm512_sub_ps(t12, c3);
  _mm512_store_ps(T(1, 2), t12);
  t22 = _mm512_sub_ps(c2, c3);
  t22 = _mm512_fmadd_ps(z2, c1, t22);
  _mm512_store_ps(T(2, 2), t22);
  t32 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 2), t32);
  t42 = _mm512_sub_ps(f43, f42);
  t42 = _mm512_fmsub_ps(z2, f41, t42);
  t42 = _mm512_fmsub_ps(z2, c1, t42);
  t42 = _mm512_fmadd_ps(z2, c3, t42);
  t42 = _mm512_add_ps(t42, c2);
  _mm512_store_ps(T(4, 2), t42);

  c1 = _mm512_sub_ps(f11, f13);
  c2 = _mm512_sub_ps(f21, f23);
  c3 = _mm512_sub_ps(f31, f33);
  t03 = _mm512_sub_ps(c1, c3);
  t03 = _mm512_fmadd_ps(z2, f03, t03);
  t03 = _mm512_fmsub_ps(z2, f01, t03);
  t03 = _mm512_fmadd_ps(z2, c2, t03);
  _mm512_store_ps(T(0, 3), t03);
  t13 = _mm512_mul_ps(z3, c2);
  t13 = _mm512_fmsub_ps(z2, c1, t13);
  t13 = _mm512_sub_ps(t13, c3);
  _mm512_store_ps(T(1, 3), t13);
  t23 = _mm512_sub_ps(c2, c3);
  t23 = _mm512_fmadd_ps(z2, c1, t23);
  _mm512_store_ps(T(2, 3), t23);
  t33 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 3), t33);
  t43 = _mm512_sub_ps(f43, f41);
  t33 = _mm512_mul_ps(z2, t33);
  t43 = _mm512_add_ps(t43, t33);
  t43 = _mm512_add_ps(t43, c2);
  _mm512_store_ps(T(4, 3), t43);

  f04 = F(0, 4);
  f14 = F(1, 4);
  f24 = F(2, 4);
  f34 = F(3, 4);
  f44 = F(4, 4);

  c1 = _mm512_sub_ps(f12, f14);
  c1 = _mm512_fmadd_ps(z2, f13, c1);
  c1 = _mm512_fmsub_ps(z2, f11, c1);
  c2 = _mm512_sub_ps(f22, f24);
  c2 = _mm512_fmadd_ps(z2, f23, c2);
  c2 = _mm512_fmsub_ps(z2, f21, c2);
  c3 = _mm512_sub_ps(f32, f34);
  c3 = _mm512_fmadd_ps(z2, f33, c3);
  c3 = _mm512_fmsub_ps(z2, f31, c3);
  t34 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 4), t34);
  t04 = _mm512_sub_ps(f01, f03);
  t04 = _mm512_mul_ps(z4, t04);
  t04 = _mm512_fmsub_ps(z2, f02, t04);
  t04 = _mm512_fmadd_ps(z2, f04, t04);
  t04 = _mm512_fmadd_ps(z2, c2, t04);
  t04 = _mm512_add_ps(t04, t34);
  _mm512_store_ps(T(0, 4), t04);
  t44 = _mm512_sub_ps(f41, f43);
  t44 = _mm512_mul_ps(z2, t44);
  t44 = _mm512_sub_ps(t44, f42);
  t44 = _mm512_add_ps(t44, f44);
  t34 = _mm512_mul_ps(z2, t34);
  t44 = _mm512_sub_ps(t44, t34);
  t44 = _mm512_add_ps(t44, c2);
  _mm512_store_ps(T(4, 4), t44);
  t14 = _mm512_mul_ps(z3, c2);
  t14 = _mm512_fmsub_ps(z2, c1, t14);
  t14 = _mm512_sub_ps(t14, c3);
  _mm512_store_ps(T(1, 4), t14);
  t24 = _mm512_sub_ps(c2, c3);
  t24 = _mm512_fmadd_ps(z2, c1, t24);
  _mm512_store_ps(T(2, 4), t24);
}

template <>
void elk_trans_input<float, 5, 3, 16, ISA_SKX_AVX512>(
    elx_conv_t<float> &xc, float atinput[5][5][16], float *input, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end) {

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

  auto f = [&](int _h, int _w) {
    int _i = _h * xc.input_strides[2] + _w * 16;
    if (_h < _hT_start || _w < _wT_start || _h > _hT_end || _w > _wT_end)
      return z0;
    else
      return _mm512_load_ps(input + _i);
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f(_h, _w)
#define T(h, w) atinput[h][w]

  f00 = F(0, 0);
  f01 = F(0, 1);
  f02 = F(0, 2);
  f03 = F(0, 3);
  f10 = F(1, 0);
  f11 = F(1, 1);
  f12 = F(1, 2);
  f13 = F(1, 3);
  f20 = F(2, 0);
  f21 = F(2, 1);
  f22 = F(2, 2);
  f23 = F(2, 3);
  f30 = F(3, 0);
  f31 = F(3, 1);
  f32 = F(3, 2);
  f33 = F(3, 3);
  f40 = F(4, 0);
  f41 = F(4, 1);
  f42 = F(4, 2);
  f43 = F(4, 3);

  c1 = _mm512_sub_ps(f11, f13);
  c1 = _mm512_fmadd_ps(z2, f12, c1);
  c1 = _mm512_fmsub_ps(z2, f10, c1);
  c2 = _mm512_sub_ps(f21, f23);
  c2 = _mm512_fmadd_ps(z2, f22, c2);
  c2 = _mm512_fmsub_ps(z2, f20, c2);
  c3 = _mm512_sub_ps(f31, f33);
  c3 = _mm512_fmadd_ps(z2, f32, c3);
  c3 = _mm512_fmsub_ps(z2, f30, c3);
  t00 = _mm512_sub_ps(c1, c3);
  t00 = _mm512_fmadd_ps(z4, f00, t00);
  t00 = _mm512_fmsub_ps(z2, f01, t00);
  t00 = _mm512_fmsub_ps(z4, f02, t00);
  t00 = _mm512_fmadd_ps(z2, f03, t00);
  t00 = _mm512_fmadd_ps(z2, c2, t00);
  _mm512_store_ps(T(0, 0), t00);
  t10 = _mm512_mul_ps(z3, c2);
  t10 = _mm512_fmsub_ps(z2, c1, t10);
  t10 = _mm512_sub_ps(t10, c3);
  _mm512_store_ps(T(1, 0), t10);
  t20 = _mm512_sub_ps(c2, c3);
  t20 = _mm512_fmadd_ps(z2, c1, t20);
  _mm512_store_ps(T(2, 0), t20);
  t30 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 0), t30);
  t40 = _mm512_sub_ps(f43, f41);
  t40 = _mm512_fmadd_ps(z2, f40, t40);
  t40 = _mm512_fmsub_ps(z2, f42, t40);
  t40 = _mm512_fmsub_ps(z2, c1, t40);
  t40 = _mm512_fmadd_ps(z2, c3, t40);
  t40 = _mm512_add_ps(t40, c2);
  _mm512_store_ps(T(4, 0), t40);

  c1 = _mm512_mul_ps(z3, f12);
  c1 = _mm512_fmsub_ps(z2, f11, c1);
  c1 = _mm512_sub_ps(c1, f13);
  c2 = _mm512_mul_ps(z3, f22);
  c2 = _mm512_fmsub_ps(z2, f21, c2);
  c2 = _mm512_sub_ps(c2, f23);
  c3 = _mm512_mul_ps(z3, f32);
  c3 = _mm512_fmsub_ps(z2, f31, c3);
  c3 = _mm512_sub_ps(c3, f33);
  t01 = _mm512_sub_ps(c1, c3);
  t01 = _mm512_fmadd_ps(z4, f01, t01);
  t01 = _mm512_fmsub_ps(z6, f02, t01);
  t01 = _mm512_fmadd_ps(z2, f03, t01);
  t01 = _mm512_fmadd_ps(z2, c2, t01);
  _mm512_store_ps(T(0, 1), t01);
  t11 = _mm512_mul_ps(z3, c2);
  t11 = _mm512_fmsub_ps(z2, c1, t11);
  t11 = _mm512_sub_ps(t11, c3);
  _mm512_store_ps(T(1, 1), t11);
  t21 = _mm512_sub_ps(c2, c3);
  t21 = _mm512_fmadd_ps(z2, c1, t21);
  _mm512_store_ps(T(2, 1), t21);
  t31 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 1), t31);
  t41 = _mm512_add_ps(f43, c2);
  t41 = _mm512_fmadd_ps(z2, f41, t41);
  t41 = _mm512_fmsub_ps(z3, f42, t41);
  t41 = _mm512_fmsub_ps(z2, c1, t41);
  t41 = _mm512_fmadd_ps(z2, c3, t41);
  _mm512_store_ps(T(4, 1), t41);

  c1 = _mm512_sub_ps(f12, f13);
  c1 = _mm512_fmadd_ps(z2, f11, c1);
  c2 = _mm512_sub_ps(f22, f23);
  c2 = _mm512_fmadd_ps(z2, f21, c2);
  c3 = _mm512_sub_ps(f32, f33);
  c3 = _mm512_fmadd_ps(z2, f31, c3);
  t02 = _mm512_sub_ps(c1, c3);
  t02 = _mm512_fmadd_ps(z2, f03, t02);
  t02 = _mm512_fmsub_ps(z2, f02, t02);
  t02 = _mm512_fmsub_ps(z4, f01, t02);
  t02 = _mm512_fmadd_ps(z2, c2, t02);
  _mm512_store_ps(T(0, 2), t02);
  t12 = _mm512_mul_ps(z3, c2);
  t12 = _mm512_fmsub_ps(z2, c1, t12);
  t12 = _mm512_sub_ps(t12, c3);
  _mm512_store_ps(T(1, 2), t12);
  t22 = _mm512_sub_ps(c2, c3);
  t22 = _mm512_fmadd_ps(z2, c1, t22);
  _mm512_store_ps(T(2, 2), t22);
  t32 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 2), t32);
  t42 = _mm512_sub_ps(f43, f42);
  t42 = _mm512_fmsub_ps(z2, f41, t42);
  t42 = _mm512_fmsub_ps(z2, c1, t42);
  t42 = _mm512_fmadd_ps(z2, c3, t42);
  t42 = _mm512_add_ps(t42, c2);
  _mm512_store_ps(T(4, 2), t42);

  c1 = _mm512_sub_ps(f11, f13);
  c2 = _mm512_sub_ps(f21, f23);
  c3 = _mm512_sub_ps(f31, f33);
  t03 = _mm512_sub_ps(c1, c3);
  t03 = _mm512_fmadd_ps(z2, f03, t03);
  t03 = _mm512_fmsub_ps(z2, f01, t03);
  t03 = _mm512_fmadd_ps(z2, c2, t03);
  _mm512_store_ps(T(0, 3), t03);
  t13 = _mm512_mul_ps(z3, c2);
  t13 = _mm512_fmsub_ps(z2, c1, t13);
  t13 = _mm512_sub_ps(t13, c3);
  _mm512_store_ps(T(1, 3), t13);
  t23 = _mm512_sub_ps(c2, c3);
  t23 = _mm512_fmadd_ps(z2, c1, t23);
  _mm512_store_ps(T(2, 3), t23);
  t33 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 3), t33);
  t43 = _mm512_sub_ps(f43, f41);
  t33 = _mm512_mul_ps(z2, t33);
  t43 = _mm512_add_ps(t43, t33);
  t43 = _mm512_add_ps(t43, c2);
  _mm512_store_ps(T(4, 3), t43);

  f04 = F(0, 4);
  f14 = F(1, 4);
  f24 = F(2, 4);
  f34 = F(3, 4);
  f44 = F(4, 4);
  c1 = _mm512_sub_ps(f12, f14);
  c1 = _mm512_fmadd_ps(z2, f13, c1);
  c1 = _mm512_fmsub_ps(z2, f11, c1);
  c2 = _mm512_sub_ps(f22, f24);
  c2 = _mm512_fmadd_ps(z2, f23, c2);
  c2 = _mm512_fmsub_ps(z2, f21, c2);
  c3 = _mm512_sub_ps(f32, f34);
  c3 = _mm512_fmadd_ps(z2, f33, c3);
  c3 = _mm512_fmsub_ps(z2, f31, c3);
  t34 = _mm512_sub_ps(c1, c3);
  _mm512_store_ps(T(3, 4), t34);
  t04 = _mm512_sub_ps(f01, f03);
  t04 = _mm512_mul_ps(z4, t04);
  t04 = _mm512_fmsub_ps(z2, f02, t04);
  t04 = _mm512_fmadd_ps(z2, f04, t04);
  t04 = _mm512_fmadd_ps(z2, c2, t04);
  t04 = _mm512_add_ps(t04, t34);
  _mm512_store_ps(T(0, 4), t04);
  t44 = _mm512_sub_ps(f41, f43);
  t44 = _mm512_mul_ps(z2, t44);
  t44 = _mm512_sub_ps(t44, f42);
  t44 = _mm512_add_ps(t44, f44);
  t34 = _mm512_mul_ps(z2, t34);
  t44 = _mm512_sub_ps(t44, t34);
  t44 = _mm512_add_ps(t44, c2);
  _mm512_store_ps(T(4, 4), t44);
  t14 = _mm512_mul_ps(z3, c2);
  t14 = _mm512_fmsub_ps(z2, c1, t14);
  t14 = _mm512_sub_ps(t14, c3);
  _mm512_store_ps(T(1, 4), t14);
  t24 = _mm512_sub_ps(c2, c3);
  t24 = _mm512_fmadd_ps(z2, c1, t24);
  _mm512_store_ps(T(2, 4), t24);
}

}  // namespace euler
