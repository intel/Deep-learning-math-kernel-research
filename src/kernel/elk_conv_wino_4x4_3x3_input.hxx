#pragma once
#include <assert.h>
#include "el_intrin.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_def.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
template <typename UserTypes, typename TrOpType, int v>
class convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, v, 6, 3> {
  protected:
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  constexpr static int V = v;
  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int A = 6;
  constexpr static int K = 3;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V],
      InputType *input, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V],
      InputType *input, int wA, int hA_start, int hA_end, int wA_start, int _wA_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<UserTypes> &xc, OutputType *output,
      TrOpType atoutput[A][A][V], BiasType *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<UserTypes> &xc,
      TrOpType *toutputa, TrOpType *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<UserTypes> &xc, OutputType *output,
      TrOpType aoutputa[A][A - K + 1][V], BiasType *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(
      TrOpType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]);
};

#undef ADD
#undef SUB
#undef MUL
#undef FMADD
#undef FMSUB
#undef FNMADD
#undef FNMSUB
#undef MAX
#undef XOR

#define ADD     _mm<V>::add_ps
#define SUB     _mm<V>::sub_ps
#define MUL     _mm<V>::mul_ps
#define FMADD   _mm<V>::fmadd_ps
#define FMSUB   _mm<V>::fmsub_ps
#define FNMADD  _mm<V>::fnmadd_ps
#define FNMSUB  _mm<V>::fnmsub_ps
#define MAX     _mm<V>::max_ps
#define XOR     _mm<V>::xor_ps

template <typename UserTypes, typename TrOpType, int V>
template <bool is_border>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_input(
      elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V], InputType *input,
      int hT_start, int hT_end, int wT_start, int wT_end)
{

#if 0
  // Inputs
  __m<V> f00, f01, f02, f03, f04, f05,
         f10, f11, f12, f13, f14, f15,
         f20, f21, f22, f23, f24, f25,
         f30, f31, f32, f33, f34, f35,
         f40, f41, f42, f43, f44, f45,
         f50, f51, f52, f53, f54, f55;
  // Cache
  __m<V> c1, c2, c3, c4;
  // Buffer
  __m<V> a00, a01, a02, a03;
  __m<V> b00, b01, b02, b03, b04, b05, b06, b07;
  __m<V> d00, d01, d02, d03;
  // Outputs
  __m<V> t00, t01, t02, t03, t04, t05,
         t10, t11, t12, t13, t14, t15,
         t20, t21, t22, t23, t24, t25,
         t30, t31, t32, t33, t34, t35,
         t40, t41, t42, t43, t44, t45,
         t50, t51, t52, t53, t54, t55;
#endif

  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(InputType, ainput, input, A, A, V);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
        return _mm<V>::cvtph_ps(f16);
      }
    } else {
      MD3(InputType, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end || _w > wT_end)) {
        return _mm<V>::setzero_ps();
      } else if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
        return _mm<V>::cvtph_ps(f16);
      }
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
#define ISTORE(i, j) _mm<V>::store_ps(T(i, j), t##i##j);

  auto z0 = _mm<V>::set1_ps(-2.25f);
  auto z1 = _mm<V>::set1_ps(-0.390625f);
  auto z2 = _mm<V>::set1_ps(0.87890625f);
  auto z3 = _mm<V>::set1_ps(-2.640625f);
  auto z4 = _mm<V>::set1_ps(0.625f);
  auto z5 = _mm<V>::set1_ps(1.5f);

    auto f00 = F(0, 0);
    auto f01 = F(0, 1);
    auto f02 = F(0, 2);
    auto f03 = F(0, 3);
    auto f04 = F(0, 4);
    auto f05 = F(0, 5);

    auto f10 = F(1, 0);
    auto f11 = F(1, 1);
    auto f12 = F(1, 2);
    auto f13 = F(1, 3);
    auto f14 = F(1, 4);
    auto f15 = F(1, 5);

    auto f20 = F(2, 0);
    auto f21 = F(2, 1);
    auto f22 = F(2, 2);
    auto f23 = F(2, 3);
    auto f24 = F(2, 4);
    auto f25 = F(2, 5);

    auto f30 = F(3, 0);
    auto f31 = F(3, 1);
    auto f32 = F(3, 2);
    auto f33 = F(3, 3);
    auto f34 = F(3, 4);
    auto f35 = F(3, 5);

    auto f40 = F(4, 0);
    auto f41 = F(4, 1);
    auto f42 = F(4, 2);
    auto f43 = F(4, 3);
    auto f44 = F(4, 4);
    auto f45 = F(4, 5);

    auto f50 = F(5, 0);

    auto t0 = f20 * z0 + f40;
    auto t1 = f10 * z0 + f30;
    auto t2 = f20 * z1 + f40;
    auto t3 = f10 * z1 + f30;
    auto t4 = f00 * z2 + f40;
    auto t5 = f10 * z2 + f50;

    auto m00 = f20 * z3 + t4;
    auto m10 = t1 * z4 + t0;
    auto m20 = t0 - t1 * z4;
    auto m30 = t3 * z5 + t2;
    auto m40 = t2 - t3 * z5;
    auto m50 = f30 * z3 + t5;

    auto f51 = F(5, 1);

    t0 = f21 * z0 + f41;
    t1 = f11 * z0 + f31;
    t2 = f21 * z1 + f41;
    t3 = f11 * z1 + f31;
    t4 = f01 * z2 + f41;
    t5 = f11 * z2 + f51;

    auto m01 = f21 * z3 + t4;
    auto m11 = t1 * z4 + t0;
    auto m21 = t0 - t1 * z4;
    auto m31 = t3 * z5 + t2;
    auto m41 = t2 - t3 * z5;
    auto m51 = f31 * z3 + t5;

    auto f52 = F(5, 2);

    t0 = f22 * z0 + f42;
    t1 = f12 * z0 + f32;
    t2 = f22 * z1 + f42;
    t3 = f12 * z1 + f32;
    t4 = f02 * z2 + f42;
    t5 = f12 * z2 + f52;

    auto m02 = f22 * z3 + t4;
    auto m12 = t1 * z4 + t0;
    auto m22 = t0 - t1 * z4;
    auto m32 = t3 * z5 + t2;
    auto m42 = t2 - t3 * z5;
    auto m52 = f32 * z3 + t5;

    auto f53 = F(5, 3);

    t0 = f23 * z0 + f43;
    t1 = f13 * z0 + f33;
    t2 = f23 * z1 + f43;
    t3 = f13 * z1 + f33;
    t4 = f03 * z2 + f43;
    t5 = f13 * z2 + f53;

    auto m03 = f23 * z3 + t4;
    auto m13 = t1 * z4 + t0;
    auto m23 = t0 - t1 * z4;
    auto m33 = t3 * z5 + t2;
    auto m43 = t2 - t3 * z5;
    auto m53 = f33 * z3 + t5;

    auto f54 = F(5, 4);

    t0 = f24 * z0 + f44;
    t1 = f14 * z0 + f34;
    t2 = f24 * z1 + f44;
    t3 = f14 * z1 + f34;
    t4 = f04 * z2 + f44;
    t5 = f14 * z2 + f54;

    auto m04 = f24 * z3 + t4;
    auto m14 = t1 * z4 + t0;
    auto m24 = t0 - t1 * z4;
    auto m34 = t3 * z5 + t2;
    auto m44 = t2 - t3 * z5;
    auto m54 = f34 * z3 + t5;

    auto f55 = F(5, 5);

    t0 = f25 * z0 + f45;
    t1 = f15 * z0 + f35;
    t2 = f25 * z1 + f45;
    t3 = f15 * z1 + f35;
    t4 = f05 * z2 + f45;
    t5 = f15 * z2 + f55;

    auto m05 = f25 * z3 + t4;
    auto m15 = t1 * z4 + t0;
    auto m25 = t0 - t1 * z4;
    auto m35 = t3 * z5 + t2;
    auto m45 = t2 - t3 * z5;
    auto m55 = f35 * z3 + t5;

    auto f0 = m00;
    auto f1 = m01;
    auto f2 = m02;
    auto f3 = m03;
    auto f4 = m04;
    auto f5 = m05;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(0, 0) = f2 * z3 + t4;
    *(__m<V>*)T(0, 1) = t1 * z4 + t0;
    *(__m<V>*)T(0, 2) = t0 - t1 * z4;
    *(__m<V>*)T(0, 3) = t3 * z5 + t2;
    *(__m<V>*)T(0, 4) = t2 - t3 * z5;
    *(__m<V>*)T(0, 5) = f3 * z3 + t5;

    f0 = m10;
    f1 = m11;
    f2 = m12;
    f3 = m13;
    f4 = m14;
    f5 = m15;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(1, 0) = f2 * z3 + t4;
    *(__m<V>*)T(1, 1) = t1 * z4 + t0;
    *(__m<V>*)T(1, 2) = t0 - t1 * z4;
    *(__m<V>*)T(1, 3) = t3 * z5 + t2;
    *(__m<V>*)T(1, 4) = t2 - t3 * z5;
    *(__m<V>*)T(1, 5) = f3 * z3 + t5;

    f0 = m20;
    f1 = m21;
    f2 = m22;
    f3 = m23;
    f4 = m24;
    f5 = m25;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(2, 0) = f2 * z3 + t4;
    *(__m<V>*)T(2, 1) = t1 * z4 + t0;
    *(__m<V>*)T(2, 2) = t0 - t1 * z4;
    *(__m<V>*)T(2, 3) = t3 * z5 + t2;
    *(__m<V>*)T(2, 4) = t2 - t3 * z5;
    *(__m<V>*)T(2, 5) = f3 * z3 + t5;

    f0 = m30;
    f1 = m31;
    f2 = m32;
    f3 = m33;
    f4 = m34;
    f5 = m35;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(3, 0) = f2 * z3 + t4;
    *(__m<V>*)T(3, 1) = t1 * z4 + t0;
    *(__m<V>*)T(3, 2) = t0 - t1 * z4;
    *(__m<V>*)T(3, 3) = t3 * z5 + t2;
    *(__m<V>*)T(3, 4) = t2 - t3 * z5;
    *(__m<V>*)T(3, 5) = f3 * z3 + t5;

    f0 = m40;
    f1 = m41;
    f2 = m42;
    f3 = m43;
    f4 = m44;
    f5 = m45;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(4, 0) = f2 * z3 + t4;
    *(__m<V>*)T(4, 1) = t1 * z4 + t0;
    *(__m<V>*)T(4, 2) = t0 - t1 * z4;
    *(__m<V>*)T(4, 3) = t3 * z5 + t2;
    *(__m<V>*)T(4, 4) = t2 - t3 * z5;
    *(__m<V>*)T(4, 5) = f3 * z3 + t5;

    f0 = m50;
    f1 = m51;
    f2 = m52;
    f3 = m53;
    f4 = m54;
    f5 = m55;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V>*)T(5, 0) = f2 * z3 + t4;
    *(__m<V>*)T(5, 1) = t1 * z4 + t0;
    *(__m<V>*)T(5, 2) = t0 - t1 * z4;
    *(__m<V>*)T(5, 3) = t3 * z5 + t2;
    *(__m<V>*)T(5, 4) = t2 - t3 * z5;
    *(__m<V>*)T(5, 5) = f3 * z3 + t5;

#if 0
  VECTOR_DEF(M6, ME3);

  auto z4 = _mm<V>::set1_ps(4.0f);
  auto z5 = _mm<V>::set1_ps(5.0f);

  c1 = FMSUB(z4, f10, (FMSUB(z5, f12, f14)));
  c2 = FMSUB(z4, f20, (FMSUB(z5, f22, f24)));
  c3 = FMSUB(z4, f30, (FMSUB(z5, f32, f34)));
  c4 = FMSUB(z4, f40, (FMSUB(z5, f42, f44)));

  auto z2 = _mm<V>::set1_ps(2.0f);
  auto z16 = _mm<V>::set1_ps(16.0f);
  auto z20 = _mm<V>::set1_ps(20.0f);

  t00 = FMADD(z16, f00, FNMADD(z20, f02, FMADD(z4, f04, FNMADD(z5, c2, c4))));
  ISTORE(0, 0)
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t10 = SUB(a01, a00);
  ISTORE(1, 0)
  t20 = ADD(a01, a00);
  ISTORE(2, 0)
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t30 = FNMSUB(z2, a02, a03);
  ISTORE(3, 0)
  t40 = FMSUB(z2, a02, a03);
  ISTORE(4, 0);
  t50 = FMADD(z4, ADD(c1, f50), FNMADD(z5, ADD(c3, f52), f54));
  ISTORE(5, 0)

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
  ISTORE(0, 1)
  a00 = FMSUB(z4, c1, c3);
  a01 = FMSUB(z4, c2, c4);
  t11 = ADD(a01, a00);
  ISTORE(1, 1)
  t21 = SUB(a01, a00);
  ISTORE(2, 1)
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t31 = FMADD(z2, a02, a03);
  ISTORE(3, 1)
  t41 = FNMADD(z2, a02, a03);
  ISTORE(4, 1)
  d02 = FMSUB(z4, f51, f53);
  d03 = FNMADD(z4, f52, f54);
  t51 = FNMADD(z4, c1, FMADD(z5, c3, SUB(d03, d02)));
  ISTORE(5, 1)

  c1 = SUB(b00, b04);
  c2 = SUB(b01, b05);
  c3 = SUB(b02, b06);
  c4 = SUB(b03, b07);

  t02 = FMADD(z4, d00, FNMADD(z4, d01, FNMADD(z5, c2, c4)));
  ISTORE(0, 2)
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t12 = SUB(a01, a00);
  ISTORE(1, 2)
  t22 = ADD(a01, a00);
  ISTORE(2, 2)
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t32 = FNMADD(z2, a02, a03);
  ISTORE(3, 2)
  t42 = FMADD(z2, a02, a03);
  ISTORE(4, 2)
  t52 = FMADD(z4, c1, FNMADD(z5, c3, ADD(d03, d02)));
  ISTORE(5, 2)

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

  auto z8 = _mm<V>::set1_ps(8.0f);

  d00 = SUB(f04, f02);
  d01 = SUB(f03, f01);
  t03 = FMADD(z8, d01, FMADD(z4, d00, FMSUB(z5, c2, c4)));
  ISTORE(0, 3)
  a00 = FMSUB(z4, c1, c3);
  a01 = FMSUB(z4, c2, c4);
  t13 = ADD(a01, a00);
  ISTORE(1, 3)
  t23 = SUB(a01, a00);
  ISTORE(2, 3)
  a02 = SUB(c1, c3);
  a03 = SUB(c2, c4);
  t33 = FMADD(z2, a02, a03);
  ISTORE(3, 3)
  t43 = FNMADD(z2, a02, a03);
  ISTORE(4, 3)
  d02 = SUB(f53, f51);
  d03 = SUB(f54, f52);
  t53 = FNMADD(z4, c1, FMADD(z5, c3, FMADD(z2, d02, d03)));
  ISTORE(5, 3)

  c1 = FMSUB(z2, b00, b04);
  c2 = FMSUB(z2, b01, b05);
  c3 = FMSUB(z2, b02, b06);
  c4 = FMSUB(z2, b03, b07);

  t04 = FNMADD(z8, d01, FMADD(z4, d00, FNMADD(z5, c2, c4)));
  ISTORE(0, 4)
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t14 = SUB(a01, a00);
  ISTORE(1, 4)
  t24 = ADD(a01, a00);
  ISTORE(2, 4)
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t34 = FNMADD(z2, a02, a03);
  ISTORE(3, 4)
  t44 = FMADD(z2, a02, a03);
  ISTORE(4, 4)
  t54 = FMADD(z4, c1, FNMADD(z5, c3, FNMADD(z2, d02, d03)));
  ISTORE(5, 4)

  VECTOR_DEF(M6, (5));

  c1 = FMADD(z4, f11, FNMADD(z5, f13, f15));
  c2 = FMADD(z4, f21, FNMADD(z5, f23, f25));
  c3 = FMADD(z4, f31, FNMADD(z5, f33, f35));
  c4 = FMADD(z4, f41, FNMADD(z5, f43, f45));

  t05 = FMADD(z4, FMADD(z4, f01, f05), FNMADD(z5, FMADD(z4, f03, c2), c4));
  ISTORE(0, 5)
  a00 = FMSUB(z4, c1, c3);
  a01 = FNMADD(z4, c2, c4);
  t15 = SUB(a01, a00);
  ISTORE(1, 5)
  t25 = ADD(a01, a00);
  ISTORE(2, 5)
  a02 = SUB(c1, c3);
  a03 = SUB(c4, c2);
  t35 = FNMADD(z2, a02, a03);
  ISTORE(3, 5)
  t45 = FMADD(z2, a02, a03);
  ISTORE(4, 5)
  t55 = FMADD(z4, ADD(c1, f51), FNMADD(z5, ADD(c3, f53), f55));
  ISTORE(5, 5);
#endif
}

template <typename UserTypes, typename TrOpType, int V>
template <bool is_border>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_inputa(elx_conv_t<UserTypes> &xc,
    TrOpType atinput[A][A][V], InputType *input, int wA, int hT_start,
    int hT_end, int wT_start, int wT_end)
{
  // TODO
  el_error("Unimplemented");
}
}
