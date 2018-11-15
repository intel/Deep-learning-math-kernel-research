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

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_output(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A][V], BiasType *bias,
    int hOA_end, int wOA_end)
{
#if 0
  __m<V> z2 = _mm<V>::set1_ps(2.0f);
  __m<V> z4 = _mm<V>::set1_ps(4.0f);
  __m<V> z8 = _mm<V>::set1_ps(8.0f);

  // Inputs
  __m<V> t00, t01, t02, t03, t04, t05,
         t10, t11, t12, t13, t14, t15,
         t20, t21, t22, t23, t24, t25,
         t30, t31, t32, t33, t34, t35,
         t40, t41, t42, t43, t44, t45,
         t50, t51, t52, t53, t54, t55;
  // Cache
  __m<V> c0, c1, c2, c3, zero;
  // Buffer
  __m<V> a00, a01, a02, a03;
  __m<V> b00, b01, b02, b03;
  __m<V> d00, d01, d02, d03,
         d04, d05, d06, d07;
  // Outputs
  __m<V> p00, p01, p02, p03,
         p10, p11, p12, p13,
         p20, p21, p22, p23,
         p30, p31, p32, p33;
#endif

  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  // TODO replace bias != nullptr with last_ic4 condition
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) OutputType dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
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
#undef FUSE_BIAS

#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)
#define t(m, n) t##m##n
#define OP(m,n) t(m,n) = _mm<V>::load_ps(T(m, n))

#define ISTORE(i, j)                                              \
  if (std::is_same<OutputType, float>::value)                     \
    _mm<V>::store_ps(P(i, j), p##i##j);                           \
  else {                                                          \
    auto f16 = _mm<V>::cvtps_ph(p##i##j,                          \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
    _mm<V/2>::store_si256((__m256i *)P(i, j), f16);               \
  }

#define FUSE_BIAS(p)                                              \
  if (std::is_same<BiasType, float>::value) {                     \
    p = ADD(p, *(__m<V>*)bias);                                   \
  } else {                                                        \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);            \
    p = ADD(p, _mm<V>::cvtph_ps(f16v));                           \
  }

  float M[4][6][16];

  __m<V> z0 = _mm<V>::set1_ps(1.5f);
  __m<V> z1 = _mm<V>::set1_ps(2.25f);
  __m<V> z2 = _mm<V>::set1_ps(3.375f);
  __m<V> z3 = _mm<V>::set1_ps(0.625f);
  __m<V> z4 = _mm<V>::set1_ps(0.390625f);
  __m<V> z5 = _mm<V>::set1_ps(0.244140625f);
  __m<V> z = XOR(z, z);

#pragma unroll
  for (int i = 0; i < 6; i++) {
    auto f0 = _mm<V>::load_ps(T(0, i));
    auto f1 = _mm<V>::load_ps(T(1, i));
    auto f2 = _mm<V>::load_ps(T(2, i));
    auto f3 = _mm<V>::load_ps(T(3, i));
    auto f4 = _mm<V>::load_ps(T(4, i));
    auto f5 = _mm<V>::load_ps(T(5, i));
    auto t0 = f1 + f2;
    auto t1 = f3 + f4;
    auto t2 = f1 - f2;
    auto t3 = f3 - f4;

    *(__m<V>*)M[0][i] = t0 + t1 + f0;
    *(__m<V>*)M[1][i] = t2 * z3 + t3 * z0;
    *(__m<V>*)M[2][i] = t0 * z4 + t1 * z1;
    *(__m<V>*)M[3][i] = t2 * z5 + t3 * z2 + f5;
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
      auto f0 = _mm<V>::load_ps(M[i][0]);
      auto f1 = _mm<V>::load_ps(M[i][1]);
      auto f2 = _mm<V>::load_ps(M[i][2]);
      auto f3 = _mm<V>::load_ps(M[i][3]);
      auto f4 = _mm<V>::load_ps(M[i][4]);
      auto f5 = _mm<V>::load_ps(M[i][5]);
      auto t0 = f1 + f2;
      auto t1 = f3 + f4;
      auto t2 = f1 - f2;
      auto t3 = f3 - f4;

      auto p0 = t0 + t1 + f0;
      auto p1 = t2 * z3 + t3 * z0;
      auto p2 = t0 * z4 + t1 * z1;
      auto p3 = t2 * z5 + t3 * z2 + f5;

      if (fuse_bias) {
        p0 += *(__m<V> *)bias;
        p1 += *(__m<V> *)bias;
        p2 += *(__m<V> *)bias;
        p3 += *(__m<V> *)bias;
      }
      if (fuse_ip_sum) {
        p0 += *(__m<V> *)P(i, 0);
        p1 += *(__m<V> *)P(i, 1);
        p2 += *(__m<V> *)P(i, 2);
        p3 += *(__m<V> *)P(i, 3);
      }
      if (fuse_relu) {
        p0 = MAX(p0, z);
        p1 = MAX(p1, z);
        p2 = MAX(p2, z);
        p3 = MAX(p3, z);
      }
      *(__m<V>*)P(i, 0) = p0;
      *(__m<V>*)P(i, 1) = p1;
      *(__m<V>*)P(i, 2) = p2;
      *(__m<V>*)P(i, 3) = p3;
  }

#if 0
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
  if (fuse_bias) {FUSE_BIAS(p00)}
  if (fuse_ip_sum) p00 = ADD(p00, *(__m<V>*)P(0, 0));
  if (fuse_relu) {
    zero = XOR(zero, zero);
    p00 = MAX(p00, zero);
  }
  ISTORE(0, 0);
  p10 = FMADD(z2, a03, a02);
  if (fuse_bias) {FUSE_BIAS(p10)}
  if (fuse_ip_sum) p10 = ADD(p10, *(__m<V>*)P(1, 0));
  if (fuse_relu) p10 = MAX(p10, zero);
  ISTORE(1, 0);
  p20 = FMADD(z4, a01, a00);
  if (fuse_bias) {FUSE_BIAS(p20)}
  if (fuse_ip_sum) p20 = ADD(p20, *(__m<V>*)P(2, 0));
  if (fuse_relu) p20 = MAX(p20, zero);
  ISTORE(2, 0);
  p30 = FMADD(z8, a03, ADD(a02, ADD(t50, ADD(b02, b03))));
  if (fuse_bias) {FUSE_BIAS(p30)}
  if (fuse_ip_sum) p30 = ADD(p30, *(__m<V>*)P(3, 0));
  if (fuse_relu) p30 = MAX(p30, zero);
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
  if (fuse_bias) {FUSE_BIAS(p02)}
  if (fuse_ip_sum) p02 = ADD(p02, *(__m<V>*)P(0, 2));
  if (fuse_relu) p02 = MAX(p02, zero);
  ISTORE(0, 2);
  p12 = FMADD(z2, a03, a02);
  if (fuse_bias) {FUSE_BIAS(p12)}
  if (fuse_ip_sum) p12 = ADD(p12, *(__m<V>*)P(1, 2));
  if (fuse_relu) p12 = MAX(p12, zero);
  ISTORE(1, 2);
  p22 = FMADD(z4, a01, a00);
  if (fuse_bias) {FUSE_BIAS(p22)}
  if (fuse_ip_sum) p22 = ADD(p22, *(__m<V>*)P(2, 2));
  if (fuse_relu) p22 = MAX(p22, zero);
  ISTORE(2, 2);
  p32 = ADD(FMADD(z8, a03, a02), FMADD(z4, b03, b02));
  if (fuse_bias) {FUSE_BIAS(p32)}
  if (fuse_ip_sum) p32 = ADD(p32, *(__m<V>*)P(3, 2));
  if (fuse_relu) p32 = MAX(p32, zero);
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
  if (fuse_bias) {FUSE_BIAS(p01)}
  if (fuse_ip_sum) p01 = ADD(p01, *(__m<V>*)P(0, 1));
  if (fuse_relu) p01 = MAX(p01, zero);
  ISTORE(0, 1);
  p11 = FMADD(z2, a03, a02);
  if (fuse_bias) {FUSE_BIAS(p11)}
  if (fuse_ip_sum) p11 = ADD(p11, *(__m<V>*)P(1, 1));
  if (fuse_relu) p11 = MAX(p11, zero);
  ISTORE(1, 1);
  p21 = FMADD(z4, a01, a00);
  if (fuse_bias) {FUSE_BIAS(p21)}
  if (fuse_ip_sum) p21 = ADD(p21, *(__m<V>*)P(2, 1));
  if (fuse_relu) p21 = MAX(p21, zero);
  ISTORE(2, 1);
  p31 = ADD(FMADD(z8, a03, a02), FMADD(z2, b03, b02));
  if (fuse_bias) {FUSE_BIAS(p31)}
  if (fuse_ip_sum) p31 = ADD(p31, *(__m<V>*)P(3, 1));
  if (fuse_relu) p31 = MAX(p31, zero);
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
  if (fuse_bias) {FUSE_BIAS(p03)}
  if (fuse_ip_sum) p03 = ADD(p03, *(__m<V>*)P(0, 3));
  if (fuse_relu) p03 = MAX(p03, zero);
  ISTORE(0, 3);
  p13 = FMADD(z2, a03, a02);
  if (fuse_bias) {FUSE_BIAS(p13)}
  if (fuse_ip_sum) p13 = ADD(p13, *(__m<V>*)P(1, 3));
  if (fuse_relu) p13 = MAX(p13, zero);
  ISTORE(1, 3);
  p23 = FMADD(z4, a01, a00);
  if (fuse_bias) {FUSE_BIAS(p23)}
  if (fuse_ip_sum) p23 = ADD(p23, *(__m<V>*)P(2, 3));
  if (fuse_relu) p23 = MAX(p23, zero);
  ISTORE(2, 3);
  p33 = ADD(FMADD(z8, a03, a02), ADD(FMADD(z8, b03, b02), t55));
  if (fuse_bias) {FUSE_BIAS(p33)}
  if (fuse_ip_sum) p33 = ADD(p33, *(__m<V>*)P(3, 3));
  if (fuse_relu) p33 = MAX(p33, zero);
  ISTORE(3, 3);
#endif
}

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_outputa_th(elx_conv_t<UserTypes> &xc,
    TrOpType *toutputa, TrOpType *toutput, int Tz, bool stream_out)
{
  el_error("Unimplemented");
}

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_outputa_bh(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A - K + 1][V], BiasType *bias,
    int hOA_end, int wOA_end)
{
  // TODO
  el_error("Unimplemented");
}
} // namespace euler
