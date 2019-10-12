#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
#undef FUSE_BIAS
#define FUSE_BIAS(p)                                                           \
  if (std::is_same<BiasType, float>::value) {                                  \
    p = p + *(__m<V>*)bias;                                                    \
  } else {                                                                     \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);                         \
    p = p + _mm<V>::cvtph_ps(f16v);                                            \
  }

#define AVX512_CALCULATE_O_0(n)                                                \
  c##n = t##n##0 + t##n##1 + t##n##2 + t##n##3 + t##n##4 + t##n##5;
#define AVX512_CALCULATE_O_1(n)                                                \
  c##n = ((z2 * (t##n##2 - t##n##3) + t##n##0)                                 \
          + (z1_2 * (t##n##4 - t##n##5) - t##n##1));
#define AVX512_CALCULATE_O_2(n)                                                \
  c##n = ((z4 * (t##n##2 + t##n##3) + t##n##0)                                 \
          + (z1_4 * (t##n##4 + t##n##5) + t##n##1));
#define AVX512_CALCULATE_O_3(n)                                                \
  c##n = ((z8 * (t##n##2 - t##n##3) + t##n##0)                                 \
          + (z1_8 * (t##n##4 - t##n##5) - t##n##1));
#define AVX512_CALCULATE_O_4(n)                                                \
  c##n = ((z16 * (t##n##2 + t##n##3) + (t##n##0 + t##n##1))                    \
          + (z1_16 * (t##n##4 + t##n##5) + t##n##6));

#undef STORE
#define STORE(i, j)                                                            \
  if (std::is_same<OutputType, float>::value)                                  \
    _mm<V>::store_ps(out_ptr(i, j), p##i##j);                                  \
  else {                                                                       \
    auto f16 = _mm<V>::cvtps_ph(p##i##j,                                       \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                        \
    _mm<V/2>::store_si256((__m256i *)out_ptr(i, j), f16);                      \
  }

#define AVX512_CALCULATE_O(n)                                                  \
  __m<V> p0##n = c0 + c1 + c2 + c3 + c4 + c5;                                  \
  if (fuse_bias) {FUSE_BIAS(p0##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p0##n = p0##n + *(__m<V>*)out_ptr(0, n);                                   \
  if (fuse_relu) {                                                             \
    zero = _mm<V>::xor_ps(zero, zero);                                         \
    p0##n = _mm<V>::max_ps(p0##n, zero);                                       \
  }                                                                            \
  STORE(0, n)                                                                  \
  __m<V> p1##n = (z2 * (c2 - c3) + c0) + (z1_2 * (c4 - c5) - c1);              \
  if (fuse_bias) {FUSE_BIAS(p1##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p1##n = (p1##n + *(__m<V>*)out_ptr(1, n));                                 \
  if (fuse_relu)                                                               \
    p1##n = _mm<V>::max_ps(p1##n, zero);                                       \
  STORE(1, n)                                                                  \
  __m<V> p2##n = (z4 * (c2 + c3) + c0) + (z1_4 * (c4 + c5) + c1);              \
  if (fuse_bias) {FUSE_BIAS(p2##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p2##n = p2##n + *(__m<V>*)out_ptr(2, n);                                   \
  if (fuse_relu)                                                               \
    p2##n = _mm<V>::max_ps(p2##n, zero);                                       \
  STORE(2, n)                                                                  \
  __m<V> p3##n = (z8 * (c2 - c3) + c0) + (z1_8 * (c4 - c5) - c1);              \
  if (fuse_bias) {FUSE_BIAS(p3##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p3##n = p3##n + *(__m<V>*)out_ptr(3, n);                                   \
  if (fuse_relu)                                                               \
    p3##n = _mm<V>::max_ps(p3##n, zero);                                       \
  STORE(3, n)                                                                  \
  __m<V> p4##n = (z16 * (c2 + c3) + c0) + (z1_16 * (c4 + c5) + c1);

#define AVX512_ADD_B(n);                                                       \
  if (fuse_bias) {FUSE_BIAS(p4##n)}                                            \
  if (fuse_ip_sum)                                                             \
    p4##n = p4##n + *(__m<V>*)out_ptr(4, n);                                   \
  if (fuse_relu)                                                               \
    p4##n = _mm<V>::max_ps(p4##n, zero);                                       \
  STORE(4, n)

template <typename OutputType, typename BiasType,
    int format, bool is_border, bool with_bias, bool with_relu,
    bool with_ip_sum, int V>
struct elk_conv_wino_trans_output<float, OutputType, BiasType, format,
    is_border, with_bias, with_relu, with_ip_sum, ISA_AVX512, 7, 3, V> {
  constexpr static int A = 7;
  constexpr static int K = 3;

  static void execute(elx_param_t &ep, OutputType *output,
      float *toutput, BiasType *bias, int hOA_end, int wOA_end)
  {
    ENABLE_AVX512F();
    bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
    bool fuse_bias = with_bias && (bias != nullptr);
    bool fuse_relu = with_relu && (bias != nullptr);

    MD3(float, atoutput, toutput, A, A, V);
    alignas(64) OutputType dummy[16];
    auto out_ptr = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
        return &md3(aoutput, _h, _w, 0);
      } else if (format == TKF_BLOCKED) {
        MD3(OutputType, aoutput, output, ep.oh, ep.ow, V);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      } else {
        MD3(OutputType, aoutput, output, ep.oh, ep.ow, ep.oc);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      }
    };

    __m<V> c0, c1, c2, c3, c4, c5, zero;

    __m<V> z2 = _mm<V>::set1_ps(2.0f);
    __m<V> z4 = _mm<V>::set1_ps(4.0f);
    __m<V> z8 = _mm<V>::set1_ps(8.0f);
    __m<V> z16 = _mm<V>::set1_ps(16.0f);
    __m<V> z1_2 = _mm<V>::set1_ps(1.0f / 2.0f);
    __m<V> z1_4 = _mm<V>::set1_ps(1.0f / 4.0f);
    __m<V> z1_8 = _mm<V>::set1_ps(1.0f / 8.0f);
    __m<V> z1_16 = _mm<V>::set1_ps(1.0f / 16.0f);

    __m<V> t00 = _mm<V>::load_ps(&md3(atoutput, 0, 0, 0));
    __m<V> t01 = _mm<V>::load_ps(&md3(atoutput, 0, 1, 0));
    __m<V> t02 = _mm<V>::load_ps(&md3(atoutput, 0, 2, 0));
    __m<V> t03 = _mm<V>::load_ps(&md3(atoutput, 0, 3, 0));
    __m<V> t04 = _mm<V>::load_ps(&md3(atoutput, 0, 4, 0));
    __m<V> t05 = _mm<V>::load_ps(&md3(atoutput, 0, 5, 0));
    __m<V> t10 = _mm<V>::load_ps(&md3(atoutput, 1, 0, 0));
    __m<V> t11 = _mm<V>::load_ps(&md3(atoutput, 1, 1, 0));
    __m<V> t12 = _mm<V>::load_ps(&md3(atoutput, 1, 2, 0));
    __m<V> t13 = _mm<V>::load_ps(&md3(atoutput, 1, 3, 0));
    __m<V> t14 = _mm<V>::load_ps(&md3(atoutput, 1, 4, 0));
    __m<V> t15 = _mm<V>::load_ps(&md3(atoutput, 1, 5, 0));
    __m<V> t20 = _mm<V>::load_ps(&md3(atoutput, 2, 0, 0));
    __m<V> t21 = _mm<V>::load_ps(&md3(atoutput, 2, 1, 0));
    __m<V> t22 = _mm<V>::load_ps(&md3(atoutput, 2, 2, 0));
    __m<V> t23 = _mm<V>::load_ps(&md3(atoutput, 2, 3, 0));
    __m<V> t24 = _mm<V>::load_ps(&md3(atoutput, 2, 4, 0));
    __m<V> t25 = _mm<V>::load_ps(&md3(atoutput, 2, 5, 0));
    __m<V> t30 = _mm<V>::load_ps(&md3(atoutput, 3, 0, 0));
    __m<V> t31 = _mm<V>::load_ps(&md3(atoutput, 3, 1, 0));
    __m<V> t32 = _mm<V>::load_ps(&md3(atoutput, 3, 2, 0));
    __m<V> t33 = _mm<V>::load_ps(&md3(atoutput, 3, 3, 0));
    __m<V> t34 = _mm<V>::load_ps(&md3(atoutput, 3, 4, 0));
    __m<V> t35 = _mm<V>::load_ps(&md3(atoutput, 3, 5, 0));
    __m<V> t40 = _mm<V>::load_ps(&md3(atoutput, 4, 0, 0));
    __m<V> t41 = _mm<V>::load_ps(&md3(atoutput, 4, 1, 0));
    __m<V> t42 = _mm<V>::load_ps(&md3(atoutput, 4, 2, 0));
    __m<V> t43 = _mm<V>::load_ps(&md3(atoutput, 4, 3, 0));
    __m<V> t44 = _mm<V>::load_ps(&md3(atoutput, 4, 4, 0));
    __m<V> t45 = _mm<V>::load_ps(&md3(atoutput, 4, 5, 0));
    __m<V> t50 = _mm<V>::load_ps(&md3(atoutput, 5, 0, 0));
    __m<V> t51 = _mm<V>::load_ps(&md3(atoutput, 5, 1, 0));
    __m<V> t52 = _mm<V>::load_ps(&md3(atoutput, 5, 2, 0));
    __m<V> t53 = _mm<V>::load_ps(&md3(atoutput, 5, 3, 0));
    __m<V> t54 = _mm<V>::load_ps(&md3(atoutput, 5, 4, 0));
    __m<V> t55 = _mm<V>::load_ps(&md3(atoutput, 5, 5, 0));
    __m<V> t60 = _mm<V>::load_ps(&md3(atoutput, 6, 0, 0));
    __m<V> t61 = _mm<V>::load_ps(&md3(atoutput, 6, 1, 0));
    __m<V> t62 = _mm<V>::load_ps(&md3(atoutput, 6, 2, 0));
    __m<V> t63 = _mm<V>::load_ps(&md3(atoutput, 6, 3, 0));
    __m<V> t64 = _mm<V>::load_ps(&md3(atoutput, 6, 4, 0));
    __m<V> t65 = _mm<V>::load_ps(&md3(atoutput, 6, 5, 0));

    AVX512_CALCULATE_O_0(0);
    AVX512_CALCULATE_O_0(1);
    AVX512_CALCULATE_O_0(2);
    AVX512_CALCULATE_O_0(3);
    AVX512_CALCULATE_O_0(4);
    AVX512_CALCULATE_O_0(5);
    AVX512_CALCULATE_O(0);
    p40 = p40 + t60 + t61 + t62 + t63 + t64 + t65;
    AVX512_ADD_B(0)

    AVX512_CALCULATE_O_1(0);
    AVX512_CALCULATE_O_1(1);
    AVX512_CALCULATE_O_1(2);
    AVX512_CALCULATE_O_1(3);
    AVX512_CALCULATE_O_1(4);
    AVX512_CALCULATE_O_1(5);
    AVX512_CALCULATE_O(1);
    p41 = p41 + z2 * (t62 - t63) + (z1_2 * (t64 - t65) + (t60 - t61));
    AVX512_ADD_B(1)

    AVX512_CALCULATE_O_2(0);
    AVX512_CALCULATE_O_2(1);
    AVX512_CALCULATE_O_2(2);
    AVX512_CALCULATE_O_2(3);
    AVX512_CALCULATE_O_2(4);
    AVX512_CALCULATE_O_2(5);
    AVX512_CALCULATE_O(2);
    p42 = p42 + (z4 * (t62 + t63) + (z1_4 * (t64 + t65) + (t60 + t61)));
    AVX512_ADD_B(2)

    AVX512_CALCULATE_O_3(0);
    AVX512_CALCULATE_O_3(1);
    AVX512_CALCULATE_O_3(2);
    AVX512_CALCULATE_O_3(3);
    AVX512_CALCULATE_O_3(4);
    AVX512_CALCULATE_O_3(5);
    AVX512_CALCULATE_O(3);
    p43 = p43 + (z8 * (t62 - t63) + (z1_8 * (t64 - t65) + (t60 - t61)));
    AVX512_ADD_B(3)

    __m<V> t06 = _mm<V>::load_ps(&md3(atoutput, 0, 6, 0));
    __m<V> t16 = _mm<V>::load_ps(&md3(atoutput, 1, 6, 0));
    __m<V> t26 = _mm<V>::load_ps(&md3(atoutput, 2, 6, 0));
    __m<V> t36 = _mm<V>::load_ps(&md3(atoutput, 3, 6, 0));
    __m<V> t46 = _mm<V>::load_ps(&md3(atoutput, 4, 6, 0));
    __m<V> t56 = _mm<V>::load_ps(&md3(atoutput, 5, 6, 0));
    __m<V> t66 = _mm<V>::load_ps(&md3(atoutput, 6, 6, 0));
    AVX512_CALCULATE_O_4(0);
    AVX512_CALCULATE_O_4(1);
    AVX512_CALCULATE_O_4(2);
    AVX512_CALCULATE_O_4(3);
    AVX512_CALCULATE_O_4(4);
    AVX512_CALCULATE_O_4(5);
    AVX512_CALCULATE_O(4);
    p44 = p44 + (z16 * (t62 + t63) + (z1_16 * (t64 + t65) + (t60 + t61 + t66)));
    AVX512_ADD_B(4)
  }
}; // elk_conv_wino_trans_output
} // namespace euler
