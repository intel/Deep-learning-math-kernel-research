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
template <int... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_output(elx_conv_t &xc,
    OutputType *output, TrOpType atoutput[A][A][V], BiasType *bias,
    int hOA_end, int wOA_end)
{
  constexpr int output_format = cd_traits<conditions...>::output_format;
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
    if (output_format == TKF_COMPACT) {
      MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else if (output_format == TKF_BLOCKED) {
      MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(OutputType, aoutput, output, xc.oh, xc.ow, xc.oc);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#undef t
#undef STORE

#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)
#define t(m, n) t##m##n

#define BIAS                                                      \
  std::is_same<BiasType, float>::value                            \
  ? *(__m<V>*)bias                                                \
  : _mm<V>::cvtph_ps(_mm<V/2>::load_si256((__m256i *)bias))

#define STORE(i, j)                                               \
  if (std::is_same<OutputType, float>::value)                     \
    _mm<V>::store_ps(P(i, j), p##j);                              \
  else {                                                          \
    auto f16 = _mm<V>::cvtps_ph(p##j,                             \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
    _mm<V/2>::store_si256((__m256i *)P(i, j), f16);               \
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
        p0 += BIAS;
        p1 += BIAS;
        p2 += BIAS;
        p3 += BIAS;
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
      STORE(i, 0)
      STORE(i, 1)
      STORE(i, 2)
      STORE(i, 3)
  }
}

} // namespace euler
