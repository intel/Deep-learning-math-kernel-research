#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_3x3_3x3_input.hxx"

// #ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
// #error "Don't include this file directly"
// #endif
namespace euler {

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_output(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A][V], BiasType *bias,
    int hOA_end, int wOA_end)
{
  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) OutputType dummy[V];
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
#undef STORE

#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)
#define t(m, n) t##m##n
#define OP(m,n) t(m,n) = _mm<V>::load_ps(T(m, n))

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

  __m<V> M[3][5];

  auto z0 = _mm<V>::set1_ps(0.3333333333333333f);
  auto z1 = _mm<V>::set1_ps(0.6666666666666666f);
  auto z2 = _mm<V>::set1_ps(1.3333333333333333f);
  __m<V> z = XOR(z, z);

#pragma unroll
  for (int i = 0; i < 5; i++) {
    auto f0 = _mm<V>::load_ps(T(0, i));
    auto f1 = _mm<V>::load_ps(T(1, i));
    auto f2 = _mm<V>::load_ps(T(2, i));
    auto f3 = _mm<V>::load_ps(T(3, i));
    auto f4 = _mm<V>::load_ps(T(4, i));

    auto t0 = f2 * z0;
    auto t1 = t0 + f1;

    M[0][i] = f3 * z2 + f0 + t1;
    M[1][i] = t0 - f1 - f3 * z1;
    M[2][i] = f3 * z0 + f4 + t1;
  }

#pragma unroll
  for (int i = 0; i < 3; i++) {
    auto f0 = M[i][0];
    auto f1 = M[i][1];
    auto f2 = M[i][2];
    auto f3 = M[i][3];
    auto f4 = M[i][4];

    auto t0 = f2 * z0;
    auto t1 = t0 + f1;

    auto p0 = f3 * z2 + f0 + t1;
    auto p1 = t0 - f1 - f3 * z1;
    auto p2 = f3 * z0 + f4 + t1;

    if (fuse_bias) {
      p0 += BIAS;
      p1 += BIAS;
      p2 += BIAS;
    }
    if (fuse_ip_sum) {
      p0 += *(__m<V> *)P(i, 0);
      p1 += *(__m<V> *)P(i, 1);
      p2 += *(__m<V> *)P(i, 2);
    }
    if (fuse_relu) {
      p0 = MAX(p0, z);
      p1 = MAX(p1, z);
      p2 = MAX(p2, z);
    }
    STORE(i, 0)
    STORE(i, 1)
    STORE(i, 2)
  }
}

} // namespace euler
