#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename WeightsType, int V>
struct elk_conv_wino_trans_weights<float, WeightsType, ISA_SKX_AVX512,
    5, 3, V> {
  constexpr static int A = 5;
  constexpr static int K = 3;
  constexpr static int I = ISA_SKX_AVX512;

  static void execute(
      float atweights[A][A][V][V], WeightsType aweights[K][K][V][V])
  {
    ENABLE_AVX512F();

#undef F
#undef T
#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]

#undef f
#define f(m, n) f##m##n
#define LOAD(m, n)                                                             \
  std::is_same<WeightsType, float>::value                                      \
      ? _mm<V>::load_ps(F(m, n))                                               \
      : _mm<V>::cvtph_ps(_mm<V / 2>::load_si256((__m256i *)F(m, n)))

    __m<V> M[5][3];

    auto z0 = _mm<V>::set1_ps(2.0f);
    auto z1 = _mm<V>::set1_ps(0.5f);

    for (int _V = 0; _V < 16; ++_V) {
#pragma unroll
      for (int i = 0; i < 3; i++) {
        auto f0 = LOAD(0, i);
        auto f1 = LOAD(1, i);
        auto f2 = LOAD(2, i);

        auto t0 = f0 * z0;
        auto t1 = f0 + f2;

        M[0][i] = t0;
        M[1][i] = f1 - t1;
        M[2][i] = f1 + t1;
        M[3][i] = f2 * z1 + t0 - f1;
        M[4][i] = f2;
      }

#pragma unroll
      for (int i = 0; i < 5; i++) {
        auto f0 = M[i][0];
        auto f1 = M[i][1];
        auto f2 = M[i][2];

        auto t0 = f0 * z0;
        auto t1 = f0 + f2;

        *(__m<V> *)T(i, 0) = t0;
        *(__m<V> *)T(i, 1) = f1 - t1;
        *(__m<V> *)T(i, 2) = f1 + t1;
        *(__m<V> *)T(i, 3) = f2 * z1 + t0 - f1;
        *(__m<V> *)T(i, 4) = f2;
      }
    }
  }
}; // elk_conv_wino_trans_weights
} // namespace euler
