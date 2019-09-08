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

    __m<V> M[5][3];

    auto z0 = _mm<V>::set1_ps(2.0f);
    auto z1 = _mm<V>::set1_ps(0.5f);

    for (int _V = 0; _V < 16; ++_V) {
#pragma unroll
      for (int i = 0; i < 3; i++) {
        __m<V> f0, f1, f2;

        if (std::is_same<WeightsType, float>::value) {
          f0 = _mm<V>::load_ps(aweights[0][i][_V]);
          f1 = _mm<V>::load_ps(aweights[1][i][_V]);
          f2 = _mm<V>::load_ps(aweights[2][i][_V]);
        } else {
          f0 = _mm<V>::cvtph_ps(_mm<V / 2>::load_si256((__m256i *)aweights[0][i][_V]));
          f1 = _mm<V>::cvtph_ps(_mm<V / 2>::load_si256((__m256i *)aweights[1][i][_V]));
          f2 = _mm<V>::cvtph_ps(_mm<V / 2>::load_si256((__m256i *)aweights[2][i][_V]));
        }

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

        *(__m<V> *)atweights[i][0][_V] = t0;
        *(__m<V> *)atweights[i][1][_V] = f1 - t1;
        *(__m<V> *)atweights[i][2][_V] = f1 + t1;
        *(__m<V> *)atweights[i][3][_V] = f2 * z1 + t0 - f1;
        *(__m<V> *)atweights[i][4][_V] = f2;
      }
    }
  }
}; // elk_conv_wino_trans_weights
} // namespace euler
