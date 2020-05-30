#pragma once

#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename WeightsType, int V>
struct elk_conv_wino_trans_weights<float, WeightsType, ISA_AVX512,
    6, 3, V> {
  constexpr static int A = 6;
  constexpr static int K = 3;
  constexpr static int I = ISA_AVX512;

  static void execute(
      float atweights[A][A][V][V], WeightsType aweights[K][K][V][V])
  {
    ENABLE_AVX512F();

    alignas(64) float M[6][3][16];

    auto z0 = _mm<V>::set1_ps(0.26890756302521f);
    auto z1 = _mm<V>::set1_ps(-0.688403361344538f);
    auto z2 = _mm<V>::set1_ps(0.119514472455649f);
    auto z3 = _mm<V>::set1_ps(1.13777777777778f);
    auto z4 = _mm<V>::set1_ps(0.430252100840336f);
    auto z5 = _mm<V>::set1_ps(0.179271708683473f);

    for (int _V = 0; _V < 16; _V++) {
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
        auto t0 = z0 * f2;
        auto t1 = z1 * f0 - t0;
        auto t2 = t0 + z2 * f0;

        *(__m<V> *)M[0][i] = z3 * f0;
        *(__m<V> *)M[1][i] = t1 - z4 * f1;
        *(__m<V> *)M[2][i] = t1 + z4 * f1;
        *(__m<V> *)M[3][i] = t2 + z5 * f1;
        *(__m<V> *)M[4][i] = t2 - z5 * f1;
        *(__m<V> *)M[5][i] = f2;
      }
#pragma unroll
      for (int i = 0; i < 6; i++) {
        auto f0 = _mm<V>::load_ps(M[i][0]);
        auto f1 = _mm<V>::load_ps(M[i][1]);
        auto f2 = _mm<V>::load_ps(M[i][2]);
        auto t0 = z0 * f2;
        auto t1 = z1 * f0 - t0;
        auto t2 = t0 + z2 * f0;

        *(__m<V> *)atweights[i][0][_V] = z3 * f0;
        *(__m<V> *)atweights[i][1][_V] = t1 - z4 * f1;
        *(__m<V> *)atweights[i][2][_V] = t1 + z4 * f1;
        *(__m<V> *)atweights[i][3][_V] = t2 + z5 * f1;
        *(__m<V> *)atweights[i][4][_V] = t2 - z5 * f1;
        *(__m<V> *)atweights[i][5][_V] = f2;
      }
    }
  }

}; // elk_conv_wino_trans_weights
} // namespace euler
