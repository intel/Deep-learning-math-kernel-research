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
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 6, 3>::__trans_weights(TrOpType atweights[A][A][V][V],
    WeightsType aweights[K][K][V][V])
{
#undef F
#undef T

#undef F
#undef T
#undef f

#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]
#define f(m, n) f##m##n

#define LOAD(m,n)                                                 \
  std::is_same<WeightsType, float>::value                         \
  ? _mm<V>::load_ps(F(m, n))                                      \
  : _mm<V>::cvtph_ps(_mm<V/2>::load_si256((__m256i *)F(m, n)))

  float M[6][3][16];

  auto z0 = _mm<V>::set1_ps(0.26890756302521f);
  auto z1 = _mm<V>::set1_ps(-0.688403361344538f);
  auto z2 = _mm<V>::set1_ps(0.119514472455649f);
  auto z3 = _mm<V>::set1_ps(1.13777777777778f);
  auto z4 = _mm<V>::set1_ps(0.430252100840336f);
  auto z5 = _mm<V>::set1_ps(0.179271708683473f);

  for (int _V = 0; _V < 16; _V++) {
#pragma unroll
    for (int i = 0; i < 3; i++) {
      auto f0 = LOAD(0, i);
      auto f1 = LOAD(1, i);
      auto f2 = LOAD(2, i);
      auto t0 = z0 * f2;
      auto t1 = z1 * f0 - t0;
      auto t2 = t0 + z2 * f0;

      *(__m<V>*)M[0][i] = z3 * f0;
      *(__m<V>*)M[1][i] = t1 - z4 * f1;
      *(__m<V>*)M[2][i] = t1 + z4 * f1;
      *(__m<V>*)M[3][i] = t2 + z5 * f1;
      *(__m<V>*)M[4][i] = t2 - z5 * f1;
      *(__m<V>*)M[5][i] = f2;
    }
#pragma unroll
    for (int i = 0; i < 6; i++) {
      auto f0 = _mm<V>::load_ps(M[i][0]);
      auto f1 = _mm<V>::load_ps(M[i][1]);
      auto f2 = _mm<V>::load_ps(M[i][2]);
      auto t0 = z0 * f2;
      auto t1 = z1 * f0 - t0;
      auto t2 = t0 + z2 * f0;

      *(__m<V>*)T(i, 0) = z3 * f0;
      *(__m<V>*)T(i, 1) = t1 - z4 * f1;
      *(__m<V>*)T(i, 2) = t1 + z4 * f1;
      *(__m<V>*)T(i, 3) = t2 + z5 * f1;
      *(__m<V>*)T(i, 4) = t2 - z5 * f1;
      *(__m<V>*)T(i, 5) = f2;
    }
  }
}

} // namespace euler
