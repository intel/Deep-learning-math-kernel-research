#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_3x3_3x3_input.hxx"

namespace euler {

template <typename UserTypes, typename TrOpType, int V>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_weights(TrOpType atweights[A][A][V][V],
    WeightsType aweights[K][K][V][V])
{
#undef F
#undef T
#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]

#undef f
#define f(m, n) f##m##n
#define LOAD(m,n)                                                 \
  std::is_same<WeightsType, float>::value                         \
  ? _mm<V>::load_ps(F(m, n))                                      \
  : _mm<V>::cvtph_ps(_mm<V/2>::load_si256((__m256i *)F(m, n)))

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

      *(__m<V>*)T(i, 0) = t0;
      *(__m<V>*)T(i, 1) = f1 - t1;
      *(__m<V>*)T(i, 2) = f1 + t1;
      *(__m<V>*)T(i, 3) = f2 * z1 + t0 - f1;
      *(__m<V>*)T(i, 4) = f2;
    }
  }

}
} // namespace euler
