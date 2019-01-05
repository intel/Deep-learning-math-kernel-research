#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename UserTypes, typename TrOpType, int v>
class convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, v, 5, 3> {
  protected:
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int V = 16;
  constexpr static int A = 5;
  constexpr static int K = 3;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V],
      InputType *input, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<UserTypes> &xc, OutputType *output,
      TrOpType atoutput[A][A][V], BiasType *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(
      TrOpType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]);
};

template <typename UserTypes, typename TrOpType, int V>
template <bool is_border>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 5, 3>::__trans_input(
      elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V], InputType *input,
      int hT_start, int hT_end, int wT_start, int wT_end)
{
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
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return _mm<V>::setzero_ps();
      else if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
        return _mm<V>::cvtph_ps(f16);
      }
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)

  __m<V> M[5][5];

  auto z0 = _mm<V>::set1_ps(0.5f);
  auto z1 = _mm<V>::set1_ps(1.5f);

#pragma unroll
  for (int i = 0; i < 5; i++) {
    auto f0 = F(0, i);
    auto f1 = F(1, i);
    auto f2 = F(2, i);
    auto f3 = F(3, i);
    auto f4 = F(4, i);

    auto t0 = f1 * z0;
    auto t1 = f2 * z0;
    auto t2 = f3 - f1;

    M[0][i] = f0 * z0 - t1 - t2;
    M[1][i] = f3 - t0 - t1;
    M[2][i] = f2 * z1 + f3 + t0;
    M[3][i] = t2;
    M[4][i] = f3 * z0 + f4 - f2 - t0;
  }

#pragma unroll
  for (int i = 0; i < 5; i++) {
    auto f0 = M[i][0];
    auto f1 = M[i][1];
    auto f2 = M[i][2];
    auto f3 = M[i][3];
    auto f4 = M[i][4];

    auto t0 = f1 * z0;
    auto t1 = f2 * z0;
    auto t2 = f3 - f1;

    *(__m<V>*)T(i, 0) = f0 * z0 - t1 - t2;
    *(__m<V>*)T(i, 1) = f3 - t0 - t1;
    *(__m<V>*)T(i, 2) = f2 * z1 + f3 + t0;
    *(__m<V>*)T(i, 3) = t2;
    *(__m<V>*)T(i, 4) = f3 * z0 + f4 - f2 - t0;
  }
}

} // namespace euler
