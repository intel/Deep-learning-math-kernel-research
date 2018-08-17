#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

#define GENERIC_CALCULATE_I(n)                                                  \
  T(0, n) = C(1) + z2 * C(2) - C(3);                                            \
  T(1, n) = z3 * C(2) - z2 * C(1) - C(3);                                       \
  T(2, n) = z2 * C(1) + C(2) - C(3);                                            \
  T(3, n) = C(1) - C(3);                                                        \
  T(4, n) = - z2 * C(1) + C(2) + z2 * C(3);

template <>
class convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3> {
  template<typename Type, int ...configs>
    friend class convolution_winograd_kernel_base;
protected:
  constexpr static int I = ISA_GENERIC;
  constexpr static int V = 16;
  constexpr static int A = 5;
  constexpr static int K = 3;

  template <bool is_border>
  static inline void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end);

  template <bool is_border>
  static inline void __trans_inputa(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int _wA, int _hA_start, int _hA_end, int _wA_start,
      int _wA_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<float> &xc, float *output,
      float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<float> &xc, float *toutputa,
      float *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<float> &xc, float *output,
      float atoutputa[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]);
};

template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::__trans_input(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  const float z2 = 2.0f;
  const float z3 = 3.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

  auto f_cb = [&](int _h, int _w, int _V) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, V);
      return md3(ainput, _h, _w, _V);
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return 0.0f;
      else
        return md3(ainput, _h, _w, _V);
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w, _V)
#define C(n) C##n[_V]
#define T(_h, _w) atinput[_w][_h][_V]

  float C1[V], C2[V], C3[V];
# pragma omp simd
  for (int _V = 0; _V < V; ++_V) {
    C(1) = F(1, 1) + z2 * F(1, 2) - z2 * F(1, 0) - F(1, 3);
    C(2) = F(2, 1) + z2 * F(2, 2) - z2 * F(2, 0) - F(2, 3);
    C(3) = F(3, 1) + z2 * F(3, 2) - z2 * F(3, 0) - F(3, 3);
    GENERIC_CALCULATE_I(0)
    T(0, 0) += z4 * F(0, 0) - z2 * F(0, 1) - z4 * F(0, 2) + z2 * F(0, 3);
    T(4, 0) += z2 * F(4, 0) - F(4, 1) - z2 * F(4, 2) + F(4, 3);

    C(1) = z3 * F(1, 2) - z2 * F(1, 1) - F(1, 3);
    C(2) = z3 * F(2, 2) - z2 * F(2, 1) - F(2, 3);
    C(3) = z3 * F(3, 2) - z2 * F(3, 1) - F(3, 3);
    GENERIC_CALCULATE_I(1)
    T(0, 1) += z4 * F(0, 1) - z6 * F(0, 2) + z2 * F(0, 3);
    T(4, 1) += z2 * F(4, 1) - z3 * F(4, 2) + F(4, 3);

    C(1) = z2 * F(1, 1) + F(1, 2) - F(1, 3);
    C(2) = z2 * F(2, 1) + F(2, 2) - F(2, 3);
    C(3) = z2 * F(3, 1) + F(3, 2) - F(3, 3);
    GENERIC_CALCULATE_I(2)
    T(0, 2) += z2 * F(0, 3) - z2 * F(0, 2) - z4 * F(0, 1);
    T(4, 2) += F(4, 3) - z2 * F(4, 1) - F(4, 2);

    C(1) = F(1, 1) - F(1, 3);
    C(2) = F(2, 1) - F(2, 3);
    C(3) = F(3, 1) - F(3, 3);
    GENERIC_CALCULATE_I(3)
    T(0, 3) += z2 * F(0, 3) - z2 * F(0, 1);
    T(4, 3) += F(4, 3) - F(4, 1);

    C(1) = F(1, 2) + z2 * F(1, 3) - z2 * F(1, 1) - F(1, 4);
    C(2) = F(2, 2) + z2 * F(2, 3) - z2 * F(2, 1) - F(2, 4);
    C(3) = F(3, 2) + z2 * F(3, 3) - z2 * F(3, 1) - F(3, 4);
    GENERIC_CALCULATE_I(4)
    T(0, 4) += z4 * F(0, 1) - z2 * F(0, 2) - z4 * F(0, 3) + z2 * F(0, 4);
    T(4, 4) += z2 * F(4, 1) - F(4, 2) - z2 * F(4, 3) + F(4, 4);
  }
}

template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::
__trans_inputa(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  const float z2 = 2.0f;
  const float z3 = 3.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

  auto f_cb = [&](int _h, int _w, int _V) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, V);
      return md3(ainput, _h, _w, _V);
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return 0.0f;
      else
        return md3(ainput, _h, _w, _V);
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w, _V)
#define C(n) C##n[_V]
#define T(_h, _w) atinput[_h][_w][_V]

  float C1[V], C2[V], C3[V];
  switch (wA) {
  case 0:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = F(1, 1) + z2 * F(1, 2) - z2 * F(1, 0) - F(1, 3);
      C(2) = F(2, 1) + z2 * F(2, 2) - z2 * F(2, 0) - F(2, 3);
      C(3) = F(3, 1) + z2 * F(3, 2) - z2 * F(3, 0) - F(3, 3);
      GENERIC_CALCULATE_I(0)
      T(0, 0) += z4 * F(0, 0) - z2 * F(0, 1) - z4 * F(0, 2) + z2 * F(0, 3);
      T(4, 0) += z2 * F(4, 0) - F(4, 1) - z2 * F(4, 2) + F(4, 3);
    }

    break;

  case 1:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = z3 * F(1, 2) - z2 * F(1, 1) - F(1, 3);
      C(2) = z3 * F(2, 2) - z2 * F(2, 1) - F(2, 3);
      C(3) = z3 * F(3, 2) - z2 * F(3, 1) - F(3, 3);
      GENERIC_CALCULATE_I(1)
      T(0, 1) += z4 * F(0, 1) - z6 * F(0, 2) + z2 * F(0, 3);
      T(4, 1) += z2 * F(4, 1) - z3 * F(4, 2) + F(4, 3);
    }

    break;

  case 2:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = z2 * F(1, 1) + F(1, 2) - F(1, 3);
      C(2) = z2 * F(2, 1) + F(2, 2) - F(2, 3);
      C(3) = z2 * F(3, 1) + F(3, 2) - F(3, 3);
      GENERIC_CALCULATE_I(2)
      T(0, 2) += z2 * F(0, 3) - z2 * F(0, 2) - z4 * F(0, 1);
      T(4, 2) += F(4, 3) - z2 * F(4, 1) - F(4, 2);
    }

    break;

  case 3:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = F(1, 1) - F(1, 3);
      C(2) = F(2, 1) - F(2, 3);
      C(3) = F(3, 1) - F(3, 3);
      GENERIC_CALCULATE_I(3)
      T(0, 3) += z2 * F(0, 3) - z2 * F(0, 1);
      T(4, 3) += F(4, 3) - F(4, 1);
    }

    break;

  case 4:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = F(1, 2) + z2 * F(1, 3) - z2 * F(1, 1) - F(1, 4);
      C(2) = F(2, 2) + z2 * F(2, 3) - z2 * F(2, 1) - F(2, 4);
      C(3) = F(3, 2) + z2 * F(3, 3) - z2 * F(3, 1) - F(3, 4);
      GENERIC_CALCULATE_I(4)
      T(0, 4) += z4 * F(0, 1) - z2 * F(0, 2) - z4 * F(0, 3) + z2 * F(0, 4);
      T(4, 4) += z2 * F(4, 1) - F(4, 2) - z2 * F(4, 3) + F(4, 4);
    }

    break;
  }
}
} // namespace euler
