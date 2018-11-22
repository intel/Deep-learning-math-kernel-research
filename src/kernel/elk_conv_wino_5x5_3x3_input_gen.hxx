#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include <math.h>

namespace euler {

#define GENERIC_CALCULATE_I_5(n)                                                \
  T(0, n) = C(1) + C(2) - z17_10 * C(3) - z17_5 * C(4) + C(5) + C(6);           \
  T(1, n) = C(1) - C(2) - z17_10 * C(3) + z17_5 * C(4) + C(5) - C(6);           \
  T(2, n) = - z1_2 * C(1) - z1_4 * C(2) + C(3) + C(4) - z2 * C(5) - C(6);       \
  T(3, n) = z1_2 * C(1) - z1_4 * C(2) - C(3) + C(4) + z2 * C(5) - C(6);         \
  T(4, n) = - z2 * C(1) + C(3) + z4 * (C(4) - C(2)) - z1_2 * C(5) - C(6);       \
  T(5, n) = z2 * C(1) - C(3) + z4 * (C(4) - C(2)) + z1_2 * C(5) - C(6);         \
  T(6, n) = C(1) - z21_10 * C(3) + z21_4 * C(5);

template <>
class convolution_winograd_kernel_base<conv::FP32, float, ISA_GENERIC, 16, 7, 3> {
protected:
  constexpr static int I = ISA_GENERIC;
  constexpr static int V = 16;
  constexpr static int A = 7;
  constexpr static int K = 3;

  template <bool is_border>
  static inline void __trans_input(elx_conv_t<conv::FP32> &xc,
      float atinput[A][A][V], float *input, int hT_start, int hT_end, int wT_start,
      int wT_end);

  template <bool is_border>
  static inline void __trans_inputa(elx_conv_t<conv::FP32> &xc,
      float atinput[A][A][V], float *input, int _wA, int _hA_start, int _hA_end,
      int _wA_start, int _wA_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<conv::FP32> &xc,
      float *output, float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<conv::FP32> &xc,
      float *toutputa, float *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<conv::FP32> &xc,
      float *output, float atoutputa[A][A - K + 1][V], float *bias, int hOA_end,
      int wOA_end);

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]);
};

template <bool is_border>
inline void convolution_winograd_kernel_base<
    conv::FP32, float, ISA_GENERIC, 16, 7, 3>::__trans_input(
      elx_conv_t<conv::FP32> &xc, float atinput[A][A][V], float *input,
      int hT_start, int hT_end, int wT_start, int wT_end) {
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z1_2 = 1.0f / 2.0f;
  const float z1_4 = 1.0f / 4.0f;
  const float z5_2 = 5.0f / 2.0f;
  const float z5_4 = 5.0f / 4.0f;
  const float z17_4 = 17.0f / 4.0f;
  const float z17_5 = 17.0f / 5.0f;
  const float z17_10 = 17.0f / 10.0f;
  const float z21_4 = 21.0f / 4.0f;
  const float z21_10 = 21.0f / 10.0f;
  const float z85_8 = 85.0f / 8.0f;
  const float z85_16 = 85.0f / 16.0f;

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

  float C1[V], C2[V], C3[V], C4[V], C5[V], C6[V];

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(0, 0) + F(0, 1) - z17_4 * (F(0, 2) + F(0, 3)) + F(0, 4) + F(0, 5);
    C(2) = F(1, 0) + F(1, 1) - z17_4 * (F(1, 2) + F(1, 3)) + F(1, 4) + F(1, 5);
    C(3) = z5_2 * (F(2, 0) + F(2, 1) + F(2, 4) + F(2,5)) - z85_8 * (F(2, 2) + F(2, 3));
    C(4) = z5_4 * (F(3, 0) + F(3, 1) + F(3, 4) + F(3,5)) - z85_16 * (F(3, 2) + F(3, 3));
    C(5) = F(4, 0) + F(4, 1) - z17_4 * (F(4, 2) + F(4, 3)) + F(4, 4) + F(4, 5);
    C(6) = F(5, 0) + F(5, 1) - z17_4 * (F(5, 2) + F(5, 3)) + F(5, 4) + F(5, 5);

    GENERIC_CALCULATE_I_5(0)
    T(6, 0) += - F(6, 0) - F (6, 1) + z17_4 * (F(6, 2) + F(6, 3)) - F(6, 4) - F(6, 5);



    C(1) = F(0, 0) - F(0, 1) + z17_4 * (F(0, 3) - F(0, 2)) + F(0, 4) - F(0, 5);
    C(2) = F(1, 0) - F(1, 1) + z17_4 * (F(1, 3) - F(1, 2)) + F(1, 4) - F(1, 5);
    C(3) = z5_2 * (F(2, 0) - F(2, 1) + F(2, 4) - F(2,5)) + z85_8 * (F(2, 3) - F(2, 2));
    C(4) = z5_4 * (F(3, 0) - F(3, 1) + F(3, 4) - F(3,5)) + z85_16 * (F(3, 3) - F(3, 2));
    C(5) = F(4, 0) - F(4, 1) + z17_4 * (F(4, 3) - F(4, 2)) + F(4, 4) - F(4, 5);
    C(6) = F(5, 0) - F(5, 1) + z17_4 * (F(5, 3) - F(5, 2)) + F(5, 4) - F(5, 5);

    GENERIC_CALCULATE_I_5(1)
    T(6, 1) += - F(6, 0) + F (6, 1) + z17_4 * (F(6, 2) - F(6, 3)) - F(6, 4) + F(6, 5);



    const float z5 = 5.0f;
    const float z5_8 = 5.0f / 8.0f;
    const float z5_16 = 5.0f / 16.0f;
    const float z25_4 = 25.0f / 4.0f;
    const float z25_8 = 25.0f / 8.0f;
    const float z25_16 = 25.0f / 16.0f;

    C(1) = - z1_2 * F(0, 0) - z1_4 * F(0, 1) + z5_2 * F(0, 2) + z5_4 * F(0, 3)
        - z2 * F(0, 4) - F(0, 5);
    C(2) = - z1_2 * F(1, 0) - z1_4 * F(1, 1) + z5_2 * F(1, 2) + z5_4 * F(1, 3)
        - z2 * F(1, 4) - F(1, 5);
    C(3) = - z5_4 * F(2, 0) - z5_8 * F(2, 1) + z25_4 * F(2, 2) + z25_8 * F(2, 3)
        - z5 * F(2, 4) - z5_2 * F(2, 5);
    C(4) = - z5_8 * F(3, 0) - z5_16 * F(3, 1) + z25_8 * F(3, 2) + z25_16 * F(3, 3)
        - z5_2 * F(3, 4) - z5_4 * F(3, 5);
    C(5) = - z1_2 * F(4, 0) - z1_4 * F(4, 1) + z5_2 * F(4, 2) + z5_4 * F(4, 3)
        - z2 * F(4, 4) - F(4, 5);
    C(6) = - z1_2 * F(5, 0) - z1_4 * F(5, 1) + z5_2 * F(5, 2) + z5_4 * F(5, 3)
        - z2 * F(5, 4) - F(5, 5);

    GENERIC_CALCULATE_I_5(2)
    T(6, 2) += z1_2 * F(6, 0) + z1_4 * F (6, 1) - z5_2 * F(6, 2) - z5_4 * F(6, 3)
        + z2 * F(6, 4) + F(6, 5);



    C(1) = z1_2 * F(0, 0) - z1_4 * F(0, 1) - z5_2 * F(0, 2) + z5_4 * F(0, 3)
        + z2 * F(0, 4) - F(0, 5);
    C(2) = z1_2 * F(1, 0) - z1_4 * F(1, 1) - z5_2 * F(1, 2) + z5_4 * F(1, 3)
        + z2 * F(1, 4) - F(1, 5);
    C(3) = z5_4 * F(2, 0) - z5_8 * F(2, 1) - z25_4 * F(2, 2) + z25_8 * F(2, 3)
        + z5 * F(2, 4) - z5_2 * F(2, 5);
    C(4) = z5_8 * F(3, 0) - z5_16 * F(3, 1) - z25_8 * F(3, 2) + z25_16 * F(3, 3)
        + z5_2 * F(3, 4) - z5_4 * F(3, 5);
    C(5) = z1_2 * F(4, 0) - z1_4 * F(4, 1) - z5_2 * F(4, 2) + z5_4 * F(4, 3)
        + z2 * F(4, 4) - F(4, 5);
    C(6) = z1_2 * F(5, 0) - z1_4 * F(5, 1) - z5_2 * F(5, 2) + z5_4 * F(5, 3)
        + z2 * F(5, 4) - F(5, 5);

    GENERIC_CALCULATE_I_5(3)
    T(6, 3) += - z1_2 * F(6, 0) + z1_4 * F (6, 1) + z5_2 * F(6, 2) - z5_4 * F(6, 3)
        - z2 * F(6, 4) + F(6, 5);



    const float z10 = 10.0f;
    const float z25_2 = 25.0f / 2.0f;

    C(1) = - z2 * F(0, 0) - z4 * F(0, 1) + z5_2 * F(0, 2) + z5 * F(0, 3)
        - z1_2 * F(0, 4) - F(0, 5);
    C(2) = - z2 * F(1, 0) - z4 * F(1, 1) + z5_2 * F(1, 2) + z5 * F(1, 3)
        - z1_2 * F(1, 4) - F(1, 5);
    C(3) = - z5 * F(2, 0) - z10 * F(2, 1) + z25_4 * F(2, 2) + z25_2 * F(2, 3)
        - z5_4 * F(2, 4) - z5_2 * F(2, 5);
    C(4) = - z5_2 * F(3, 0) - z5 * F(3, 1) + z25_8 * F(3, 2) + z25_4 * F(3, 3)
        - z5_8 * F(3, 4) - z5_4 * F(3, 5);
    C(5) = - z2 * F(4, 0) - z4 * F(4, 1) + z5_2 * F(4, 2) + z5 * F(4, 3)
        - z1_2 * F(4, 4) - F(4, 5);
    C(6) = - z2 * F(5, 0) - z4 * F(5, 1) + z5_2 * F(5, 2) + z5 * F(5, 3)
        - z1_2 * F(5, 4) - F(5, 5);

    GENERIC_CALCULATE_I_5(4)
    T(6, 4) += z2 * F(6, 0) + z4 * F (6, 1) - z5_2 * F(6, 2) - z5 * F(6, 3)
        + z1_2 * F(6, 4) + F(6, 5);



    C(1) = z2 * F(0, 0) - z4 * F(0, 1) - z5_2 * F(0, 2) + z5 * F(0, 3)
        + z1_2 * F(0, 4) - F(0, 5);
    C(2) = z2 * F(1, 0) - z4 * F(1, 1) - z5_2 * F(1, 2) + z5 * F(1, 3)
        + z1_2 * F(1, 4) - F(1, 5);
    C(3) = z5 * F(2, 0) - z10 * F(2, 1) - z25_4 * F(2, 2) + z25_2 * F(2, 3)
        + z5_4 * F(2, 4) - z5_2 * F(2, 5);
    C(4) = z5_2 * F(3, 0) - z5 * F(3, 1) - z25_8 * F(3, 2) + z25_4 * F(3, 3)
        + z5_8 * F(3, 4) - z5_4 * F(3, 5);
    C(5) = z2 * F(4, 0) - z4 * F(4, 1) - z5_2 * F(4, 2) + z5 * F(4, 3)
        + z1_2 * F(4, 4) - F(4, 5);
    C(6) = z2 * F(5, 0) - z4 * F(5, 1) - z5_2 * F(5, 2) + z5 * F(5, 3)
        + z1_2 * F(5, 4) - F(5, 5);

    GENERIC_CALCULATE_I_5(5)
    T(6, 5) += - z2 * F(6, 0) + z4 * F (6, 1) + z5_2 * F(6, 2) - z5 * F(6, 3)
        - z1_2 * F(6, 4) + F(6, 5);



    const float z105_8 = 105.0f / 8.0f;
    const float z105_16 = 105.0f / 16.0f;

    C(1) = F(0, 0) + z21_4 * (F(0, 4) - F(0, 2)) - F(0, 6);
    C(2) = F(1, 0) + z21_4 * (F(1, 4) - F(1, 2)) - F(1, 6);
    C(3) = z5_2 * (F(2, 0) - F(2, 6)) + z105_8 * (F(2, 4) - F(2, 2));
    C(4) = z5_4 * (F(3, 0) - F(3, 6)) + z105_16 * (F(3, 4) - F(3, 2));
    C(5) = F(4, 0) + z21_4 * (F(4, 4) - F(4, 2)) - F(4, 6);
    C(6) = F(5, 0) + z21_4 * (F(5, 4) - F(5, 2)) - F(5, 6);

    GENERIC_CALCULATE_I_5(6)
    T(6, 6) += - F(6, 0) + z21_4 * (F(6, 2) - F(6, 4)) + F(6, 6);
  }
}

template <bool is_border>
inline void convolution_winograd_kernel_base<conv::FP32, float, ISA_GENERIC, 16, 7, 3>::
__trans_inputa(
    elx_conv_t<conv::FP32> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z1_2 = 1.0f / 2.0f;
  const float z1_4 = 1.0f / 4.0f;
  const float z5_2 = 5.0f / 2.0f;
  const float z5_4 = 5.0f / 4.0f;
  const float z17_4 = 17.0f / 4.0f;
  const float z17_5 = 17.0f / 5.0f;
  const float z17_10 = 17.0f / 10.0f;
  const float z21_4 = 21.0f / 4.0f;
  const float z21_10 = 21.0f / 10.0f;
  const float z85_8 = 85.0f / 8.0f;
  const float z85_16 = 85.0f / 16.0f;

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

  float C1[V], C2[V], C3[V], C4[V], C5[V], C6[V];
  switch (wA) {
  case 0:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = F(0, 0) + F(0, 1) - z17_4 * (F(0, 2) + F(0, 3)) + F(0, 4) + F(0, 5);
      C(2) = F(1, 0) + F(1, 1) - z17_4 * (F(1, 2) + F(1, 3)) + F(1, 4) + F(1, 5);
      C(3) = z5_2 * (F(2, 0) + F(2, 1) + F(2, 4) + F(2,5)) - z85_8 * (F(2, 2) + F(2, 3));
      C(4) = z5_4 * (F(3, 0) + F(3, 1) + F(3, 4) + F(3,5)) - z85_16 * (F(3, 2) + F(3, 3));
      C(5) = F(4, 0) + F(4, 1) - z17_4 * (F(4, 2) + F(4, 3)) + F(4, 4) + F(4, 5);
      C(6) = F(5, 0) + F(5, 1) - z17_4 * (F(5, 2) + F(5, 3)) + F(5, 4) + F(5, 5);

      GENERIC_CALCULATE_I_5(0)
      T(6, 0) += - F(6, 0) - F (6, 1) + z17_4 * (F(6, 2) + F(6, 3)) - F(6, 4) - F(6, 5);
    }

    break;
  case 1:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      C(1) = F(0, 0) - F(0, 1) + z17_4 * (F(0, 3) - F(0, 2)) + F(0, 4) - F(0, 5);
      C(2) = F(1, 0) - F(1, 1) + z17_4 * (F(1, 3) - F(1, 2)) + F(1, 4) - F(1, 5);
      C(3) = z5_2 * (F(2, 0) - F(2, 1) + F(2, 4) - F(2,5)) + z85_8 * (F(2, 3) - F(2, 2));
      C(4) = z5_4 * (F(3, 0) - F(3, 1) + F(3, 4) - F(3,5)) + z85_16 * (F(3, 3) - F(3, 2));
      C(5) = F(4, 0) - F(4, 1) + z17_4 * (F(4, 3) - F(4, 2)) + F(4, 4) - F(4, 5);
      C(6) = F(5, 0) - F(5, 1) + z17_4 * (F(5, 3) - F(5, 2)) + F(5, 4) - F(5, 5);

      GENERIC_CALCULATE_I_5(1)
      T(6, 1) += - F(6, 0) + F (6, 1) + z17_4 * (F(6, 2) - F(6, 3)) - F(6, 4) + F(6, 5);
    }

    break;
  case 2:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      const float z5 = 5.0f;
      const float z5_8 = 5.0f / 8.0f;
      const float z5_16 = 5.0f / 16.0f;
      const float z25_4 = 25.0f / 4.0f;
      const float z25_8 = 25.0f / 8.0f;
      const float z25_16 = 25.0f / 16.0f;

      C(1) = - z1_2 * F(0, 0) - z1_4 * F(0, 1) + z5_2 * F(0, 2) + z5_4 * F(0, 3)
          - z2 * F(0, 4) - F(0, 5);
      C(2) = - z1_2 * F(1, 0) - z1_4 * F(1, 1) + z5_2 * F(1, 2) + z5_4 * F(1, 3)
          - z2 * F(1, 4) - F(1, 5);
      C(3) = - z5_4 * F(2, 0) - z5_8 * F(2, 1) + z25_4 * F(2, 2) + z25_8 * F(2, 3)
          - z5 * F(2, 4) - z5_2 * F(2, 5);
      C(4) = - z5_8 * F(3, 0) - z5_16 * F(3, 1) + z25_8 * F(3, 2) + z25_16 * F(3, 3)
          - z5_2 * F(3, 4) - z5_4 * F(3, 5);
      C(5) = - z1_2 * F(4, 0) - z1_4 * F(4, 1) + z5_2 * F(4, 2) + z5_4 * F(4, 3)
          - z2 * F(4, 4) - F(4, 5);
      C(6) = - z1_2 * F(5, 0) - z1_4 * F(5, 1) + z5_2 * F(5, 2) + z5_4 * F(5, 3)
          - z2 * F(5, 4) - F(5, 5);

      GENERIC_CALCULATE_I_5(2)
      T(6, 2) += z1_2 * F(6, 0) + z1_4 * F (6, 1) - z5_2 * F(6, 2) - z5_4 * F(6, 3)
          + z2 * F(6, 4) + F(6, 5);
    }

    break;
  case 3:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      const float z5 = 5.0f;
      const float z5_8 = 5.0f / 8.0f;
      const float z5_16 = 5.0f / 16.0f;
      const float z25_4 = 25.0f / 4.0f;
      const float z25_8 = 25.0f / 8.0f;
      const float z25_16 = 25.0f / 16.0f;

      C(1) = z1_2 * F(0, 0) - z1_4 * F(0, 1) - z5_2 * F(0, 2) + z5_4 * F(0, 3)
          + z2 * F(0, 4) - F(0, 5);
      C(2) = z1_2 * F(1, 0) - z1_4 * F(1, 1) - z5_2 * F(1, 2) + z5_4 * F(1, 3)
          + z2 * F(1, 4) - F(1, 5);
      C(3) = z5_4 * F(2, 0) - z5_8 * F(2, 1) - z25_4 * F(2, 2) + z25_8 * F(2, 3)
          + z5 * F(2, 4) - z5_2 * F(2, 5);
      C(4) = z5_8 * F(3, 0) - z5_16 * F(3, 1) - z25_8 * F(3, 2) + z25_16 * F(3, 3)
          + z5_2 * F(3, 4) - z5_4 * F(3, 5);
      C(5) = z1_2 * F(4, 0) - z1_4 * F(4, 1) - z5_2 * F(4, 2) + z5_4 * F(4, 3)
          + z2 * F(4, 4) - F(4, 5);
      C(6) = z1_2 * F(5, 0) - z1_4 * F(5, 1) - z5_2 * F(5, 2) + z5_4 * F(5, 3)
          + z2 * F(5, 4) - F(5, 5);

      GENERIC_CALCULATE_I_5(3)
      T(6, 3) += - z1_2 * F(6, 0) + z1_4 * F (6, 1) + z5_2 * F(6, 2) - z5_4 * F(6, 3)
          - z2 * F(6, 4) + F(6, 5);
    }

    break;
  case 4:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      const float z5 = 5.0f;
      const float z10 = 10.0f;
      const float z5_8 = 5.0f / 8.0f;
      const float z25_2 = 25.0f / 2.0f;
      const float z25_4 = 25.0f / 4.0f;
      const float z25_8 = 25.0f / 8.0f;

      C(1) = - z2 * F(0, 0) - z4 * F(0, 1) + z5_2 * F(0, 2) + z5 * F(0, 3)
          - z1_2 * F(0, 4) - F(0, 5);
      C(2) = - z2 * F(1, 0) - z4 * F(1, 1) + z5_2 * F(1, 2) + z5 * F(1, 3)
          - z1_2 * F(1, 4) - F(1, 5);
      C(3) = - z5 * F(2, 0) - z10 * F(2, 1) + z25_4 * F(2, 2) + z25_2 * F(2, 3)
          - z5_4 * F(2, 4) - z5_2 * F(2, 5);
      C(4) = - z5_2 * F(3, 0) - z5 * F(3, 1) + z25_8 * F(3, 2) + z25_4 * F(3, 3)
          - z5_8 * F(3, 4) - z5_4 * F(3, 5);
      C(5) = - z2 * F(4, 0) - z4 * F(4, 1) + z5_2 * F(4, 2) + z5 * F(4, 3)
          - z1_2 * F(4, 4) - F(4, 5);
      C(6) = - z2 * F(5, 0) - z4 * F(5, 1) + z5_2 * F(5, 2) + z5 * F(5, 3)
          - z1_2 * F(5, 4) - F(5, 5);

      GENERIC_CALCULATE_I_5(4)
      T(6, 4) += z2 * F(6, 0) + z4 * F (6, 1) - z5_2 * F(6, 2) - z5 * F(6, 3)
          + z1_2 * F(6, 4) + F(6, 5);
    }

    break;
  case 5:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      const float z5 = 5.0f;
      const float z10 = 10.0f;
      const float z5_8 = 5.0f / 8.0f;
      const float z25_2 = 25.0f / 2.0f;
      const float z25_4 = 25.0f / 4.0f;
      const float z25_8 = 25.0f / 8.0f;

      C(1) = z2 * F(0, 0) - z4 * F(0, 1) - z5_2 * F(0, 2) + z5 * F(0, 3)
          + z1_2 * F(0, 4) - F(0, 5);
      C(2) = z2 * F(1, 0) - z4 * F(1, 1) - z5_2 * F(1, 2) + z5 * F(1, 3)
          + z1_2 * F(1, 4) - F(1, 5);
      C(3) = z5 * F(2, 0) - z10 * F(2, 1) - z25_4 * F(2, 2) + z25_2 * F(2, 3)
          + z5_4 * F(2, 4) - z5_2 * F(2, 5);
      C(4) = z5_2 * F(3, 0) - z5 * F(3, 1) - z25_8 * F(3, 2) + z25_4 * F(3, 3)
          + z5_8 * F(3, 4) - z5_4 * F(3, 5);
      C(5) = z2 * F(4, 0) - z4 * F(4, 1) - z5_2 * F(4, 2) + z5 * F(4, 3)
          + z1_2 * F(4, 4) - F(4, 5);
      C(6) = z2 * F(5, 0) - z4 * F(5, 1) - z5_2 * F(5, 2) + z5 * F(5, 3)
          + z1_2 * F(5, 4) - F(5, 5);

      GENERIC_CALCULATE_I_5(5)
      T(6, 5) += - z2 * F(6, 0) + z4 * F (6, 1) + z5_2 * F(6, 2) - z5 * F(6, 3)
          - z1_2 * F(6, 4) + F(6, 5);
    }

    break;
  case 6:
#pragma omp simd
    for (int _V = 0; _V < V; ++_V) {
      const float z105_8 = 105.0f / 8.0f;
      const float z105_16 = 105.0f / 16.0f;

      C(1) = F(0, 0) + z21_4 * (F(0, 4) - F(0, 2)) - F(0, 6);
      C(2) = F(1, 0) + z21_4 * (F(1, 4) - F(1, 2)) - F(1, 6);
      C(3) = z5_2 * (F(2, 0) - F(2, 6)) + z105_8 * (F(2, 4) - F(2, 2));
      C(4) = z5_4 * (F(3, 0) - F(3, 6)) + z105_16 * (F(3, 4) - F(3, 2));
      C(5) = F(4, 0) + z21_4 * (F(4, 4) - F(4, 2)) - F(4, 6);
      C(6) = F(5, 0) + z21_4 * (F(5, 4) - F(5, 2)) - F(5, 6);

      GENERIC_CALCULATE_I_5(6)
      T(6, 6) += - F(6, 0) + z21_4 * (F(6, 2) - F(6, 4)) + F(6, 6);
    }

    break;
  }
}
} // namespace euler
