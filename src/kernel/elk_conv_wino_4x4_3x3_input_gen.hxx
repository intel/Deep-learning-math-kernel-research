#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <>
class convolution_winograd_kernel_base<conv::FP32, float, ISA_GENERIC, 16, 6, 3> {
protected:
  constexpr static int I = ISA_GENERIC;
  constexpr static int V = 16;
  constexpr static int A = 6;
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
    conv::FP32, float, ISA_GENERIC, 16, 6, 3>::__trans_input(
    elx_conv_t<conv::FP32> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z5 = 5.0f;
  const float z8 = 8.0f;
  const float z16 = 16.0f;
  const float z20 = 20.0f;

  auto f_cb = [&](int _h, int _w, int _V) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return md3(ainput, _h, _w, _V);
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
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

  float C1[16], C2[16], C3[16], C4[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = (z4*F(1,0) - z5*F(1,2) + F(1,4));
    C(2) = (z4*F(2,0) - z5*F(2,2) + F(2,4));
    C(3) = (z4*F(3,0) - z5*F(3,2) + F(3,4));
    C(4) = (z4*F(4,0) - z5*F(4,2) + F(4,4));

    T(0, 0) =   z16*F(0,0) - z20*F(0,2) + z4*F(0,4) - z5*C(2) + C(4);
    T(1, 0) = - z4*C(1) - z4*C(2) + C(3) + C(4);
    T(2, 0) =   z4*C(1) - z4*C(2) - C(3) + C(4);
    T(3, 0) = - z2*C(1) - C(2) + z2*C(3) + C(4);
    T(4, 0) =   z2*C(1) - C(2) - z2*C(3) + C(4);
    T(5, 0) =   z4*C(1) - z5*C(3) + z4*F(5,0) - z5*F(5,2) + F(5,4);

    C(1) = (z4*F(1,1) + z4*F(1,2) - F(1,3) - F(1,4));
    C(2) = (z4*F(2,1) + z4*F(2,2) - F(2,3) - F(2,4));
    C(3) = (z4*F(3,1) + z4*F(3,2) - F(3,3) - F(3,4));
    C(4) = (z4*F(4,1) + z4*F(4,2) - F(4,3) - F(4,4));

    T(0, 1) =   z4*F(0,3) - z16*F(0,2) - z16*F(0,1) + z4*F(0,4) + z5*C(2) - C(4);
    T(1, 1) =   z4*C(1) + z4*C(2) - C(3) - C(4);
    T(2, 1) = - z4*C(1) + z4*C(2) + C(3) - C(4);
    T(3, 1) =   z2*C(1) + C(2) - z2*C(3) - C(4);
    T(4, 1) = - z2*C(1) + C(2) + z2*C(3) - C(4);
    T(5, 1) = - z4*C(1) + z5*C(3) - z4*F(5,1) - z4*F(5,2) + F(5,3) + F(5,4);

    C(1) = (z4*F(1,1) - z4*F(1,2) - F(1,3) + F(1,4));
    C(2) = (z4*F(2,1) - z4*F(2,2) - F(2,3) + F(2,4));
    C(3) = (z4*F(3,1) - z4*F(3,2) - F(3,3) + F(3,4));
    C(4) = (z4*F(4,1) - z4*F(4,2) - F(4,3) + F(4,4));

    T(0, 2) =   z16*F(0,1) - z16*F(0,2) - z4*F(0,3) + z4*F(0,4) - z5*C(2) + C(4);
    T(1, 2) = - z4*C(1) - z4*C(2) + C(3) + C(4);
    T(2, 2) =   z4*C(1) - z4*C(2) - C(3) + C(4);
    T(3, 2) = - z2*C(1) - C(2) + z2*C(3) + C(4);
    T(4, 2) =   z2*C(1) - C(2) - z2*C(3) + C(4);
    T(5, 2) =   z4*C(1) - z5*C(3) + z4*F(5,1) - z4*F(5,2) - F(5,3) + F(5,4);

    C(1) = (z2*F(1,1) + F(1,2) - z2*F(1,3) - F(1,4));
    C(2) = (z2*F(2,1) + F(2,2) - z2*F(2,3) - F(2,4));
    C(3) = (z2*F(3,1) + F(3,2) - z2*F(3,3) - F(3,4));
    C(4) = (z2*F(4,1) + F(4,2) - z2*F(4,3) - F(4,4));

    T(0, 3) =   z8*F(0,3) - z4*F(0,2) - z8*F(0,1) + z4*F(0,4) + z5*C(2) - C(4);
    T(1, 3) =   z4*C(1) + z4*C(2) - C(3) - C(4);
    T(2, 3) = - z4*C(1) + z4*C(2) + C(3) - C(4);
    T(3, 3) =   z2*C(1) + C(2) - z2*C(3) - C(4);
    T(4, 3) = - z2*C(1) + C(2) + z2*C(3) - C(4);
    T(5, 3) = - z4*C(1) + z5*C(3) - z2*F(5,1) - F(5,2) + z2*F(5,3) + F(5,4);

    C(1) = (z2*F(1,1) - F(1,2) - z2*F(1,3) + F(1,4));
    C(2) = (z2*F(2,1) - F(2,2) - z2*F(2,3) + F(2,4));
    C(3) = (z2*F(3,1) - F(3,2) - z2*F(3,3) + F(3,4));
    C(4) = (z2*F(4,1) - F(4,2) - z2*F(4,3) + F(4,4));

    T(0, 4) =   z8*F(0,1) - z4*F(0,2) - z8*F(0,3) + z4*F(0,4) - 5*C(2) + C(4);
    T(1, 4) = - z4*C(1) - z4*C(2) + C(3) + C(4);
    T(2, 4) =   z4*C(1) - z4*C(2) - C(3) + C(4);
    T(3, 4) = - z2*C(1) - C(2) + z2*C(3) + C(4);
    T(4, 4) =   z2*C(1) - C(2) - z2*C(3) + C(4);
    T(5, 4) =   z4*C(1) - z5*C(3) + z2*F(5,1) - F(5,2) - z2*F(5,3) + F(5,4);

    C(1) = (z4*F(1,1) - z5*F(1,3) + F(1,5));
    C(2) = (z4*F(2,1) - z5*F(2,3) + F(2,5));
    C(3) = (z4*F(3,1) - z5*F(3,3) + F(3,5));
    C(4) = (z4*F(4,1) - z5*F(4,3) + F(4,5));

    T(0, 5) =   z16*F(0,1) - z20*F(0,3) + z4*F(0,5) - z5*C(2) + C(4);
    T(1, 5) = - z4*C(1) - z4*C(2) + C(3) + C(4);
    T(2, 5) =   z4*C(1) - z4*C(2) - C(3) + C(4);
    T(3, 5) = - z2*C(1) - C(2) + z2*C(3) + C(4);
    T(4, 5) =   z2*C(1) - C(2) - z2*C(3) + C(4);
    T(5, 5) =   z4*C(1) - z5*C(3) + z4*F(5,1) - z5*F(5,3) + F(5,5);
  }
}

template <bool is_border>
inline void convolution_winograd_kernel_base<conv::FP32, float, ISA_GENERIC, 16, 6, 3>::
__trans_inputa(
    elx_conv_t<conv::FP32> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  // TODO
  el_error("Unimplemented");
}
}
