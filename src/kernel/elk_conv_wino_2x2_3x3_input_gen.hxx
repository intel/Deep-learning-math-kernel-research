#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

// template <const bool is_border_>
// Params:
//    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//    int _hT_start, int _hT_end, int _wT_start, int _wT_end
 template <>
class convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3> {
  template<typename Type, int ...configs>
    friend class convolution_winograd_kernel_base;
protected:
  constexpr static int I = ISA_GENERIC;
  constexpr static int V = 16;
  constexpr static int A = 4;
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
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3>::__trans_input(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {

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

  float C1[16], C2[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,2) - F(1,0);
    C(2) = F(2,0) - F(2,2);
    T(0,0) = F(0,0) - F(0,2) - C(2);
    T(1,0) = C(1) + C(2);
    T(2,0) = C(2) - C(1);
    T(3,0) = C(1) + F(3,0) - F(3,2);

    C(1) = F(1,2) - F(1,1);
    C(2) = F(2,2) - F(2,1);
    T(0,1) = F(0,2) - F(0,1) - C(2);
    T(1,1) = C(2) - C(1);
    T(2,1) = C(2) + C(1);
    T(3,1) = F(3,2) - F(3,1) - C(1);

    C(1) = F(1,1) + F(1,2);
    C(2) = F(2,1) + F(2,2);
    T(0,2) = F(0,1) + F(0,2) - C(2);
    T(1,2) = C(2) - C(1);
    T(2,2) = C(2) + C(1);
    T(3,2) = F(3,1) + F(3,2) - C(1);

    C(1) = F(1,1) - F(1,3);
    C(2) = F(2,3) - F(2,1);
    T(0,3) = F(0,3) - F(0,1) - C(2);
    T(1,3) = C(1) + C(2);
    T(2,3) = C(2) - C(1);
    T(3,3) = C(1) - F(3,1) + F(3,3);
  }
}

// template <const bool is_border_>
// Params:
//   elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//   int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)
template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3>::
__trans_inputa(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {

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

  float C1[16], C2[16];
  switch(wA){
  case 0:
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,2) - F(1,0);
    C(2) = F(2,0) - F(2,2);
    T(0,0) = F(0,0) - F(0,2) - C(2);
    T(1,0) = C(1) + C(2);
    T(2,0) = C(2) - C(1);
    T(3,0) = C(1) + F(3,0) - F(3,2);
  }

  break;

  case 1:
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,2) - F(1,1);
    C(2) = F(2,2) - F(2,1);
    T(0,1) = F(0,2) - F(0,1) - C(2);
    T(1,1) = C(2) - C(1);
    T(2,1) = C(2) + C(1);
    T(3,1) = F(3,2) - F(3,1) - C(1);
  }

  break;

  case 2:
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,1) + F(1,2);
    C(2) = F(2,1) + F(2,2);
    T(0,2) = F(0,1) + F(0,2) - C(2);
    T(1,2) = C(2) - C(1);
    T(2,2) = C(2) + C(1);
    T(3,2) = F(3,1) + F(3,2) - C(1);
  }

  break;

   case 3:
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,1) - F(1,3);
    C(2) = F(2,3) - F(2,1);
    T(0,3) = F(0,3) - F(0,1) - C(2);
    T(1,3) = C(1) + C(2);
    T(2,3) = C(2) - C(1);
    T(3,3) = C(1) - F(3,1) + F(3,3);
  }

  break;
 }
}

}
