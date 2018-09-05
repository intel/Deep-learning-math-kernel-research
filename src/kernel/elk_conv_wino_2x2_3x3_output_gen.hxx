#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_2x2_3x3_input_gen.hxx"

namespace euler {

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
 template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3>::
__trans_output(elx_conv_t<float> &xc, float *output,
      float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end) {
  float dummy[16];
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

  auto p_cb = [&](int _h, int _w, int _V) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput, _h, _w, _V);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, 16);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return &dummy[_V];
      else
        return &md3(aoutput, _h, _w, _V);
    }
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]
  float c0[16], c1[16], c2[16], c3[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(1,0) + T(1,1) + T(1,2);
    C(1) = T(2,0) + T(2,1) + T(2,2);
    C(2) = T(1,2) + T(1,3) - T(1,1);
    C(3) = T(2,2) + T(2,3) - T(2,1);

    P(0,0) = T(0,0) + T(0,1) + T(0,2) + C(0) + C(1);
    if (with_bias) P(0, 0) += B;
    if (with_relu) P(0, 0) = P(0, 0) > 0 ? P(0, 0) : 0;
    P(1,0) = C(1) - C(0) + T(3,0) + T(3,1) + T(3,2);
    if (with_bias) P(1, 0) += B;
    if (with_relu) P(1, 0) = P(1, 0) > 0 ? P(1, 0) : 0;
    P(0,1) = T(0,2) - T(0,1) + T(0,3) + C(2) + C(3);
    if (with_relu) P(0, 1) = P(0, 1) > 0 ? P(0, 1) : 0;
    if (with_bias) P(0, 1) += B;
    P(1,1) = C(3) - C(2) - T(3,1) + T(3,2) + T(3,3);
    if (with_bias) P(1, 1) += B;
    if (with_relu) P(1, 1) = P(1, 1) > 0 ? P(1, 1) : 0;
  }
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
 template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3>::
__trans_outputa_th(elx_conv_t<float> &xc, float *toutputa, float *toutput,
    int Tz, bool stream_out) {

  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) md4(atoutput, _h, 0, 0, _V)
#define P(_h) md2(atoutputa, _h, _V)

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    P(0) = T(0) + T(1) + T(2);
    P(1) = - T(1) + T(2) + T(3);
  }
}


#define GENERIC_CALCULATE_TILE_4(z, n, nil)            \
  P(n, 0) = T(n, 0) + T(n, 1) + T(n, 2);               \
  if (with_bias) P(n, 0) += B;                         \
  if (with_relu) P(n, 0) = P(n, 0) > 0 ? P(n, 0) : 0;  \
  P(n, 1) = T(n, 2) - T(n, 1) + T(n, 3);               \
  if (with_bias) P(n, 1) += B;                         \
  if (with_relu) P(n, 1) = P(n, 1) > 0 ? P(n, 1) : 0;

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
template <bool ...conditions>
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 4, 3>::
__trans_outputa_bh(elx_conv_t<float> &xc, float *output,
    float atoutput[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;

   float dummy[16];
   auto p_cb = [&](int _h, int _w, int _V) {
     if (wOA_end == -1) {
       MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
       return &md3(aoutput, _h, _w, _V);
     } else {
       MD3(float, aoutput, output, xc.oh, xc.ow, 16);
       if (is_border && (_h > hOA_end || _w > wOA_end))
         return &dummy[_V];
       else
         return &md3(aoutput, _h, _w , _V);
     }
   };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    BOOST_PP_REPEAT(2, GENERIC_CALCULATE_TILE_4, nil)
  }

}

} // namespace euler
