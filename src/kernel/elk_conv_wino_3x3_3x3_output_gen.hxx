#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

#ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
#error "Don't include this file directly"
#endif
namespace euler {


// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUT(float, 5, 3, 16, ISA_GENERIC)
{
  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    if (_wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput, _h, _w, _V);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, 16);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
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
  float c0[16], c1[16], c2[16], c3[16], c4[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(0, 0) + T(0, 1) + T(0, 2) + T(0, 3);
    C(1) = T(1, 0) + T(1, 1) + T(1, 2) + T(1, 3);
    C(2) = T(2, 0) + T(2, 1) + T(2, 2) + T(2, 3);
    C(3) = T(3, 0) + T(3, 1) + T(3, 2) + T(3, 3);
    C(4) = T(4, 0) + T(4, 1) + T(4, 2) + T(4, 3);
    P(0, 0) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 0) += B;
    if (with_relu_) P(0, 0) = P(0, 0) > 0 ? P(0, 0) : 0;
    P(1, 0) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 0) += B;
    if (with_relu_) P(1, 0) = P(1, 0) > 0 ? P(1, 0) : 0;
    P(2, 0) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 0) += B;
    if (with_relu_) P(2, 0) = P(2, 0) > 0 ? P(2, 0) : 0;

    C(0) = T(0, 2) - T(0, 1) + 2 * T(0, 3);
    C(1) = T(1, 2) - T(1, 1) + 2 * T(1, 3);
    C(2) = T(2, 2) - T(2, 1) + 2 * T(2, 3);
    C(3) = T(3, 2) - T(3, 1) + 2 * T(3, 3);
    C(4) = T(4, 2) - T(4, 1) + 2 * T(4, 3);
    P(0, 1) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 1) += B;
    if (with_relu_) P(0, 1) = P(0, 1) > 0 ? P(0, 1) : 0;
    P(1, 1) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 1) += B;
    if (with_relu_) P(1, 1) = P(1, 1) > 0 ? P(1, 1) : 0;
    P(2, 1) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 1) += B;
    if (with_relu_) P(2, 1) = P(2, 1) > 0 ? P(2, 1) : 0;

    C(0) = T(0, 1) + T(0, 2) + 4 * T(0, 3) + T(0, 4);
    C(1) = T(1, 1) + T(1, 2) + 4 * T(1, 3) + T(1, 4);
    C(2) = T(2, 1) + T(2, 2) + 4 * T(2, 3) + T(2, 4);
    C(3) = T(3, 1) + T(3, 2) + 4 * T(3, 3) + T(3, 4);
    C(4) = T(4, 1) + T(4, 2) + 4 * T(4, 3) + T(4, 4);
    P(0, 2) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 2) += B;
    if (with_relu_) P(0, 2) = P(0, 2) > 0 ? P(0, 2) : 0;
    P(1, 2) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 2) += B;
    if (with_relu_) P(1, 2) = P(1, 2) > 0 ? P(1, 2) : 0;
    P(2, 2) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 2) += B;
    if (with_relu_) P(2, 2) = P(2, 2) > 0 ? P(2, 2) : 0;
  }
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 5, 3, 16, ISA_GENERIC)
{
  MD4(float, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(float, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) md4(atoutput, _h, 0, 0, _V)
#define P(_h) md2(atoutputa, _h, _V)

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    P(0) = T(0) + T(1) + T(2) + T(3);
    P(1) = - T(1) + T(2) + 2.0f * T(3);
    P(2) = T(1) + T(2) + 4.0f * T(3) + T(4);
  }
}

#define GENERIC_CALCULATE_TILE_5(z, n, nil)                    \
  P(n, 0) = T(n, 0) + T(n, 1) + T(n, 2) + T(n, 3);             \
  if (with_bias_) P(n, 0) += B;                                \
  if (with_relu_) P(n, 0) = P(n, 0) > 0 ? P(n, 0) : 0;         \
  P(n, 1) = T(n, 2) - T(n, 1) + z2 * T(n, 3);                  \
  if (with_bias_) P(n, 1) += B;                                \
  if (with_relu_) P(n, 1) = P(n, 1) > 0 ? P(n, 1) : 0;         \
  P(n, 2) = T(n, 1) + T(n, 2) + z4 * T(n, 3) + T(n, 4);        \
  if (with_bias_) P(n, 2) += B;                                \
  if (with_relu_) P(n, 2) = P(n, 2) > 0 ? P(n, 2) : 0;

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_GENERIC)
{
  const float z2 = 2.0f;
  const float z4 = 4.0f;

  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    if (_wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput, _h, _w, _V);
    } else {
      MD3(float, aoutput, output, xc.oh, xc.ow, 16);
      if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
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
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]

#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    BOOST_PP_REPEAT(3, GENERIC_CALCULATE_TILE_5, nil)
  }
}


TRANS_OUPUT(float, 5, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_TH(float, 5, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_BH(float, 5, 3, 16, ISA_GENERIC);

} // namespace euler
