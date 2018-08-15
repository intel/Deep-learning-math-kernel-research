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
__TRANS_OUTPUT(float, 6, 3, 16, ISA_GENERIC)
{
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z8 = 8.0f;

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

  float c0[16], c1[16], c2[16], c3[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(1, 0) + T(1, 1) + T(1, 2) + T(1, 3) + T(1, 4);
    C(1) = T(2, 0) + T(2, 1) + T(2, 2) + T(2, 3) + T(2, 4);
    C(2) = T(3, 0) + T(3, 1) + T(3, 2) + T(3, 3) + T(3, 4);
    C(3) = T(4, 0) + T(4, 1) + T(4, 2) + T(4, 3) + T(4, 4);

    P(0, 0) = T(0, 0) + T(0, 1) + T(0, 2) + T(0, 3) + T(0, 4)
      + C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 0) += B;
    P(1, 0) = C(0) - C(1) + z2*C(2) - z2*C(3);
    if (with_bias_) P(1, 0) += B;
    P(2, 0) = C(0) + C(1) + z4*C(2) + z4*C(3);
    if (with_bias_) P(2, 0) += B;
    P(3, 0) = C(0) - C(1) + z8*C(2) - z8*C(3)
      + T(5, 0) + T(5, 1) + T(5, 2) + T(5, 3) + T(5, 4);
    if (with_bias_) P(3, 0) += B;

    C(0) = T(1, 1) - T(1, 2) + z2*T(1, 3) - z2*T(1, 4);
    C(1) = T(2, 1) - T(2, 2) + z2*T(2, 3) - z2*T(2, 4);
    C(2) = T(3, 1) - T(3, 2) + z2*T(3, 3) - z2*T(3, 4);
    C(3) = T(4, 1) - T(4, 2) + z2*T(4, 3) - z2*T(4, 4);

    P(0, 1) = T(0, 1) - T(0, 2) + z2*T(0, 3) - z2*T(0, 4)
      + C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 1) += B;
    P(1, 1) = C(0) - C(1) + z2*C(2) - z2*C(3);
    if (with_bias_) P(1, 1) += B;
    P(2, 1) = C(0) + C(1) + z4*C(2) + z4*C(3);
    if (with_bias_) P(2, 1) += B;
    P(3, 1) = C(0) - C(1) + z8*C(2) - z8*C(3)
      + T(5, 1) - T(5, 2) + z2*T(5, 3) - z2*T(5, 4);
    if (with_bias_) P(3, 1) += B;

    C(0) = T(1, 1) + T(1, 2) + z4*T(1, 3) + z4*T(1, 4);
    C(1) = T(2, 1) + T(2, 2) + z4*T(2, 3) + z4*T(2, 4);
    C(2) = T(3, 1) + T(3, 2) + z4*T(3, 3) + z4*T(3, 4);
    C(3) = T(4, 1) + T(4, 2) + z4*T(4, 3) + z4*T(4, 4);

    P(0, 2) = T(0, 1) + T(0, 2) + z4*T(0, 3) + z4*T(0, 4)
      + C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 2) += B;
    P(1, 2) = C(0) - C(1) + z2*C(2) - z2*C(3);
    if (with_bias_) P(1, 2) += B;
    P(2, 2) = C(0) + C(1) + z4*C(2) + z4*C(3);
    if (with_bias_) P(2, 2) += B;
    P(3, 2) = C(0) - C(1) + z8*C(2) - z8*C(3)
      + T(5, 1) + T(5, 2) + z4*T(5, 3) + z4*T(5, 4);
    if (with_bias_) P(3, 2) += B;

    C(0) = T(1, 1) - T(1, 2) + z8*T(1, 3) - z8*T(1, 4) + T(1, 5);
    C(1) = T(2, 1) - T(2, 2) + z8*T(2, 3) - z8*T(2, 4) + T(2, 5);
    C(2) = T(3, 1) - T(3, 2) + z8*T(3, 3) - z8*T(3, 4) + T(3, 5);
    C(3) = T(4, 1) - T(4, 2) + z8*T(4, 3) - z8*T(4, 4) + T(4, 5);

    P(0, 3) = T(0, 1) - T(0, 2) + z8*T(0, 3) - z8*T(0, 4) + T(0, 5)
      + C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 3) += B;
    P(1, 3) = C(0) - C(1) + z2*C(2) - z2*C(3);
    if (with_bias_) P(1, 3) += B;
    P(2, 3) = C(0) + C(1) + z4*C(2) + z4*C(3);
    if (with_bias_) P(2, 3) += B;
    P(3, 3) = C(0) - C(1) + z8*C(2) - z8*C(3)
      + T(5, 1) - T(5, 2) + z8*T(5, 3) - z8*T(5, 4) + T(5, 5);
    if (with_bias_) P(3, 3) += B;
  }
}

// Params:
//   elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz,
//   bool stream_out
__TRANS_OUTPUTA_TH( float, 6, 3, 16, ISA_GENERIC)
{
  // TODO
  el_error("Unimplemented");
}

// template <const bool is_border_, const bool with_bias_>
// Params:
//   elx_conv_t<float> &xc,
//   float *output, float atoutput[A][A - K + 1][V], float *bias,
//   int _hOA_end, int _wOA_end
__TRANS_OUTPUTA_BH(float, 6, 3, 16, ISA_GENERIC)
{
  // TODO
  el_error("Unimplemented");
}

TRANS_OUPUT(float, 6, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_TH(float, 6, 3, 16, ISA_GENERIC);
TRANS_OUTPUTA_BH(float, 6, 3, 16, ISA_GENERIC);

} // namespace euler
