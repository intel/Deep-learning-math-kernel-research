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

// template <const bool is_border_>
// Params:
//    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//    int _hT_start, int _hT_end, int _wT_start, int _wT_end
__TRANS_INPUT(float, 4, 3, 16, ISA_GENERIC)
{
  auto f_cb = [&](int _h, int _w, int _V) {
    if (_wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return md3(ainput, _h, _w, _V);
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border_
          && (_h < _hT_start || _w < _wT_start || _h > _hT_end
                 || _w > _wT_end))
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
__TRANS_INPUTA(float, 4, 3, 16, ISA_GENERIC)
{
  // TODO
  el_error("Unimplemented");
}

TRANS_INPUT(float, 4, 3, 16, ISA_GENERIC);
TRANS_INPUTA(float, 4, 3, 16, ISA_GENERIC);

}
