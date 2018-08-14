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
__TRANS_INPUT(float, 6, 3, 16, ISA_GENERIC)
{
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z5 = 5.0f;
  const float z8 = 8.0f;
  const float z16 = 16.0f;
  const float z20 = 20.0f;

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

// template <const bool is_border_>
// Params:
//   elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//   int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)
__TRANS_INPUTA(float, 6, 3, 16, ISA_GENERIC)
{
  // TODO
  el_error("Unimplemented");
}


TRANS_INPUT(float, 6, 3, 16, ISA_GENERIC);
TRANS_INPUTA(float, 6, 3, 16, ISA_GENERIC);

}
