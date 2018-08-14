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

#define GENERIC_CALCULATE_I(n)                                                  \
  T(0, n) = C(1) + z2 * C(2) - C(3);                                            \
  T(1, n) = z3 * C(2) - z2 * C(1) - C(3);                                       \
  T(2, n) = z2 * C(1) + C(2) - C(3);                                            \
  T(3, n) = C(1) - C(3);                                                        \
  T(4, n) = - z2 * C(1) + C(2) + z2 * C(3);

// template <const bool is_border_>
// Params:
//    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//    int _hT_start, int _hT_end, int _wT_start, int _wT_end
__TRANS_INPUT(float, 5, 3, 16, ISA_GENERIC)
{
  const float z2 = 2.0f;
  const float z3 = 3.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

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

  float C1[16], C2[16], C3[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
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

// template <const bool is_border_>
// Params:
//   elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//   int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)
__TRANS_INPUTA(float, 5, 3, 16, ISA_GENERIC)
{
  const float z2 = 2.0f;
  const float z3 = 3.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

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
#define T(_h, _w) atinput[_h][_w][_V]

  float C1[16], C2[16], C3[16];
  switch (_wA) {
  case 0:
#pragma omp simd
    for (int _V = 0; _V < 16; ++_V) {
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
    for (int _V = 0; _V < 16; ++_V) {
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
    for (int _V = 0; _V < 16; ++_V) {
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
    for (int _V = 0; _V < 16; ++_V) {
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
    for (int _V = 0; _V < 16; ++_V) {
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

TRANS_INPUT(float, 5, 3, 16, ISA_GENERIC);
TRANS_INPUTA(float, 5, 3, 16, ISA_GENERIC);

} // namespace euler
