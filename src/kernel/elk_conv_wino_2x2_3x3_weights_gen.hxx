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

// float atweights[A][A][V][V] <- float aweights[K][K][V][V])
__TRANS_WEIGHTS(float, 4, 3, 16, ISA_GENERIC)
{
  const float r4 = 1.0f / 4.0f;
  const float r2 = 1.0f / 2.0f;

  float C10[16], C11[16], C12[16], C20[16], C21[16], C22[16];
#undef F
#undef T
#undef C
#define F(h, w) aweights[h][w][_IV][_OV]
#define T(h, w) atweights[w][h][_IV][_OV]
#define C(c, n) C##c##n[_OV]
  for (int _IV = 0; _IV < 16; ++_IV) {
#pragma omp simd
    for (int _OV = 0; _OV < 16; ++_OV) {
      T(0,0) = F(0,0);
      T(1,0) = r2 * (F(0,0) - F(1,0) + F(2,0));
      T(2,0) = r2 * (F(0,0) + F(1,0) + F(2,0));
      T(3,0) = F(2,0);

      C(1,0) = r4 * (F(0,0) - F(0,1) + F(0,2));
      C(1,1) = r4 * (F(1,0) - F(1,1) + F(1,2));
      C(1,2) = r4 * (F(2,0) - F(2,1) + F(2,2));
      T(0,1) = 2 * C(1,0);
      T(1,1) = C(1,0) - C(1,1) + C(1,2);
      T(2,1) = C(1,0) + C(1,1) + C(1,2);
      T(3,1) = 2 * C(1,2);

      C(2,0) = r4 * (F(0,0) + F(0,1) + F(0,2));
      C(2,1) = r4 * (F(1,0) + F(1,1) + F(1,2));
      C(2,2) = r4 * (F(2,0) + F(2,1) + F(2,2));
      T(0,2) = 2 * C(2,0);
      T(1,2) = C(2,0) - C(2,1) + C(2,2);
      T(2,2) = C(2,0) + C(2,1) + C(2,2);
      T(3,2) = 2 * C(2,2);

      T(0,3) = F(0,2);
      T(1,3) = r2 * (F(0,2) - F(1,2) + F(2,2));
      T(2,3) = r2 * (F(0,2) + F(1,2) + F(2,2));
      T(3,3) = F(2,2);
    }
  }
}

TRANS_WEIGHTS(float, 4, 3, 16, ISA_GENERIC);

} // namespace euler
