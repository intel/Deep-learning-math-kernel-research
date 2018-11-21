#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_4x4_3x3_input_gen.hxx"

namespace euler {

inline void convolution_winograd_kernel_base<float, float, float, float, float, ISA_GENERIC, 16, 6, 3>::
__trans_weights(float atweights[A][A][V][V], float aweights[K][K][V][V]) {
  const float z2 = 2.0f;
  const float z4 = 4.0f;
  const float z6 = 6.0f;

  const float z1_4 = 1.0f / 4.0f;
  const float z1_6 = 1.0f / 6.0f;
  const float z1_12 = 1.0f / 12.0f;
  const float z1_16 = 1.0f / 16.0f;
  const float z1_24 = 1.0f / 24.0f;
  const float z1_36 = 1.0f / 36.0f;
  const float z1_48 = 1.0f / 48.0f;
  const float z1_72 = 1.0f / 72.0f;
  const float z1_96 = 1.0f / 96.0f;
  const float z1_144 = 1.0f / 144.0f;
  const float z1_288 = 1.0f / 288.0f;
  const float z1_576 = 1.0f / 576.0f;

  float C10[16], C11[16], C12[16],
        C20[16], C21[16], C22[16],
        C30[16], C31[16], C32[16],
        C40[16], C41[16], C42[16];

#undef F
#undef T
#undef C

#define F(h, w) aweights[h][w][_IV][_OV]
#define T(h, w) atweights[w][h][_IV][_OV]
#define C(c, n) C##c##n[_OV]

  for (int _IV = 0; _IV < 16; ++_IV) {
#pragma omp simd
    for (int _OV = 0; _OV < 16; ++_OV) {
      T(0, 0) = z1_16 * F(0, 0);
      T(1, 0) = - z1_24 * (F(0, 0) + F(1, 0) + F(2, 0));
      T(2, 0) = - z1_24 * (F(0, 0) - F(1, 0) + F(2, 0));
      T(3, 0) = z1_96 * F(0, 0) + z1_48 * F(1, 0) + z1_24 * F(2, 0);
      T(4, 0) = z1_96 * F(0, 0) - z1_48 * F(1, 0) + z1_24 * F(2, 0);
      T(5, 0) = z1_4 * F(2, 0);

      C(1, 0) = z1_144 * (F(0, 0) + F(0, 1) + F(0, 2));
      C(1, 1) = z1_72 * (F(1, 0) + F(1, 1) + F(1, 2));
      C(1, 2) = z1_36 * (F(2, 0) + F(2, 1) + F(2, 2));

      T(0, 1) = - z6 * C(1, 0);
      T(1, 1) = z4 * C(1, 0) + z2 * C(1, 1) + C(1, 2);
      T(2, 1) = z4 * C(1, 0) - z2 * C(1, 1) + C(1, 2);
      T(3, 1) = - C(1, 0) - C(1, 1) - C(1, 2);
      T(4, 1) = - C(1, 0) + C(1, 1) - C(1, 2);
      T(5, 1) = - z6 * C(1, 2);

      C(2, 0) = z1_144 * (F(0, 0) - F(0, 1) + F(0, 2));
      C(2, 1) = z1_72 * (F(1, 0) - F(1, 1) + F(1, 2));
      C(2, 2) = z1_36 * (F(2, 0) - F(2, 1) + F(2, 2));

      T(0, 2) = - z6 * C(2, 0);
      T(1, 2) = z4 * C(2, 0) + z2 * C(2, 1) + C(2, 2);
      T(2, 2) = z4 * C(2, 0) - z2 * C(2, 1) + C(2, 2);
      T(3, 2) = - C(2, 0) - C(2, 1) - C(2, 2);
      T(4, 2) = - C(2, 0) + C(2, 1) - C(2, 2);
      T(5, 2) = - z6 * C(2, 2);

      C(3, 0) = (z1_576 * F(0, 0) + z1_288 * F(0, 1) + z1_144 * F(0, 2));
      C(3, 1) = (z1_288 * F(1, 0) + z1_144 * F(1, 1) + z1_72 * F(1, 2));
      C(3, 2) = (z1_144 * F(2, 0) + z1_72 * F(2, 1) + z1_36 * F(2, 2));

      T(0, 3) = z6 * C(3, 0);
      T(1, 3) = - z4 * C(3, 0) - z2 * C(3, 1) - C(3, 2);
      T(2, 3) = - z4 * C(3, 0) + z2 * C(3, 1) - C(3, 2);
      T(3, 3) = C(3, 0) + C(3, 1) + C(3, 2);
      T(4, 3) = C(3, 0) - C(3, 1) + C(3, 2);
      T(5, 3) = z6 * C(3, 2);

      C(4, 0) = (z1_576 * F(0, 0) - z1_288 * F(0, 1) + z1_144 * F(0, 2));
      C(4, 1) = (z1_288 * F(1, 0) - z1_144 * F(1, 1) + z1_72 * F(1, 2));
      C(4, 2) = (z1_144 * F(2, 0) - z1_72 * F(2, 1) + z1_36 * F(2, 2));

      T(0, 4) = z6 * C(4, 0);
      T(1, 4) = - z4 * C(4, 0) - z2 * C(4, 1) - C(4, 2);
      T(2, 4) = - z4 * C(4, 0) + z2 * C(4, 1) - C(4, 2);
      T(3, 4) = C(4, 0) + C(4, 1) + C(4, 2);
      T(4, 4) = C(4, 0) - C(4, 1) + C(4, 2);
      T(5, 4) = z6 * C(4, 2);

      T(0, 5) = z1_4 * F(0, 2);
      T(1, 5) = - z1_6 * (F(0, 2) + F(1, 2) + F(2, 2));
      T(2, 5) = - z1_6 * (F(0, 2) - F(1, 2) + F(2, 2));
      T(3, 5) = z1_24 * F(0, 2) + z1_12 * F(1, 2) + z1_6 * F(2, 2);
      T(4, 5) = z1_24 * F(0, 2) - z1_12 * F(1, 2) + z1_6 * F(2, 2);
      T(5, 5) = F(2, 2);
    }
  }
}
} // namespace euler
