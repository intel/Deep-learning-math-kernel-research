#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_3x3_3x3_input_gen.hxx"

namespace euler {
inline void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 5, 3>::
__trans_weights(float atweights[A][A][V][V], float aweights[K][K][V][V]) {
  const float r12 = 1.0f / 12.0f;
  const float r6 = 1.0f / 6.0f;
  const float r3 = 1.0f / 3.0f;
  const float r4 = 1.0f / 4.0f;
  const float r2 = 1.0f / 2.0f;
  const float r2_3 = 2.0f / 3.0f;

  float C10[V], C11[V], C12[V], C20[V], C21[V], C22[V], C30[V], C31[V],
      C32[V];
#undef F
#undef T
#undef C
#define F(h, w) aweights[h][w][_IV][_OV]
#define T(h, w) atweights[w][h][_IV][_OV]
#define C(c, n) C##c##n[_OV]
  for (int _IV = 0; _IV < V; ++_IV) {
#pragma omp simd
    for (int _OV = 0; _OV < V; ++_OV) {
      T(0, 0) = r4 * F(0, 0);
      T(1, 0) = -r12 * (F(0, 0) - F(1, 0) + F(2, 0));
      T(2, 0) = -r4 * (F(0, 0) + F(1, 0) + F(2, 0));
      T(3, 0) = r12 * F(0, 0) + r6 * F(1, 0) + r3 * F(2, 0);
      T(4, 0) = r2 * F(2, 0);

      C(1, 0) = -r6 * (F(0, 0) - F(0, 1) + F(0, 2));
      C(1, 1) = -r6 * (F(1, 0) - F(1, 1) + F(1, 2));
      C(1, 2) = -r6 * (F(2, 0) - F(2, 1) + F(2, 2));

      T(0, 1) = r2 * C(1, 0);
      T(1, 1) = -r6 * (C(1, 0) - C(1, 1) + C(1, 2));
      T(2, 1) = -r2 * (C(1, 0) + C(1, 1) + C(1, 2));
      T(3, 1) = r6 * C(1, 0) + r3 * C(1, 1) + r2_3 * C(1, 2);
      T(4, 1) = C(1, 2);

      C(2, 0) = -r2 * (F(0, 0) + F(0, 1) + F(0, 2));
      C(2, 1) = -r2 * (F(1, 0) + F(1, 1) + F(1, 2));
      C(2, 2) = -r2 * (F(2, 0) + F(2, 1) + F(2, 2));

      T(0, 2) = r2 * C(2, 0);
      T(1, 2) = -r6 * (C(2, 0) - C(2, 1) + C(2, 2));
      T(2, 2) = -r2 * (C(2, 0) + C(2, 1) + C(2, 2));
      T(3, 2) = r6 * C(2, 0) + r3 * C(2, 1) + r2_3 * C(2, 2);
      T(4, 2) = C(2, 2);

      C(3, 0) = r6 * F(0, 0) + r3 * F(0, 1) + r2_3 * F(0, 2);
      C(3, 1) = r6 * F(1, 0) + r3 * F(1, 1) + r2_3 * F(1, 2);
      C(3, 2) = r6 * F(2, 0) + r3 * F(2, 1) + r2_3 * F(2, 2);

      T(0, 3) = r2 * C(3, 0);
      T(1, 3) = -r6 * (C(3, 0) - C(3, 1) + C(3, 2));
      T(2, 3) = -r2 * (C(3, 0) + C(3, 1) + C(3, 2));
      T(3, 3) = r6 * C(3, 0) + r3 * C(3, 1) + r2_3 * C(3, 2);
      T(4, 3) = C(3, 2);

      T(0, 4) = r2 * F(0, 2);
      T(1, 4) = -r6 * (F(0, 2) - F(1, 2) + F(2, 2));
      T(2, 4) = -r2 * (F(0, 2) + F(1, 2) + F(2, 2));
      T(3, 4) = r6 * F(0, 2) + r3 * F(1, 2) + r2_3 * F(2, 2);
      T(4, 4) = F(2, 2);
    }
  }
}
} // namespace euler
