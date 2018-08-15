#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_5x5_3x3_input_gen.hxx"

namespace euler {

#define GENERIC_CALCULATE_W_5_0(z, n, nil)                                        \
  C(n) = r4_81 * (F(n, 0) + F(n, 1) + F(n, 2));
#define GENERIC_CALCULATE_W_5_1(z, n, nil)                                        \
  C(n) = r4_81 * (F(n, 0) - F(n, 1) + F(n, 2));
#define GENERIC_CALCULATE_W_5_2(z, n, nil)                                        \
  C(n) = r2_405 * F(n, 0) + r4_405 * F(n, 1) + r8_405 * F(n, 2);
#define GENERIC_CALCULATE_W_5_3(z, n, nil)                                        \
  C(n) = - r2_405 * F(n, 0) + r4_405 * F(n, 1) - r8_405 * F(n, 2);
#define GENERIC_CALCULATE_W_5_4(z, n, nil)                                        \
  C(n) = r32_405 * F(n, 0) + r16_405 * F(n, 1) + r8_405 * F(n, 2);
#define GENERIC_CALCULATE_W_5_5(z, n, nil)                                        \
  C(n) = - r32_405 * F(n, 0) + r16_405 * F(n, 1) - r8_405 * F(n, 2);

#define GENERIC_CALCULATE_W_5(n)                                             \
  T(0, n) = C(0) + C(1) + C(2);                                              \
  T(1, n) = C(0) - C(1) + C(2);                                              \
  T(2, n) = r1_10 * C(0) + r1_5 * C(1) + r2_5 * C(2);                        \
  T(3, n) = - r1_10 * C(0) + r1_5 * C(1) - r2_5 * C(2);                      \
  T(4, n) = r8_5 * C(0) + r4_5 * C(1) + r2_5 * C(2);                         \
  T(5, n) = - r8_5 * C(0) + r4_5 * C(1) - r2_5 * C(2);                       \
  T(6, n) = r9_2 * C(2);

// float atweights[A][A][V][V] <- float aweights[K][K][V][V])
void convolution_winograd_kernel_base<float, ISA_GENERIC, 16, 7, 3>::
__trans_weights(float atweights[A][A][V][V], float aweights[K][K][V][V]) {
  const float r1_5 = 1.0f / 5.0f;
  const float r1_10 = 1.0f / 10.0f;
  const float r2_5 = 2.0f / 5.0f;
  const float r4_5 = 4.0f / 5.0f;
  const float r4_81 = 4.0f / 81.0f;
  const float r8_5 = 8.0f / 5.0f;
  const float r9_2 = 9.0f / 2.0f;

  float C0[V], C1[V], C2[V];
#undef F
#undef T
#undef C
#define F(h, w) aweights[h][w][_IV][_OV]
#define T(h, w) atweights[w][h][_IV][_OV]
#define C(n) C##n[_OV]
  for (int _IV = 0; _IV < V; ++_IV) {
#pragma omp simd
    for (int _OV = 0; _OV < V; ++_OV) {
      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_0, nil)
      GENERIC_CALCULATE_W_5(0)


      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_1, nil)
      GENERIC_CALCULATE_W_5(1)


      const float r2_405 = 2.0f / 405.0f;
      const float r4_405 = 4.0f / 405.0f;
      const float r8_405 = 8.0f / 405.0f;

      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_2, nil)
      GENERIC_CALCULATE_W_5(2)


      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_3, nil)
      GENERIC_CALCULATE_W_5(3)


      const float r16_405 = 16.0f / 405.0f;
      const float r32_405 = 32.0f / 405.0f;

      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_4, nil)
      GENERIC_CALCULATE_W_5(4)


      BOOST_PP_REPEAT(3, GENERIC_CALCULATE_W_5_5, nil)
      GENERIC_CALCULATE_W_5(5)


      const float r2_9 = 2.0f / 9.0f;
      const float r1_45 = 1.0f / 45.0f;
      const float r2_45 = 2.0f / 45.0f;
      const float r4_45 = 4.0f / 45.0f;
      const float r8_45 = 8.0f / 45.0f;
      const float r16_45 = 16.0f / 45.0f;

      C(0) = r2_9 * (F(0, 2) + F(2, 2));
      C(1) = r1_45 * F(0, 2) + r4_45 * F(2, 2);
      C(2) = r16_45 * F(0, 2) + r4_45 * F(2, 2);
      T(0, 6) = C(0) + r2_9 * F(1, 2);
      T(1, 6) = C(0) - r2_9 * F(1, 2);
      T(2, 6) = r2_45 * F(1, 2) + C(1);
      T(3, 6) = r2_45 * F(1, 2) - C(1);
      T(4, 6) = r8_45 * F(1, 2) + C(2);
      T(5, 6) = r8_45 * F(1, 2) - C(2);
      T(6, 6) = F(2, 2);
    }
  }
}
} // namespace euler
