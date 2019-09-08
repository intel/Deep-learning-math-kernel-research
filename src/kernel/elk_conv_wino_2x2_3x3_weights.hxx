#pragma once

#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {


template <typename WeightsType, int V>
struct elk_conv_wino_trans_weights<float, WeightsType, ISA_SKX_AVX512,
    4, 3, V> {

  constexpr static int A = 4;
  constexpr static int K = 3;
  constexpr static int I = ISA_SKX_AVX512;

  static void execute(
      float atweights[A][A][V][V], WeightsType aweights[K][K][V][V])
  {

    ENABLE_AVX512F();

    // Constants
    __m<V> r4 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 4.0f));
    __m<V> r2 = _mm<V>::set_ps(IMM_BCAST16(1.0f / 2.0f));
    __m<V> z2 = _mm<V>::set_ps(IMM_BCAST16(2.0f));

    // Inputs
    __m<V> f00, f10, f20, f01, f11, f21, f02, f12, f22;
    // Cache
    __m<V> c10, c11, c12, c20, c21, c22;
    // Outputs
    __m<V> /* t00, */ t10, t20, /* t30, */ t01, t11, t21, t31, t02, t12, t22,
        t32, /* t03, */ t13, t23 /*, t33 */;

    for (int _V = 0; _V < 16; ++_V) {
      if (std::is_same<WeightsType, float>::value) {
        f00 = _mm<V>::load_ps(aweights[0][0][_V]);
        f01 = _mm<V>::load_ps(aweights[0][1][_V]);
        f02 = _mm<V>::load_ps(aweights[0][2][_V]);
        f10 = _mm<V>::load_ps(aweights[1][0][_V]);
        f11 = _mm<V>::load_ps(aweights[1][1][_V]);
        f12 = _mm<V>::load_ps(aweights[1][2][_V]);
        f20 = _mm<V>::load_ps(aweights[2][0][_V]);
        f21 = _mm<V>::load_ps(aweights[2][1][_V]);
        f22 = _mm<V>::load_ps(aweights[2][2][_V]);
      } else {
        auto f16 = _mm<V / 2>::load_si256((__m256i *)aweights[0][0][_V]);
        f00 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[0][1][_V]);
        f01 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[0][2][_V]);
        f02 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[1][0][_V]);
        f10 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[1][1][_V]);
        f11 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[1][2][_V]);
        f12 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[2][0][_V]);
        f20 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[2][1][_V]);
        f21 = _mm<V>::cvtph_ps(f16);
        f16 = _mm<V / 2>::load_si256((__m256i *)aweights[2][2][_V]);
        f22 = _mm<V>::cvtph_ps(f16);
      }

      _mm<V>::store_ps(atweights[0][0][_V], f00);
      t10 = (r2 * ((f00 - f10) + f20));
      _mm<V>::store_ps(atweights[1][0][_V], t10);
      t20 = (r2 * ((f00 + f10) + f20));
      _mm<V>::store_ps(atweights[2][0][_V], t20);
      _mm<V>::store_ps(atweights[3][0][_V], f20);

      c10 = (r4 * ((f00 - f01) + f02));
      c11 = (r4 * ((f10 - f11) + f12));
      c12 = (r4 * ((f20 - f21) + f22));
      t01 = (z2 * c10);
      _mm<V>::store_ps(atweights[0][1][_V], t01);
      t11 = c10 - c11 + c12;
      _mm<V>::store_ps(atweights[1][1][_V], t11);
      t21 = c10 + c11 + c12;
      _mm<V>::store_ps(atweights[2][1][_V], t21);
      t31 = z2 * c12;
      _mm<V>::store_ps(atweights[3][1][_V], t31);

      c20 = r4 * (f00 + f01 + f02);
      c21 = r4 * (f10 + f11 + f12);
      c22 = r4 * (f20 + f21 + f22);
      t02 = z2 * c20;
      _mm<V>::store_ps(atweights[0][2][_V], t02);
      t12 = c20 - c21 + c22;
      _mm<V>::store_ps(atweights[1][2][_V], t12);
      t22 = c20 + c21 + c22;
      _mm<V>::store_ps(atweights[2][2][_V], t22);
      t32 = z2 * c22;
      _mm<V>::store_ps(atweights[3][2][_V], t32);

      _mm<V>::store_ps(atweights[0][3][_V], f02);
      t13 = r2 * (f02 - f12 + f22);
      _mm<V>::store_ps(atweights[1][3][_V], t13);
      t23 = r2 * (f02 + f12 + f22);
      _mm<V>::store_ps(atweights[2][3][_V], t23);
      _mm<V>::store_ps(atweights[3][3][_V], f22);
    }
  }

}; // elk_conv_wino_trans_weights

} // namespace euler
