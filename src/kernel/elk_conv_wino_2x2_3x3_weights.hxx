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

// float atweights[A][A][V][V] <- float aweights[K][K][V][V])
__TRANS_WEIGHTS(float, 4, 3, 16, ISA_SKX_AVX512)
{
  ENABLE_AVX512F();

  // Constants
  __m512 r4 = _mm512_set_ps(IMM_BCAST16(1.0f / 4.0f));
  __m512 r2 = _mm512_set_ps(IMM_BCAST16(1.0f / 2.0f));
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));

  // Inputs
  __m512 f00, f10, f20, f01, f11, f21, f02, f12, f22;
  // Cache
  __m512 c10, c11, c12, c20, c21, c22;
  // Outputs
  __m512 /* t00, */t10, t20, /* t30, */t01, t11, t21, t31, t02, t12, t22,
      t32, /* t03, */t13, t23/*, t33 */;
#undef F
#undef T
#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]
  for (int _V = 0; _V < 16; ++_V) {
    f00 = _mm512_load_ps(F(0, 0));
    f01 = _mm512_load_ps(F(0, 1));
    f02 = _mm512_load_ps(F(0, 2));
    f10 = _mm512_load_ps(F(1, 0));
    f11 = _mm512_load_ps(F(1, 1));
    f12 = _mm512_load_ps(F(1, 2));
    f20 = _mm512_load_ps(F(2, 0));
    f21 = _mm512_load_ps(F(2, 1));
    f22 = _mm512_load_ps(F(2, 2));

    _mm512_store_ps(T(0, 0), f00);
    t10 = MUL(r2, ADD(SUB(f00, f10), f20));
    _mm512_store_ps(T(1, 0), t10);
    t20 = MUL(r2, ADD(ADD(f00, f10), f20));
    _mm512_store_ps(T(2, 0), t20);
    _mm512_store_ps(T(3, 0), f20);

    c10 = MUL(r4, ADD(SUB(f00, f01), f02));
    c11 = MUL(r4, ADD(SUB(f10, f11), f12));
    c12 = MUL(r4, ADD(SUB(f20, f21), f22));
    t01 = MUL(z2, c10);
    _mm512_store_ps(T(0, 1), t01);
    t11 = ADD(SUB(c10, c11), c12);
    _mm512_store_ps(T(1, 1), t11);
    t21 = ADD(ADD(c10, c11), c12);
    _mm512_store_ps(T(2, 1), t21);
    t31 = MUL(z2, c12);
    _mm512_store_ps(T(3, 1), t31);

    c20 = MUL(r4, ADD(ADD(f00, f01), f02));
    c21 = MUL(r4, ADD(ADD(f10, f11), f12));
    c22 = MUL(r4, ADD(ADD(f20, f21), f22));
    t02 = MUL(z2, c20);
    _mm512_store_ps(T(0, 2), t02);
    t12 = ADD(SUB(c20, c21), c22);
    _mm512_store_ps(T(1, 2), t12);
    t22 = ADD(ADD(c20, c21), c22);
    _mm512_store_ps(T(2, 2), t22);
    t32 = MUL(z2, c22);
    _mm512_store_ps(T(3, 2), t32);

    _mm512_store_ps(T(0, 3), f02);
    t13 = MUL(r2, ADD(SUB(f02, f12), f22));
    _mm512_store_ps(T(1, 3), t13);
    t23 = MUL(r2, ADD(ADD(f02, f12), f22));
    _mm512_store_ps(T(2, 3), t23);
    _mm512_store_ps(T(3, 3), f22);
  }
}

TRANS_WEIGHTS(float, 4, 3, 16, ISA_GENERIC);
TRANS_WEIGHTS(float, 4, 3, 16, ISA_SKX_AVX512);

} // namespace euler
