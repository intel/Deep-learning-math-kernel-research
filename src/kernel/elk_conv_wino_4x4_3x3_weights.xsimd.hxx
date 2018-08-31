#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_4x4_3x3_input.xsimd.hxx"

namespace euler {

template <int V>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, V, 6, 3>::
  __trans_weights(float atweights[A][A][V][V], float aweights[K][K][V][V]) {
  auto z0 = _mm512_setzero_ps();
  auto z1 = _mm512_set_ps(IMM_BCAST16(1.0f));
  auto z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  auto z4 = _mm512_set_ps(IMM_BCAST16(4.0f));
  auto z6 = _mm512_set_ps(IMM_BCAST16(6.0f));

  auto z1_4 = _mm512_set_ps(IMM_BCAST16(1.0f / 4.0f));
  auto z1_6 = _mm512_set_ps(IMM_BCAST16(1.0f / 6.0f));
  auto z1_12 = _mm512_set_ps(IMM_BCAST16(1.0f / 12.0f));
  auto z1_16 = _mm512_set_ps(IMM_BCAST16(1.0f / 16.0f));
  auto z1_24 = _mm512_set_ps(IMM_BCAST16(1.0f / 24.0f));
  auto z1_36 = _mm512_set_ps(IMM_BCAST16(1.0f / 36.0f));
  auto z1_48 = _mm512_set_ps(IMM_BCAST16(1.0f / 48.0f));
  auto z1_72 = _mm512_set_ps(IMM_BCAST16(1.0f / 72.0f));
  auto z1_96 = _mm512_set_ps(IMM_BCAST16(1.0f / 96.0f));
  auto z1_144 = _mm512_set_ps(IMM_BCAST16(1.0f / 144.0f));
  auto z1_288 = _mm512_set_ps(IMM_BCAST16(1.0f / 288.0f));
  auto z1_576 = _mm512_set_ps(IMM_BCAST16(1.0f / 576.0f));

  //Inputs
  __m<V> f00, f01, f02,
         f10, f11, f12,
         f20, f21, f22;
  // Cache
  __m<V> c0, c1, c2;
  // Buffer
  __m<V> a00, a01;
  __m<V> b00, b01, b02, b03, b04, b05;
  // Outputs
  __m<V> t00, t01, t02, t03, t04, t05,
         t10, t11, t12, t13, t14, t15,
         t20, t21, t22, t23, t24, t25,
         t30, t31, t32, t33, t34, t35,
         t40, t41, t42, t43, t44, t45,
         t50, t51, t52, t53, t54, t55;

#undef F
#undef T

#undef F
#undef T
#undef f
#undef OP
#undef ISTORE

#define F(h, w) aweights[h][w][_V]
#define T(h, w) atweights[w][h][_V]
#define f(m, n) f##m##n
#define OP(m,n) f(m,n) = _mm512_load_ps(F(m, n))
#define ISTORE(i, j) _mm512_store_ps(T(i, j), t##i##j)

  for (int _V = 0; _V < V; ++_V) {
    VECTOR_DEF(M3, (0));

    t00 = MUL(z1_16, f00);
    ISTORE(0, 0);
    a00 = MUL(z1_24, ADD(f00, f20));
    t10 = FNMSUB(z1_24, f10, a00);
    ISTORE(1, 0);
    t20 = FMSUB(z1_24, f10, a00);
    ISTORE(2, 0);
    a01 = MUL(z1_96, FMADD(z4, f20, f00));
    t30 = FMADD(z1_48, f10, a01);
    ISTORE(3, 0);
    t40 = FNMADD(z1_48, f10, a01);
    ISTORE(4, 0);
    t50 = MUL(z1_4, f20);
    ISTORE(5, 0);

    VECTOR_DEF(M3, (2)(1));

    b00 = ADD(f00, f02);
    b01 = ADD(f10, f12);
    b02 = ADD(f20, f22);

    b03 = MUL(z1_144, f01);
    b04 = MUL(z1_72, f11);
    b05 = MUL(z1_36, f21);

    c0 = FMADD(z1_144, b00, b03);
    c1 = FMADD(z1_72, b01, b04);
    c2 = FMADD(z1_36, b02, b05);

    t01 = FNMADD(z6, c0, z0);
    ISTORE(0, 1);
    a00 = FMADD(z4, c0, c2);
    t11 = FMADD(z2, c1, a00);
    ISTORE(1, 1);
    t21 = FNMADD(z2, c1, a00);
    ISTORE(2, 1);
    a01 = ADD(c0, c2);
    t31 = FNMSUB(z1, a01, c1);
    ISTORE(3, 1);
    t41 = FNMADD(z1, a01, c1);
    ISTORE(4, 1);
    t51 = FNMADD(z6, c2, z0);
    ISTORE(5, 1);

    c0 = FMSUB(z1_144, b00, b03);
    c1 = FMSUB(z1_72, b01, b04);
    c2 = FMSUB(z1_36, b02, b05);

    t02 = FNMADD(z6, c0, z0);
    ISTORE(0, 2);
    a00 = FMADD(z4, c0, c2);
    t12 = FMADD(z2, c1, a00);
    ISTORE(1, 2);
    t22 = FNMADD(z2, c1, a00);
    ISTORE(2, 2);
    a01 = ADD(c0, c2);
    t32 = FNMSUB(z1, a01, c1);
    ISTORE(3, 2);
    t42 = FNMADD(z1, a01, c1);
    ISTORE(4, 2);
    t52 = FNMADD(z6, c2, z0);
    ISTORE(5, 2);

    b00 = FMADD(z4, f02, f00);
    b01 = FMADD(z4, f12, f10);
    b02 = FMADD(z4, f22, f20);

    b03 = MUL(z1_288, f01);
    b04 = MUL(z1_144, f11);
    b05 = MUL(z1_72, f21);

    c0 = FMADD(z1_576, b00, b03);
    c1 = FMADD(z1_288, b01, b04);
    c2 = FMADD(z1_144, b02, b05);

    t03 = MUL(z6, c0);
    ISTORE(0, 3);
    a00 = FNMSUB(z4, c0, c2);
    t13 = FNMADD(z2, c1, a00);
    ISTORE(1, 3);
    t23 = FMADD(z2, c1, a00);
    ISTORE(2, 3);
    a01 = ADD(c0, c2);
    t33 = ADD(a01, c1);
    ISTORE(3, 3);
    t43 = SUB(a01, c1);
    ISTORE(4, 3);
    t53 = MUL(z6, c2);
    ISTORE(5, 3);

    c0 = FMSUB(z1_576, b00, b03);
    c1 = FMSUB(z1_288, b01, b04);
    c2 = FMSUB(z1_144, b02, b05);

    t04 = MUL(z6, c0);
    ISTORE(0, 4);
    a00 = FMADD(z4, c0, c2);
    t14 = FNMSUB(z2, c1, a00);
    ISTORE(1, 4);
    t24 = FMSUB(z2, c1, a00);
    ISTORE(2, 4);
    a01 = ADD(c0, c2);
    t34 = ADD(a01, c1);
    ISTORE(3, 4);
    t44 = SUB(a01, c1);
    ISTORE(4, 4);
    t54 = MUL(z6, c2);
    ISTORE(5, 4);

    t05 = MUL(z1_4, f02);
    ISTORE(0, 5);
    a00 = MUL(z1_6, ADD(f02, f22));
    t15 = FNMSUB(z1_6, f12, a00);
    ISTORE(1, 5);
    t25 = FMSUB(z1_6, f12, a00);
    ISTORE(2, 5);
    a01 = MUL(z1_24, FMADD(z4, f22, f02));
    t35 = FMADD(z1_12, f12, a01);
    ISTORE(3, 5);
    t45 = FNMADD(z1_12, f12, a01);
    ISTORE(4, 5);
    t55 = f22;
    ISTORE(5, 5);
  }
}
} // namespace euler
