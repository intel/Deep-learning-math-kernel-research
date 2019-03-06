#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename InputType, int format, bool is_border>
struct elk_conv_wino_trans_input
    <short, InputType, format, is_border, ISA_SKX_AVX512, 4, 16> {
  constexpr static int A = 4;
  constexpr static int V = 16;

  static void execute(elx_conv_params_t &xc, short atinput[A][A][V],
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end)
  {
    ENABLE_AVX512F();

    // Inputs
    __m256i f00, f01, f02, f03, f10, f11, f12, f13, f20, f21, f22, f23, f30, f31,
        f32, f33;
    // Cache
    __m256i c1, c2;
    // Outputs
    __m256i t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31,
        t32, t33;
    // __m<V> mS, mz;

    // if (std::is_same<InputType, uint8_t>::value) {
    //   mS = _mm<V>::set1_ps(xc.input_quant_S);
    //   mz = _mm<V>::set1_ps(xc.input_quant_z);
    // }

#undef ldr_u8s8_impl

#define ldr_u8s8_impl(addr) \
  ({ \
    __m256i isrcu8; \
    if (std::is_same<InputType, uint8_t>::value) { \
      isrcu8 = _mm256_cvtepu8_epi16(*(__m128i *)addr); \
    } else { \
      isrcu8 = _mm256_cvtepi8_epi16(*(__m128i *)addr); \
    } \
    (isrcu8); \
  })

  auto f_cb = [&](int _h, int _w) {
    if (format == TKF_COMPACT) {
      MD3(InputType, ainput, input, A, A, V);
      return ldr_u8s8_impl(&md3(ainput, _h, _w, 0));
    } else if (format == TKF_BLOCKED) {
      MD3(InputType, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end)) {
        return _mm256_setzero_si256();
      } else {
        return ldr_u8s8_impl(&md3(ainput, _h, _w, 0));
      }
    } else { // TKF_NHWC
      MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.ic);
      // TODO: overflow on last V
      MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0),
          xc.ic4 * xc.ic3 * xc.I2, V);
      if (is_border
          && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end)) {
        return _mm256_setzero_si256();
      } else {
        return ldr_u8s8_impl(&md2(ainput1, 0, 0));
      }
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)
#undef _add_epi16
#undef _sub_epi16
#undef _store_si256
#define _add_epi16 _mm256_add_epi16
#define _sub_epi16 _mm256_sub_epi16
#define _store_si256(addr, var) _mm256_store_si256((__m256i *)addr, var)

  MATRIX_DEF(4, 4);

  c1 = _sub_epi16(f12, f10);
  c2 = _sub_epi16(f20, f22);
  t00 =  _sub_epi16(_sub_epi16(f00, f02), c2);
  _store_si256(T(0, 0), t00);
  t10 = _add_epi16(c1, c2);
  _store_si256(T(1, 0), t10);
  t20 = _sub_epi16(c2, c1);
  _store_si256(T(2, 0), t20);
  t30 = _add_epi16(c1, _sub_epi16(f30, f32));
  _store_si256(T(3, 0), t30);

  c1 = _sub_epi16(f12, f11);
  c2 = _sub_epi16(f22, f21);
  t01 = _sub_epi16(_sub_epi16(f02, f01), c2);
  _store_si256(T(0, 1), t01);
  t11 = _sub_epi16(c2, c1);
  _store_si256(T(1, 1), t11);
  t21 = _add_epi16(c2, c1);
  _store_si256(T(2, 1), t21);
  t31 = _sub_epi16(_sub_epi16(f32, f31), c1);
  _store_si256(T(3, 1), t31);

  c1 = _add_epi16(f11, f12);
  c2 = _add_epi16(f21, f22);
  t02 = _sub_epi16(_add_epi16(f01, f02), c2);
  _store_si256(T(0, 2), t02);
  t12 = _sub_epi16(c2, c1);
  _store_si256(T(1, 2), t12);
  t22 = _add_epi16(c2, c1);
  _store_si256(T(2, 2), t22);
  t32 = _sub_epi16(_add_epi16(f31, f32), c1);
  _store_si256(T(3, 2), t32);

  c1 = _sub_epi16(f11, f13);
  c2 = _sub_epi16(f23, f21);
  t03 = _sub_epi16(_sub_epi16(f03, f01), c2);
  _store_si256(T(0, 3), t03);
  t13 = _add_epi16(c1, c2);
  _store_si256(T(1, 3), t13);
  t23 = _sub_epi16(c2, c1);
  _store_si256(T(2, 3), t23);
  t33 = _add_epi16(_sub_epi16(c1, f31), f33);
  _store_si256(T(3, 3), t33);
} // execute

}; // elk_conv_wino_trans_input

}//namespace euler
