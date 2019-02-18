#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename InputType, int format, bool is_border,
    int V>
struct elk_conv_wino_trans_input<float, InputType, format, is_border,
    ISA_SKX_AVX512, 4, V> {
  constexpr static int A = 4;

  static void execute(elx_conv_params_t &xc, float atinput[A][A][V],
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end)
  {
    ENABLE_AVX512F();

    // Inputs
    __m<V> f00, f01, f02, f03, f10, f11, f12, f13, f20, f21, f22, f23, f30, f31,
        f32, f33;
    // Cache
    __m<V> c1, c2;
    // Outputs
    __m<V> t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23, t30, t31,
        t32, t33;

    auto f_cb = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(InputType, ainput, input, A, A, V);
        if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else if (format == TKF_BLOCKED) {
        MD3(InputType, ainput, input, xc.ih, xc.iw, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return _mm<V>::setzero_ps();
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else { // TKF_NHWC
        MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.ic);
        // TODO: overflow on last V
        MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0),
            xc.ic4 * xc.ic3 * xc.I2, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return _mm<V>::setzero_ps();
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md2(ainput1, 0, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md2(ainput1, 0, 0));
          return _mm<V>::cvtph_ps(f16);
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

  MATRIX_DEF(4, 4);

   c1 = SUB(f12, f10);
   c2 = SUB(f20, f22);
   t00 =  SUB(SUB(f00, f02), c2);
   _mm<V>::store_ps(T(0, 0), t00);
   t10 = ADD(c1, c2);
   _mm<V>::store_ps(T(1, 0), t10);
   t20 = SUB(c2, c1);
   _mm<V>::store_ps(T(2, 0), t20);
   t30 = ADD(c1, SUB(f30, f32));
   _mm<V>::store_ps(T(3, 0), t30);

   c1 = SUB(f12, f11);
   c2 = SUB(f22, f21);
   t01 = SUB(SUB(f02, f01), c2);
   _mm<V>::store_ps(T(0, 1), t01);
   t11 = SUB(c2, c1);
   _mm<V>::store_ps(T(1, 1), t11);
   t21 = ADD(c2, c1);
   _mm<V>::store_ps(T(2, 1), t21);
   t31 = SUB(SUB(f32, f31), c1);
   _mm<V>::store_ps(T(3, 1), t31);

   c1 = ADD(f11, f12);
   c2 = ADD(f21, f22);
   t02 = SUB(ADD(f01, f02), c2);
   _mm<V>::store_ps(T(0, 2), t02);
   t12 = SUB(c2, c1);
   _mm<V>::store_ps(T(1, 2), t12);
   t22 = ADD(c2, c1);
   _mm<V>::store_ps(T(2, 2), t22);
   t32 = SUB(ADD(f31, f32), c1);
   _mm<V>::store_ps(T(3, 2), t32);

   c1 = SUB(f11, f13);
   c2 = SUB(f23, f21);
   t03 = SUB(SUB(f03, f01), c2);
   _mm<V>::store_ps(T(0, 3), t03);
   t13 = ADD(c1, c2);
   _mm<V>::store_ps(T(1, 3), t13);
   t23 = SUB(c2, c1);
   _mm<V>::store_ps(T(2, 3), t23);
   t33 = ADD(SUB(c1, f31), f33);
   _mm<V>::store_ps(T(3, 3), t33);
} // execute

}; // elk_conv_wino_trans_input

}//namespace euler
