#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include <math.h>

namespace euler {

#define AVX512_CALCULATE_I_5(n)                                                \
  t0##n = -((z17_10 * c3 - (c1 + c2)) + (z17_5 * c4 - (c5 + c6)));             \
  _mm<V>::store_ps(&md3(atinput, 0, n, 0), t0##n);                             \
  t1##n = ((z17_5 * c4 + (c5 - c6)) - (z17_10 * c3 - (c1 - c2)));              \
  _mm<V>::store_ps(&md3(atinput, 1, n, 0), t1##n);                             \
  t2##n = -(((z1_2 * c1 - c3) + (z1_4 * c2 - c4)) + (z2 * c5 + c6));           \
  _mm<V>::store_ps(&md3(atinput, 2, n, 0), t2##n);                             \
  t3##n = (((z1_2 * c1 - c3) - (z1_4 * c2 - c4)) + (z2 * c5 - c6));            \
  _mm<V>::store_ps(&md3(atinput, 3, n, 0), t3##n);                             \
  t4##n = ((z4 * (c4 - c2) - (z1_2 * c5 + c6)) - (z2 * c1 - c3));              \
  _mm<V>::store_ps(&md3(atinput, 4, n, 0), t4##n);                             \
  t5##n = ((z4 * (c4 - c2) + (z1_2 * c5 - c6)) + (z2 * c1 - c3));              \
  _mm<V>::store_ps(&md3(atinput, 5, n, 0), t5##n);

template <typename InputType, int format, bool is_border,
    int V>
struct elk_conv_wino_trans_input<float, InputType, format, is_border,
    ISA_AVX512, 7, V> {
  constexpr static int A = 7;

  static void execute(elx_param_t &ep, float *tinput,
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end)
  {
    ENABLE_AVX512F();
    MD3(float, atinput, tinput, A, A, V);

    // Inputs
    __m<V> f00, f01, f02, f03, f04, f05, f06, f10, f11, f12, f13, f14, f15, f16,
        f20, f21, f22, f23, f24, f25, f26, f30, f31, f32, f33, f34, f35, f36,
        f40, f41, f42, f43, f44, f45, f46, f50, f51, f52, f53, f54, f55, f56,
        f60, f61, f62, f63, f64, f65, f66;
    // Cache
    __m<V> c1, c2, c3, c4, c5, c6;
    // Outputs
    __m<V> t00, t01, t02, t03, t04, t05, t06, t10, t11, t12, t13, t14, t15, t16,
        t20, t21, t22, t23, t24, t25, t26, t30, t31, t32, t33, t34, t35, t36,
        t40, t41, t42, t43, t44, t45, t46, t50, t51, t52, t53, t54, t55, t56,
        t60, t61, t62, t63, t64, t65, t66;

    __m<V> z0 = _mm<V>::setzero_ps();
    __m<V> z2 = _mm<V>::set1_ps(2.0f);
    __m<V> z4 = _mm<V>::set1_ps(4.0f);
    __m<V> z1_2 = _mm<V>::set1_ps(1.0f / 2.0f);
    __m<V> z1_4 = _mm<V>::set1_ps(1.0f / 4.0f);
    __m<V> z5_2 = _mm<V>::set1_ps(5.0f / 2.0f);
    __m<V> z5_4 = _mm<V>::set1_ps(5.0f / 4.0f);
    __m<V> z17_4 = _mm<V>::set1_ps(17.0f / 4.0f);
    __m<V> z17_5 = _mm<V>::set1_ps(17.0f / 5.0f);
    __m<V> z17_10 = _mm<V>::set1_ps(17.0f / 10.0f);
    __m<V> z21_4 = _mm<V>::set1_ps(21.0f / 4.0f);
    __m<V> z21_10 = _mm<V>::set1_ps(21.0f / 10.0f);
    __m<V> z85_8 = _mm<V>::set1_ps(85.0f / 8.0f);
    __m<V> z85_16 = _mm<V>::set1_ps(85.0f / 16.0f);

    auto readin = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(InputType, ainput, input, A, A, V);
        if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else if (format == TKF_BLOCKED) {
        MD3(InputType, ainput, input, ep.ih, ep.iw, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return z0;
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      } else { // TKF_NHWC
        MD3(InputType, ainput0, input, ep.ih, ep.iw, ep.ic);
        // TODO: overflow on last V
        MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0),
            ep.I4 * ep.I3 * ep.I2, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end))
          return z0;
        else if (std::is_same<InputType, float>::value)
          return _mm<V>::load_ps(&md2(ainput1, 0, 0));
        else {
          auto f16 = _mm<V / 2>::load_si256((__m256i *)&md2(ainput1, 0, 0));
          return _mm<V>::cvtph_ps(f16);
        }
      }
    };


    f00 = readin(0, 0); f01 = readin(0, 1); f02 = readin(0, 2); f03 = readin(0, 3); f04 = readin(0, 4); f05 = readin(0, 5);
    f10 = readin(1, 0); f11 = readin(1, 1); f12 = readin(1, 2); f13 = readin(1, 3); f14 = readin(1, 4); f15 = readin(1, 5);
    f20 = readin(2, 0); f21 = readin(2, 1); f22 = readin(2, 2); f23 = readin(2, 3); f24 = readin(2, 4); f25 = readin(2, 5);
    f30 = readin(3, 0); f31 = readin(3, 1); f32 = readin(3, 2); f33 = readin(3, 3); f34 = readin(3, 4); f35 = readin(3, 5);
    f40 = readin(4, 0); f41 = readin(4, 1); f42 = readin(4, 2); f43 = readin(4, 3); f44 = readin(4, 4); f45 = readin(4, 5);
    f50 = readin(5, 0); f51 = readin(5, 1); f52 = readin(5, 2); f53 = readin(5, 3); f54 = readin(5, 4); f55 = readin(5, 5);
    f60 = readin(6, 0); f61 = readin(6, 1); f62 = readin(6, 2); f63 = readin(6, 3); f64 = readin(6, 4); f65 = readin(6, 5);

    c1 = ((f00 + f01) - (z17_4 * (f02 + f03) - (f04 + f05)));
    c1 = ((f00 + f01) - (z17_4 * (f02 + f03) - (f04 + f05)));
    c2 = ((f10 + f11) - (z17_4 * (f12 + f13) - (f14 + f15)));
    c3 = (z5_2 * ((f20 + f21) + (f24 + f25)) - (z85_8 * (f22 + f23)));
    c4 = (z5_4 * ((f30 + f31) + (f34 + f35)) - (z85_16 * (f32 + f33)));
    c5 = ((f40 + f41) - (z17_4 * (f42 + f43) - (f44 + f45)));
    c6 = ((f50 + f51) - (z17_4 * (f52 + f53) - (f54 + f55)));
    AVX512_CALCULATE_I_5(0)
    t60 = (((z21_4 * c5 - (f60 + f61)) + (z17_4 * (f62 + f63) - (f64 + f65))) - (z21_10 * c3 - c1));
    _mm<V>::store_ps(&md3(atinput, 6, 0, 0), t60);

    c1 = ((f00 - f01) + (z17_4 * (f03 - f02) + (f04 - f05)));
    c2 = ((f10 - f11) + (z17_4 * (f13 - f12) + (f14 - f15)));
    c3 = (z5_2 * ((f20 - f21) + (f24 - f25)) + (z85_8 * (f23 - f22)));
    c4 = (z5_4 * ((f30 - f31) + (f34 - f35)) + (z85_16 * (f33 - f32)));
    c5 = ((f40 - f41) + (z17_4 * (f43 - f42) + (f44 - f45)));
    c6 = ((f50 - f51) + (z17_4 * (f53 - f52) + (f54 - f55)));
    AVX512_CALCULATE_I_5(1)
    t61 = (((z21_4 * c5 - (f60 - f61)) + (z17_4 * (f62 - f63) - (f64 - f65))) - (z21_10 * c3 - c1));
    _mm<V>::store_ps(&md3(atinput, 6, 1, 0), t61);

    __m<V> z5 = _mm<V>::set1_ps(5.0f);
    __m<V> z5_8 = _mm<V>::set1_ps(5.0f / 8.0f);
    __m<V> z5_16 = _mm<V>::set1_ps(5.0f / 16.0f);
    __m<V> z25_4 = _mm<V>::set1_ps(25.0f / 4.0f);
    __m<V> z25_8 = _mm<V>::set1_ps(25.0f / 8.0f);
    __m<V> z25_16 = _mm<V>::set1_ps(25.0f / 16.0f);

    c1 = (z5_2 * f02 + (z5_4 * f03 - (z1_2 * f00 + (z1_4 * f01 + (z2 * f04 + f05)))));
    c2 = (z5_2 * f12 + (z5_4 * f13 - (z1_2 * f10 + (z1_4 * f11 + (z2 * f14 + f15)))));
    c3 = (z25_4 * f22 + (z25_8 * f23 - (z5_4 * f20 + (z5_8 * f21 + (z5 * f24 + z5_2 * f25)))));
    c4 = (z25_8 * f32 + (z25_16 * f33 - (z5_8 * f30 + (z5_16 * f31 + (z5_2 * f34 + z5_4 * f35)))));
    c5 = (z5_2 * f42 + (z5_4 * f43 - (z1_2 * f40 + (z1_4 * f41 + (z2 * f44 + f45)))));
    c6 = (z5_2 * f52 + (z5_4 * f53 - (z1_2 * f50 + (z1_4 * f51 + (z2 * f54 + f55)))));
    AVX512_CALCULATE_I_5(2)
    t62 = (z21_4 * c5 + (z1_2 * f60 + (z1_4 * f61 + (z2 * f64 - (z21_10 * c3 + (z5_2 * f62 + (z5_4 * f63 - (c1 + f65))))))));
    _mm<V>::store_ps(&md3(atinput, 6, 2, 0), t62);

    c1 = (z1_2 * f00 + (z5_4 * f03 + (z2 * f04 - (z1_4 * f01 + (z5_2 * f02 + f05)))));
    c2 = (z1_2 * f10 + (z5_4 * f13 + (z2 * f14 - (z1_4 * f11 + (z5_2 * f12 + f15)))));
    c3 = (z5_4 * f20 + (z25_8 * f23 + (z5 * f24 - (z5_8 * f21 + (z25_4 * f22 + z5_2 * f25)))));
    c4 = (z5_8 * f30 + (z25_16 * f33 + (z5_2 * f34 - (z5_16 * f31 + (z25_8 * f32 + z5_4 * f35)))));
    c5 = (z1_2 * f40 + (z5_4 * f43 + (z2 * f44 - (z1_4 * f41 + (z5_2 * f42 + f45)))));
    c6 = (z1_2 * f50 + (z5_4 * f53 + (z2 * f54 - (z1_4 * f51 + (z5_2 * f52 + f55)))));
    AVX512_CALCULATE_I_5(3)
    t63 = (z21_4 * c5 + (z1_4 * f61 + (z5_2 * f62 - (z21_10 * c3 + (z1_2 * f60 + (z5_4 * f63 + (z2 * f64 - (c1 + f65))))))));
    _mm<V>::store_ps(&md3(atinput, 6, 3, 0), t63);

    __m<V> z10 = _mm<V>::set1_ps(10.0f);
    __m<V> z25_2 = _mm<V>::set1_ps(25.0f / 2.0f);

    c1 = (z5_2 * f02 + (z5 * f03 - (z2 * f00 + (z4 * f01 + (z1_2 * f04 + f05)))));
    c2 = (z5_2 * f12 + (z5 * f13 - (z2 * f10 + (z4 * f11 + (z1_2 * f14 + f15)))));
    c3 = (z25_4 * f22 + (z25_2 * f23 - (z5 * f20 + (z10 * f21 + (z5_4 * f24 + z5_2 * f25)))));
    c4 = (z25_8 * f32 + (z25_4 * f33 - (z5_2 * f30 + (z5 * f31 + (z5_8 * f34 + z5_4 * f35)))));
    c5 = (z5_2 * f42 + (z5 * f43 - (z2 * f40 + (z4 * f41 + (z1_2 * f44 + f45)))));
    c6 = (z5_2 * f52 + (z5 * f53 - (z2 * f50 + (z4 * f51 + (z1_2 * f54 + f55)))));
    AVX512_CALCULATE_I_5(4)
    t64 = (z21_4 * c5 + (z2 * f60 + (z4 * f61 + (z1_2 * f64 - (z21_10 * c3 + (z5_2 * f62 + (z5 * f63 - (c1 + f65))))))));
    _mm<V>::store_ps(&md3(atinput, 6, 4, 0), t64);

    c1 = (z2 * f00 + (z5 * f03 + (z1_2 * f04 - (z4 * f01 + (z5_2 * f02 + f05)))));
    c2 = (z2 * f10 + (z5 * f13 + (z1_2 * f14 - (z4 * f11 + (z5_2 * f12 + f15)))));
    c3 = (z5 * f20 + (z25_2 * f23 + (z5_4 * f24 - (z10 * f21 + (z25_4 * f22 + z5_2 * f25)))));
    c4 = (z5_2 * f30 + (z25_4 * f33 + (z5_8 * f34 - (z5 * f31 + (z25_8 * f32 + z5_4 * f35)))));
    c5 = (z2 * f40 + (z5 * f43 + (z1_2 * f44 - (z4 * f41 + (z5_2 * f42 + f45)))));
    c6 = (z2 * f50 + (z5 * f53 + (z1_2 * f54 - (z4 * f51 + (z5_2 * f52 + f55)))));
    AVX512_CALCULATE_I_5(5)
    t65 = (z21_4 * c5 + (z4 * f61 + (z5_2 * f62 - (z21_10 * c3 + (z2 * f60 + (z5 * f63 + (z1_2 * f64 - (c1 + f65))))))));
    _mm<V>::store_ps(&md3(atinput, 6, 5, 0), t65);

    f06 = readin(0, 6);
    f16 = readin(1, 6);
    f26 = readin(2, 6);
    f36 = readin(3, 6);
    f46 = readin(4, 6);
    f56 = readin(5, 6);
    f66 = readin(6, 6);
    __m<V> z105_8 = _mm<V>::set1_ps(105.0f / 8.0f);
    __m<V> z105_16 = _mm<V>::set1_ps(105.0f / 16.0f);

    c1 = (z21_4 * (f04 - f02) + (f00 - f06));
    c2 = (z21_4 * (f14 - f12) + (f10 - f16));
    c3 = (z5_2 * (f20 - f26) + (z105_8 * (f24 - f22)));
    c4 = (z5_4 * (f30 - f36) + (z105_16 * (f34 - f32)));
    c5 = (z21_4 * (f44 - f42) + (f40 - f46));
    c6 = (z21_4 * (f54 - f52) + (f50 - f56));
    AVX512_CALCULATE_I_5(6)
    t66 = ((z21_4 * ((c5 + f62) - f64) + (c1 + f66)) - (z21_10 * c3 + f60));
    _mm<V>::store_ps(&md3(atinput, 6, 6, 0), t66);
  }
}; // elk_conv_wino_trans_input
} // namespace euler
