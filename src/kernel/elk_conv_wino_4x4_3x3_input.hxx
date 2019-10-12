#pragma once
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elk_def.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename InputType, int format, bool is_border, int V>
struct elk_conv_wino_trans_input<float, InputType, format, is_border,
    ISA_AVX512, 6, V> {
  constexpr static int A = 6;

  static void execute(elx_param_t &ep, float *tinput,
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end)
  {
    MD3(float, atinput, tinput, A, A, V);
    // __m<V> mS, mz;

    // if (std::is_same<InputType, uint8_t>::value) {
    //   mS = _mm<V>::set1_ps(ep.input_quant_S);
    //   mz = _mm<V>::set1_ps(ep.input_quant_z);
    // }

#undef ldr_f32_impl
#undef ldr_f16_impl
#undef ldr_u8_impl

#define ldr_f32_impl(addr) \
  ({ \
    _mm<V>::load_ps(addr); \
  })

#define ldr_f16_impl(addr) \
  ({ \
    __m256i f16 = _mm<V / 2>::load_si256((__m256i *)addr); \
    _mm<V>::cvtph_ps(f16); \
  })

#define ldr_u8_impl(addr) \
  ({ \
    __i<V> isrcu8 = _mm512_cvtepu8_epi32(*(__m128i *)addr); \
    __m<V> msrcu8 = _mm512_cvt_roundepi32_ps(isrcu8, \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); \
    /*(msrcu8 - mz) * mS;*/ \
    (msrcu8); \
  })

    auto readin = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(InputType, ainput, input, A, A, V);
        if (std::is_same<InputType, float>::value) {
          return ldr_f32_impl(&md3(ainput, _h, _w, 0));
        } else if (std::is_same<InputType, uint8_t>::value) {
          return ldr_u8_impl(&md3(ainput, _h, _w, 0));
        } else {
          return ldr_f16_impl(&md3(ainput, _h, _w, 0));
        }
      } else if (format == TKF_BLOCKED) {
        MD3(InputType, ainput, input, ep.ih, ep.iw, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end)) {
          return _mm<V>::setzero_ps();
        } else if (std::is_same<InputType, float>::value) {
          return ldr_f32_impl(&md3(ainput, _h, _w, 0));
        } else if (std::is_same<InputType, uint8_t>::value) {
          return ldr_u8_impl(&md3(ainput, _h, _w, 0));
        } else {
          return ldr_f16_impl(&md3(ainput, _h, _w, 0));
        }
      } else { // TKF_NHWC
        MD3(InputType, ainput0, input, ep.ih, ep.iw, ep.ic);
        // TODO: overflow on last V
        MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0),
            ep.I4 * ep.I3 * ep.I2, V);
        if (is_border
            && (_h < hA_start || _w < wA_start || _h > hA_end || _w > wA_end)) {
          return _mm<V>::setzero_ps();
        } else if (std::is_same<InputType, float>::value) {
          return ldr_f32_impl(&md2(ainput1, 0, 0));
        } else if (std::is_same<InputType, uint8_t>::value) {
          return ldr_u8_impl(&md2(ainput1, 0, 0));
        } else {
          return ldr_f16_impl(&md2(ainput1, 0, 0));
        }
      }
    };

    auto z0 = _mm<V>::set1_ps(-2.25f);
    auto z1 = _mm<V>::set1_ps(-0.390625f);
    auto z2 = _mm<V>::set1_ps(0.87890625f);
    auto z3 = _mm<V>::set1_ps(-2.640625f);
    auto z4 = _mm<V>::set1_ps(0.625f);
    auto z5 = _mm<V>::set1_ps(1.5f);

    auto f00 = readin(0, 0);
    auto f01 = readin(0, 1);
    auto f02 = readin(0, 2);
    auto f03 = readin(0, 3);
    auto f04 = readin(0, 4);
    auto f05 = readin(0, 5);

    auto f10 = readin(1, 0);
    auto f11 = readin(1, 1);
    auto f12 = readin(1, 2);
    auto f13 = readin(1, 3);
    auto f14 = readin(1, 4);
    auto f15 = readin(1, 5);

    auto f20 = readin(2, 0);
    auto f21 = readin(2, 1);
    auto f22 = readin(2, 2);
    auto f23 = readin(2, 3);
    auto f24 = readin(2, 4);
    auto f25 = readin(2, 5);

    auto f30 = readin(3, 0);
    auto f31 = readin(3, 1);
    auto f32 = readin(3, 2);
    auto f33 = readin(3, 3);
    auto f34 = readin(3, 4);
    auto f35 = readin(3, 5);

    auto f40 = readin(4, 0);
    auto f41 = readin(4, 1);
    auto f42 = readin(4, 2);
    auto f43 = readin(4, 3);
    auto f44 = readin(4, 4);
    auto f45 = readin(4, 5);

    auto f50 = readin(5, 0);

    auto t0 = f20 * z0 + f40;
    auto t1 = f10 * z0 + f30;
    auto t2 = f20 * z1 + f40;
    auto t3 = f10 * z1 + f30;
    auto t4 = f00 * z2 + f40;
    auto t5 = f10 * z2 + f50;

    auto m00 = f20 * z3 + t4;
    auto m10 = t1 * z4 + t0;
    auto m20 = t0 - t1 * z4;
    auto m30 = t3 * z5 + t2;
    auto m40 = t2 - t3 * z5;
    auto m50 = f30 * z3 + t5;

    auto f51 = readin(5, 1);

    t0 = f21 * z0 + f41;
    t1 = f11 * z0 + f31;
    t2 = f21 * z1 + f41;
    t3 = f11 * z1 + f31;
    t4 = f01 * z2 + f41;
    t5 = f11 * z2 + f51;

    auto m01 = f21 * z3 + t4;
    auto m11 = t1 * z4 + t0;
    auto m21 = t0 - t1 * z4;
    auto m31 = t3 * z5 + t2;
    auto m41 = t2 - t3 * z5;
    auto m51 = f31 * z3 + t5;

    auto f52 = readin(5, 2);

    t0 = f22 * z0 + f42;
    t1 = f12 * z0 + f32;
    t2 = f22 * z1 + f42;
    t3 = f12 * z1 + f32;
    t4 = f02 * z2 + f42;
    t5 = f12 * z2 + f52;

    auto m02 = f22 * z3 + t4;
    auto m12 = t1 * z4 + t0;
    auto m22 = t0 - t1 * z4;
    auto m32 = t3 * z5 + t2;
    auto m42 = t2 - t3 * z5;
    auto m52 = f32 * z3 + t5;

    auto f53 = readin(5, 3);

    t0 = f23 * z0 + f43;
    t1 = f13 * z0 + f33;
    t2 = f23 * z1 + f43;
    t3 = f13 * z1 + f33;
    t4 = f03 * z2 + f43;
    t5 = f13 * z2 + f53;

    auto m03 = f23 * z3 + t4;
    auto m13 = t1 * z4 + t0;
    auto m23 = t0 - t1 * z4;
    auto m33 = t3 * z5 + t2;
    auto m43 = t2 - t3 * z5;
    auto m53 = f33 * z3 + t5;

    auto f54 = readin(5, 4);

    t0 = f24 * z0 + f44;
    t1 = f14 * z0 + f34;
    t2 = f24 * z1 + f44;
    t3 = f14 * z1 + f34;
    t4 = f04 * z2 + f44;
    t5 = f14 * z2 + f54;

    auto m04 = f24 * z3 + t4;
    auto m14 = t1 * z4 + t0;
    auto m24 = t0 - t1 * z4;
    auto m34 = t3 * z5 + t2;
    auto m44 = t2 - t3 * z5;
    auto m54 = f34 * z3 + t5;

    auto f55 = readin(5, 5);

    t0 = f25 * z0 + f45;
    t1 = f15 * z0 + f35;
    t2 = f25 * z1 + f45;
    t3 = f15 * z1 + f35;
    t4 = f05 * z2 + f45;
    t5 = f15 * z2 + f55;

    auto m05 = f25 * z3 + t4;
    auto m15 = t1 * z4 + t0;
    auto m25 = t0 - t1 * z4;
    auto m35 = t3 * z5 + t2;
    auto m45 = t2 - t3 * z5;
    auto m55 = f35 * z3 + t5;

    auto f0 = m00;
    auto f1 = m01;
    auto f2 = m02;
    auto f3 = m03;
    auto f4 = m04;
    auto f5 = m05;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 0, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 0, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 0, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 0, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 0, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 0, 5, 0)) = f3 * z3 + t5;

    f0 = m10;
    f1 = m11;
    f2 = m12;
    f3 = m13;
    f4 = m14;
    f5 = m15;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 1, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 1, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 1, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 1, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 1, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 1, 5, 0)) = f3 * z3 + t5;

    f0 = m20;
    f1 = m21;
    f2 = m22;
    f3 = m23;
    f4 = m24;
    f5 = m25;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 2, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 2, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 2, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 2, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 2, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 2, 5, 0)) = f3 * z3 + t5;

    f0 = m30;
    f1 = m31;
    f2 = m32;
    f3 = m33;
    f4 = m34;
    f5 = m35;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 3, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 3, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 3, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 3, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 3, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 3, 5, 0)) = f3 * z3 + t5;

    f0 = m40;
    f1 = m41;
    f2 = m42;
    f3 = m43;
    f4 = m44;
    f5 = m45;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 4, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 4, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 4, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 4, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 4, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 4, 5, 0)) = f3 * z3 + t5;

    f0 = m50;
    f1 = m51;
    f2 = m52;
    f3 = m53;
    f4 = m54;
    f5 = m55;

    t0 = f2 * z0 + f4;
    t1 = f1 * z0 + f3;
    t2 = f2 * z1 + f4;
    t3 = f1 * z1 + f3;
    t4 = f0 * z2 + f4;
    t5 = f1 * z2 + f5;

    *(__m<V> *)(&md3(atinput, 5, 0, 0)) = f2 * z3 + t4;
    *(__m<V> *)(&md3(atinput, 5, 1, 0)) = t1 * z4 + t0;
    *(__m<V> *)(&md3(atinput, 5, 2, 0)) = t0 - t1 * z4;
    *(__m<V> *)(&md3(atinput, 5, 3, 0)) = t3 * z5 + t2;
    *(__m<V> *)(&md3(atinput, 5, 4, 0)) = t2 - t3 * z5;
    *(__m<V> *)(&md3(atinput, 5, 5, 0)) = f3 * z3 + t5;
  }

}; // elk_conv_wino_trans_input

} // euler
