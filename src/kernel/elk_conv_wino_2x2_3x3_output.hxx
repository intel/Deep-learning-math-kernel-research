#pragma once
#include <x86intrin.h>
#include "el_intrin.hpp"
#include "elk_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename OutputType, typename BiasType,
    int format, bool is_border, bool with_bias, bool with_relu,
    bool with_ip_sum, int V>
struct elk_conv_wino_trans_output<float, OutputType, BiasType, format,
    is_border, with_bias, with_relu, with_ip_sum, ISA_SKX_AVX512, 4, 3, V> {
  constexpr static int A = 4;
  constexpr static int K = 3;

  static void execute(elx_conv_params_t &xc, OutputType *output,
      float *toutput, BiasType *bias, int hOA_end, int wOA_end)
  {
    ENABLE_AVX512F();

    __m<V> z = XOR(z, z);
    __m<V> mrepS, mzp;

    MD3(float, atoutput, toutput, A, A, V);
    if (std::is_same<OutputType, uint8_t>::value
        || std::is_same<OutputType, int8_t>::value) {
      mrepS = _mm<V>::set1_ps(xc.output_quant_repS);
      mzp = _mm<V>::set1_ps(xc.output_quant_z);
    }

    bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
    bool fuse_bias = with_bias && (bias != nullptr);
    bool fuse_relu = with_relu && (bias != nullptr);

    alignas(64) OutputType dummy[16];
    auto p_cb = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
        return &md3(aoutput, _h, _w, 0);
      } else if (format == TKF_BLOCKED) {
        MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      } else {
        MD3(OutputType, aoutput, output, xc.oh, xc.ow, xc.oc);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      }
    };

#undef P
#undef T
#undef OP
#undef BIAS
#define T(_h, _w) (&md3(atoutput, _h, _w, 0))
#define P(_h, _w) p_cb(_h, _w)

    __m<V> c0, c1, c2, c3, zero;
#define t(m, n) t##m##n
#define OP(m,n) __m<V> t(m,n) = _mm<V>::load_ps(T(m, n))

#define BIAS                                                                   \
  std::is_same<BiasType, float>::value                                         \
      ? *(__m<V> *)bias                                                        \
      : _mm<V>::cvtph_ps(_mm<V / 2>::load_si256((__m256i *)bias))

#define _cvtepu8_ps(addr)                                                      \
  ({                                                                           \
    _mm<V>::cvtepi32_ps(_mm<V>::cvtepu8_epi32(*(__m128i *)addr));              \
  })

#define _cvtepi8_ps(addr)                                                      \
  ({                                                                           \
    _mm<V>::cvtepi32_ps(_mm<V>::cvtepi8_epi32(*(__m128i *)addr));              \
  })

    MATRIX_DEF(4, 4);

    c0 = ADD(ADD(t10, t11), t12);
    c1 = ADD(ADD(t20, t21), t22);
    c2 = SUB(ADD(t12, t13), t11);
    c3 = SUB(ADD(t22, t23), t21);

    __m<V> p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
    __m<V> p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
    __m<V> p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
    __m<V> p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);

    if (fuse_bias) {
      p00 += BIAS;
      p10 += BIAS;
      p01 += BIAS;
      p11 += BIAS;
    }
    if (std::is_same<OutputType, uint8_t>::value
        || std::is_same<OutputType, int8_t>::value) {
      p00 = p00 * mrepS + mzp;
      p10 = p10 * mrepS + mzp;
      p01 = p01 * mrepS + mzp;
      p11 = p11 * mrepS + mzp;
    }
    if (fuse_ip_sum) {
      if (std::is_same<OutputType, uint8_t>::value) {
        p00 += _cvtepu8_ps(P(0, 0));
        p10 += _cvtepu8_ps(P(1, 0));
        p01 += _cvtepu8_ps(P(0, 1));
        p11 += _cvtepu8_ps(P(1, 1));
      } else if (std::is_same<OutputType, int8_t>::value) {
        p00 += _cvtepi8_ps(P(0, 0));
        p10 += _cvtepi8_ps(P(1, 0));
        p01 += _cvtepi8_ps(P(0, 1));
        p11 += _cvtepi8_ps(P(1, 1));
      } else {
        p00 += *(__m<V>*)P(0, 0);
        p10 += *(__m<V>*)P(1, 0);
        p01 += *(__m<V>*)P(0, 1);
        p11 += *(__m<V>*)P(1, 1);
      }
    }
    if (fuse_relu) {
      p00 = MAX(p00, z);
      p10 = MAX(p10, z);
      p01 = MAX(p01, z);
      p11 = MAX(p11, z);
    }

#undef OP
#define p_(m, n) p##m##n
#define OP(m,n)                                                                \
  if (std::is_same<OutputType, float>::value) {                                \
    _mm<V>::store_ps(P(m, n), p_(m, n));                                       \
  } else if (std::is_same<OutputType, uint8_t>::value) {                       \
    __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(                                \
        p_(m, n), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);              \
    __m128i mresu8 = _mm<V>::cvtusepi32_epi8(mresu32);                         \
    _mm_store_si128((__m128i *)P(m, n), mresu8);                               \
  } else if (std::is_same<OutputType, int8_t>::value) {                        \
    __i<V> mresi32 = _mm<V>::cvt_roundps_epi32(                                \
        p_(m, n), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);              \
    __m128i mresi8 = _mm<V>::cvtsepi32_epi8(mresi32);                          \
    _mm_store_si128((__m128i *)P(m, n), mresi8);                               \
  } else {                                                                     \
    auto f16 = _mm<V>::cvtps_ph(p_(m, n),                                      \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                        \
    _mm<V/2>::store_si256((__m256i *)P(m, n), f16);                            \
  }

    MATRIX_DEF(2, 2);
  }

}; // elk_conv_wino_trans_output
} // namespace euler
