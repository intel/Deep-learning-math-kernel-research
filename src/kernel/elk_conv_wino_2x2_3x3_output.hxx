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
    is_border, with_bias, with_relu, with_ip_sum, ISA_AVX512, 4, 3, V> {
  constexpr static int A = 4;
  constexpr static int K = 3;

  static void execute(elx_param_t &ep, OutputType *output,
      float *toutput, BiasType *bias, int hOA_end, int wOA_end)
  {
    ENABLE_AVX512F();

    __m<V> z = _mm<V>::xor_ps(z, z);
    __m<V> mrepS, mzp;

    MD3(float, atoutput, toutput, A, A, V);
    if (std::is_same<OutputType, uint8_t>::value
        || std::is_same<OutputType, int8_t>::value) {
      mrepS = _mm<V>::set1_ps(ep.output_quant_repS);
      mzp = _mm<V>::set1_ps(ep.output_quant_z);
    }

    bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
    bool fuse_bias = with_bias && (bias != nullptr);
    bool fuse_relu = with_relu && (bias != nullptr);

    alignas(64) OutputType dummy[16];
    auto out_ptr = [&](int _h, int _w) {
      if (format == TKF_COMPACT) {
        MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
        return &md3(aoutput, _h, _w, 0);
      } else if (format == TKF_BLOCKED) {
        MD3(OutputType, aoutput, output, ep.oh, ep.ow, V);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      } else {
        MD3(OutputType, aoutput, output, ep.oh, ep.ow, ep.oc);
        if (is_border && (_h > hOA_end || _w > wOA_end))
          return dummy;
        else
          return &md3(aoutput, _h, _w, 0);
      }
    };

#undef BIAS
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

    __m<V> c0, c1, c2, c3, zero;
    __m<V> t00 = _mm<V>::load_ps((&md3(atoutput, 0, 0, 0)));
    __m<V> t01 = _mm<V>::load_ps((&md3(atoutput, 0, 1, 0)));
    __m<V> t02 = _mm<V>::load_ps((&md3(atoutput, 0, 2, 0)));
    __m<V> t03 = _mm<V>::load_ps((&md3(atoutput, 0, 3, 0)));
    __m<V> t10 = _mm<V>::load_ps((&md3(atoutput, 1, 0, 0)));
    __m<V> t11 = _mm<V>::load_ps((&md3(atoutput, 1, 1, 0)));
    __m<V> t12 = _mm<V>::load_ps((&md3(atoutput, 1, 2, 0)));
    __m<V> t13 = _mm<V>::load_ps((&md3(atoutput, 1, 3, 0)));
    __m<V> t20 = _mm<V>::load_ps((&md3(atoutput, 2, 0, 0)));
    __m<V> t21 = _mm<V>::load_ps((&md3(atoutput, 2, 1, 0)));
    __m<V> t22 = _mm<V>::load_ps((&md3(atoutput, 2, 2, 0)));
    __m<V> t23 = _mm<V>::load_ps((&md3(atoutput, 2, 3, 0)));
    __m<V> t30 = _mm<V>::load_ps((&md3(atoutput, 3, 0, 0)));
    __m<V> t31 = _mm<V>::load_ps((&md3(atoutput, 3, 1, 0)));
    __m<V> t32 = _mm<V>::load_ps((&md3(atoutput, 3, 2, 0)));
    __m<V> t33 = _mm<V>::load_ps((&md3(atoutput, 3, 3, 0)));

    c0 = t10 + t11 + t12;
    c1 = t20 + t21 + t22;
    c2 = t12 + t13 - t11;
    c3 = t22 + t23 - t21;

    __m<V> p00 = t00 + t01 + t02 + c0 + c1;
    __m<V> p10 = c1 - c0 + t30 + t31 + t32;
    __m<V> p01 = t02 - t01 + t03 + c2 + c3;
    __m<V> p11 = c3 - c2 - t31 + t32 + t33;

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
        p00 += _cvtepu8_ps(out_ptr(0, 0));
        p10 += _cvtepu8_ps(out_ptr(1, 0));
        p01 += _cvtepu8_ps(out_ptr(0, 1));
        p11 += _cvtepu8_ps(out_ptr(1, 1));
      } else if (std::is_same<OutputType, int8_t>::value) {
        p00 += _cvtepi8_ps(out_ptr(0, 0));
        p10 += _cvtepi8_ps(out_ptr(1, 0));
        p01 += _cvtepi8_ps(out_ptr(0, 1));
        p11 += _cvtepi8_ps(out_ptr(1, 1));
      } else {
        p00 += *(__m<V>*)out_ptr(0, 0);
        p10 += *(__m<V>*)out_ptr(1, 0);
        p01 += *(__m<V>*)out_ptr(0, 1);
        p11 += *(__m<V>*)out_ptr(1, 1);
      }
    }
    if (fuse_relu) {
      p00 = _mm<V>::max_ps(p00, z);
      p10 = _mm<V>::max_ps(p10, z);
      p01 = _mm<V>::max_ps(p01, z);
      p11 = _mm<V>::max_ps(p11, z);
    }

#undef STORE
#define p_(m, n) p##m##n
#define STORE(m,n)                                                             \
  if (std::is_same<OutputType, float>::value) {                                \
    _mm<V>::store_ps(out_ptr(m, n), p_(m, n));                                 \
  } else if (std::is_same<OutputType, uint8_t>::value) {                       \
    __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(                                \
        p_(m, n), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);                  \
    __m128i mresu8 = _mm<V>::cvtusepi32_epi8(mresu32);                         \
    _mm_store_si128((__m128i *)out_ptr(m, n), mresu8);                         \
  } else if (std::is_same<OutputType, int8_t>::value) {                        \
    __i<V> mresi32 = _mm<V>::cvt_roundps_epi32(                                \
        p_(m, n), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);                  \
    __m128i mresi8 = _mm<V>::cvtsepi32_epi8(mresi32);                          \
    _mm_store_si128((__m128i *)out_ptr(m, n), mresi8);                         \
  } else {                                                                     \
    auto f16 = _mm<V>::cvtps_ph(p_(m, n),                                      \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                        \
    _mm<V/2>::store_si256((__m256i *)out_ptr(m, n), f16);                      \
  }

    STORE(0, 0);
    STORE(0, 1);
    STORE(1, 0);
    STORE(1, 1);
  }

}; // elk_conv_wino_trans_output
} // namespace euler
