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
    is_border, with_bias, with_relu, with_ip_sum, ISA_AVX512, 5, 3, V> {
  constexpr static int A = 5;
  constexpr static int K = 3;

  static void execute(elx_param_t &ep, OutputType *output,
      float *toutput, BiasType *bias, int hOA_end, int wOA_end)
  {
    ENABLE_AVX512F();

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

    alignas(64) OutputType dummy[V];
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
#undef STORE

#define BIAS                                                                   \
  std::is_same<BiasType, float>::value                                         \
  ? *(__m<V>*)bias                                                             \
  : _mm<V>::cvtph_ps(_mm<V/2>::load_si256((__m256i *)bias))

#define _cvtepu8_ps(addr)                                                      \
  ({                                                                           \
    _mm<V>::cvtepi32_ps(_mm<V>::cvtepu8_epi32(*(__m128i *)addr));              \
  })

#define _cvtepi8_ps(addr)                                                      \
  ({                                                                           \
    _mm<V>::cvtepi32_ps(_mm<V>::cvtepi8_epi32(*(__m128i *)addr));              \
  })

#define STORE(i, j)                                                            \
  if (std::is_same<OutputType, float>::value) {                                \
    _mm<V>::store_ps(out_ptr(i, j), p##j);                                     \
  } else if (std::is_same<OutputType, uint8_t>::value) {                       \
    __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(                                \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    __m128i mresu8 = _mm<V>::cvtusepi32_epi8(mresu32);                         \
    _mm_store_si128((__m128i *)out_ptr(i, j), mresu8);                         \
  } else if (std::is_same<OutputType, int8_t>::value) {                        \
    __i<V> mresi32 = _mm<V>::cvt_roundps_epi32(                                \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    __m128i mresi8 = _mm<V>::cvtsepi32_epi8(mresi32);                          \
    _mm_store_si128((__m128i *)out_ptr(i, j), mresi8);                         \
  } else {                                                                     \
    auto f16 = _mm<V>::cvtps_ph(                                               \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    _mm<V / 2>::store_si256((__m256i *)out_ptr(i, j), f16);                    \
  }

    __m<V> M[3][5];

    auto z0 = _mm<V>::set1_ps(0.3333333333333333f);
    auto z1 = _mm<V>::set1_ps(0.6666666666666666f);
    auto z2 = _mm<V>::set1_ps(1.3333333333333333f);
    __m<V> z = _mm<V>::xor_ps(z, z);

#pragma unroll
    for (int i = 0; i < 5; i++) {
      auto f0 = _mm<V>::load_ps(&md3(atoutput, 0, i, 0));
      auto f1 = _mm<V>::load_ps(&md3(atoutput, 1, i, 0));
      auto f2 = _mm<V>::load_ps(&md3(atoutput, 2, i, 0));
      auto f3 = _mm<V>::load_ps(&md3(atoutput, 3, i, 0));
      auto f4 = _mm<V>::load_ps(&md3(atoutput, 4, i, 0));

      auto t0 = f2 * z0;
      auto t1 = t0 + f1;

      M[0][i] = f3 * z2 + f0 + t1;
      M[1][i] = t0 - f1 - f3 * z1;
      M[2][i] = f3 * z0 + f4 + t1;
    }

#pragma unroll
    for (int i = 0; i < 3; i++) {
      auto f0 = M[i][0];
      auto f1 = M[i][1];
      auto f2 = M[i][2];
      auto f3 = M[i][3];
      auto f4 = M[i][4];

      auto t0 = f2 * z0;
      auto t1 = t0 + f1;

      auto p0 = f3 * z2 + f0 + t1;
      auto p1 = t0 - f1 - f3 * z1;
      auto p2 = f3 * z0 + f4 + t1;

      if (fuse_bias) {
        p0 += BIAS;
        p1 += BIAS;
        p2 += BIAS;
      }
      if (std::is_same<OutputType, uint8_t>::value
          || std::is_same<OutputType, int8_t>::value) {
        p0 = p0 * mrepS + mzp;
        p1 = p1 * mrepS + mzp;
        p2 = p2 * mrepS + mzp;
      }
      if (fuse_ip_sum) {
        if (std::is_same<OutputType, uint8_t>::value) {
          p0 += _cvtepu8_ps(out_ptr(i, 0));
          p1 += _cvtepu8_ps(out_ptr(i, 1));
          p2 += _cvtepu8_ps(out_ptr(i, 2));
        } else if (std::is_same<OutputType, int8_t>::value) {
          p0 += _cvtepi8_ps(out_ptr(i, 0));
          p1 += _cvtepi8_ps(out_ptr(i, 1));
          p2 += _cvtepi8_ps(out_ptr(i, 2));
        } else {
          p0 += *(__m<V> *)out_ptr(i, 0);
          p1 += *(__m<V> *)out_ptr(i, 1);
          p2 += *(__m<V> *)out_ptr(i, 2);
        }
      }
      if (fuse_relu) {
        p0 = _mm<V>::max_ps(p0, z);
        p1 = _mm<V>::max_ps(p1, z);
        p2 = _mm<V>::max_ps(p2, z);
      }
      STORE(i, 0)
      STORE(i, 1)
      STORE(i, 2)
    }
  }
}; // elk_conv_wino_trans_output

} // namespace euler
