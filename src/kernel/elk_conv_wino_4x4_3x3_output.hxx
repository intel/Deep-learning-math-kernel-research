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
struct elk_conv_wino_trans_output<float,OutputType, BiasType, format,
    is_border, with_bias, with_relu, with_ip_sum, ISA_AVX512, 6, 3, V> {
  constexpr static int A = 6;
  constexpr static int K = 3;

  static void execute(elx_param_t &ep, OutputType *output,
      float *toutput, BiasType *bias, int hOA_end, int wOA_end)
  {
    __m<V> mrepS, mzp;

    MD3(float, atoutput, toutput, A, A, V);
    if (std::is_same<OutputType, uint8_t>::value
        || std::is_same<OutputType, int8_t>::value) {
      mrepS = _mm<V>::set1_ps(ep.output_quant_repS);
      mzp = _mm<V>::set1_ps(ep.output_quant_z);
    }

    bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
    // TODO replace bias != nullptr with last_I4 condition
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

#undef STORE
#undef BIAS
#undef STORE

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

#define STORE_PS(mem, reg)                                                     \
  if (ep.streaming_output) {                                                   \
    _mm<V>::stream_ps(mem, reg);                                               \
  } else {                                                                     \
    _mm<V>::store_ps(mem, reg);                                                \
  }
#define STORE_SI128(mem, reg)                                                  \
  if (ep.streaming_output) {                                                   \
    _mm_stream_si128(mem, reg);                                                \
  } else {                                                                     \
    _mm_store_si128(mem, reg);                                                 \
  }
#define STORE_SI256(mem, reg)                                                  \
  if (ep.streaming_output) {                                                   \
    _mm<V/2>::stream_si256(mem, reg);                                          \
  } else {                                                                     \
    _mm<V/2>::store_si256(mem, reg);                                           \
  }

#define STORE(i, j)                                                            \
  if (std::is_same<OutputType, float>::value) {                                \
    STORE_PS(out_ptr(i, j), p##j);                                             \
  } else if (std::is_same<OutputType, uint8_t>::value) {                       \
    __i<V> mresu32 = _mm<V>::cvt_roundps_epu32(                                \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    __m128i mresu8 = _mm<V>::cvtusepi32_epi8(mresu32);                         \
    STORE_SI128((__m128i *)out_ptr(i, j), mresu8);                             \
  } else if (std::is_same<OutputType, int8_t>::value) {                        \
    __i<V> mresi32 = _mm<V>::cvt_roundps_epi32(                                \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    __m128i mresi8 = _mm<V>::cvtsepi32_epi8(mresi32);                          \
    STORE_SI128((__m128i *)out_ptr(i, j), mresi8);                             \
  } else {                                                                     \
    auto f16 = _mm<V>::cvtps_ph(                                               \
        p##j, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);                  \
    STORE_SI256((__m256i *)out_ptr(i, j), f16);                                \
  }

    alignas(64) float M[4][6][16];

    __m<V> z0 = _mm<V>::set1_ps(1.5f);
    __m<V> z1 = _mm<V>::set1_ps(2.25f);
    __m<V> z2 = _mm<V>::set1_ps(3.375f);
    __m<V> z3 = _mm<V>::set1_ps(0.625f);
    __m<V> z4 = _mm<V>::set1_ps(0.390625f);
    __m<V> z5 = _mm<V>::set1_ps(0.244140625f);
    __m<V> z = _mm<V>::xor_ps(z, z);

#pragma unroll
    for (int i = 0; i < 6; i++) {
      auto f0 = _mm<V>::load_ps(&md3(atoutput, 0, i, 0));
      auto f1 = _mm<V>::load_ps(&md3(atoutput, 1, i, 0));
      auto f2 = _mm<V>::load_ps(&md3(atoutput, 2, i, 0));
      auto f3 = _mm<V>::load_ps(&md3(atoutput, 3, i, 0));
      auto f4 = _mm<V>::load_ps(&md3(atoutput, 4, i, 0));
      auto f5 = _mm<V>::load_ps(&md3(atoutput, 5, i, 0));
      auto t0 = f1 + f2;
      auto t1 = f3 + f4;
      auto t2 = f1 - f2;
      auto t3 = f3 - f4;

      *(__m<V> *)M[0][i] = t0 + t1 + f0;
      *(__m<V> *)M[1][i] = t2 * z3 + t3 * z0;
      *(__m<V> *)M[2][i] = t0 * z4 + t1 * z1;
      *(__m<V> *)M[3][i] = t2 * z5 + t3 * z2 + f5;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      auto f0 = _mm<V>::load_ps(M[i][0]);
      auto f1 = _mm<V>::load_ps(M[i][1]);
      auto f2 = _mm<V>::load_ps(M[i][2]);
      auto f3 = _mm<V>::load_ps(M[i][3]);
      auto f4 = _mm<V>::load_ps(M[i][4]);
      auto f5 = _mm<V>::load_ps(M[i][5]);
      auto t0 = f1 + f2;
      auto t1 = f3 + f4;
      auto t2 = f1 - f2;
      auto t3 = f3 - f4;

      auto p0 = t0 + t1 + f0;
      auto p1 = t2 * z3 + t3 * z0;
      auto p2 = t0 * z4 + t1 * z1;
      auto p3 = t2 * z5 + t3 * z2 + f5;

      if (fuse_bias) {
        p0 += BIAS;
        p1 += BIAS;
        p2 += BIAS;
        p3 += BIAS;
      }
      if (std::is_same<OutputType, uint8_t>::value
          || std::is_same<OutputType, int8_t>::value) {
        p0 = p0 * mrepS + mzp;
        p1 = p1 * mrepS + mzp;
        p2 = p2 * mrepS + mzp;
        p3 = p3 * mrepS + mzp;
      }
      if (fuse_ip_sum) {
        if (std::is_same<OutputType, uint8_t>::value) {
          p0 += _cvtepu8_ps(out_ptr(i, 0));
          p1 += _cvtepu8_ps(out_ptr(i, 1));
          p2 += _cvtepu8_ps(out_ptr(i, 2));
          p3 += _cvtepu8_ps(out_ptr(i, 3));
        } else if (std::is_same<OutputType, int8_t>::value) {
          p0 += _cvtepi8_ps(out_ptr(i, 0));
          p1 += _cvtepi8_ps(out_ptr(i, 1));
          p2 += _cvtepi8_ps(out_ptr(i, 2));
          p3 += _cvtepi8_ps(out_ptr(i, 3));
        } else {
          p0 += *(__m<V> *)out_ptr(i, 0);
          p1 += *(__m<V> *)out_ptr(i, 1);
          p2 += *(__m<V> *)out_ptr(i, 2);
          p3 += *(__m<V> *)out_ptr(i, 3);
        }
      }
      if (fuse_relu) {
        p0 = _mm<V>::max_ps(p0, z);
        p1 = _mm<V>::max_ps(p1, z);
        p2 = _mm<V>::max_ps(p2, z);
        p3 = _mm<V>::max_ps(p3, z);
      }
      STORE(i, 0)
      STORE(i, 1)
      STORE(i, 2)
      STORE(i, 3)
    }
  }
}; // elk_conv_wino_trans_output

} // namespace euler
