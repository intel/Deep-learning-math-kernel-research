#pragma once
#include <x86intrin.h>
#include <cassert>

#define rounding_case(func, ret_type, imm8, var)                               \
  switch (imm8) {                                                              \
  case _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC:                          \
    return func(var, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
  case _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC:                              \
    return func(var, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);               \
  case _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC:                              \
    return func(var, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);               \
  case _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC:                                 \
    return func(var, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);                  \
  case _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC:                           \
    return func(var, _MM_FROUND_CUR_DIRECTION);                                \
  default:                                                                     \
    assert(0);                                                                 \
    return func(var, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);           \
  };

template <int V> struct _mm_traits {
  typedef void vector_type;
  typedef void vector_itype;
};

template <int V> struct _mm {
};

#ifdef __AVX512F__
template <> struct _mm_traits<16> {
  typedef __m512 vector_type;
  typedef __m512i vector_itype;
};
#endif

#ifdef __AVX2__
template <> struct _mm_traits<8> {
  typedef __m256 vector_type;
  typedef __m256i vector_itype;
};
#endif

template <> struct _mm_traits<4> {
  typedef __m128 vector_type;
  typedef __m128i vector_itype;
};

template <int V> using __m = typename _mm_traits<V>::vector_type;
template <int V> using __i = typename _mm_traits<V>::vector_itype;

#ifdef __AVX512F__
#if 1
template <> struct _mm<16> {
  static constexpr int V = 16;
  static inline __m<V> load_ps(void const *adrs) noexcept {
    return _mm512_load_ps(adrs);
  }
  static inline __m<V> loadu_ps(void const *adrs) noexcept {
    return _mm512_loadu_ps(adrs);
  }
  static inline void store_ps(void *adrs, __m<V> m) noexcept {
    _mm512_store_ps(adrs, m);
  }
  static inline void mask_store_ps(void *adrs, __mmask16 k, __m<V> m) noexcept {
    _mm512_mask_store_ps(adrs, k, m);
  }
  static inline void stream_ps(void *adrs, __m<V> m) noexcept {
    _mm512_stream_ps((float *)adrs, m);
  }
  static inline __m<V> maskz_load_ps(__mmask16 k, void const *adrs) noexcept {
    return _mm512_maskz_load_ps(k, adrs);
  }
  static inline __i<V> load_si512(void const *a) noexcept {
    return _mm512_load_si512(a);
  }
  static inline void store_si512(void *a, __i<V> b) noexcept {
    return _mm512_store_si512(a, b);
  }
  static inline __i<V> stream_load_si512(void *a) noexcept {
    return _mm512_stream_load_si512(a);
  }
  static inline void stream_si512(__i<V> *a, __i<V> b) noexcept {
    return _mm512_stream_si512(a, b);
  }
  static inline __i<V> maskz_load_epi32(__mmask16 k, void const *adrs) noexcept {
    return _mm512_maskz_load_epi32(k, adrs);
  }
  static inline void i32scatter_ps(void *adrs, __i<V> vidx,
      __m<V> m, int scale) noexcept {
    switch (scale) {
      case 1:
        _mm512_i32scatter_ps(adrs, vidx, m, 1);
        break;
      case 2:
        _mm512_i32scatter_ps(adrs, vidx, m, 2);
        break;
      case 4:
        _mm512_i32scatter_ps(adrs, vidx, m, 4);
        break;
      case 8:
        _mm512_i32scatter_ps(adrs, vidx, m, 8);
        break;
    }
  }
  static inline __m<V> i32gather_ps(__i<V> vidx, void *adrs, int scale)
  noexcept {
    switch (scale) {
      case 1:
        return _mm512_i32gather_ps(vidx, adrs, 1);
      case 2:
        return _mm512_i32gather_ps(vidx, adrs, 2);
      case 4:
        return _mm512_i32gather_ps(vidx, adrs, 4);
      case 8:
        return _mm512_i32gather_ps(vidx, adrs, 8);
    }

    return _mm512_i32gather_ps(vidx, adrs, 1);
  }
  static inline __i<V> i32gather_epi32(__i<V> vidx, void *adrs, int scale)
  noexcept {
    switch (scale) {
      case 1:
        return _mm512_i32gather_epi32(vidx, adrs, 1);
      case 2:
        return _mm512_i32gather_epi32(vidx, adrs, 2);
      case 4:
        return _mm512_i32gather_epi32(vidx, adrs, 4);
      case 8:
        return _mm512_i32gather_epi32(vidx, adrs, 8);
    }

    return _mm512_i32gather_epi32(vidx, adrs, 1);
  }
  static inline __i<V> mask_i32gather_epi32(__i<V> src, __mmask16 k,
      __i<V> vidx, void *adrs, int scale) noexcept{
    switch (scale) {
    case 1:
      return _mm512_mask_i32gather_epi32(src, k, vidx, adrs, 1);
    case 2:
      return _mm512_mask_i32gather_epi32(src, k, vidx, adrs, 2);
    case 4:
      return _mm512_mask_i32gather_epi32(src, k, vidx, adrs, 4);
    case 8:
      return _mm512_mask_i32gather_epi32(src, k, vidx, adrs, 8);
    }

    return _mm512_mask_i32gather_epi32(src, k, vidx, adrs, 1);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm512_setzero_ps();
  }
  static inline __i<V> set_epi32(int e15, int e14, int e13, int e12,
      int e11, int e10, int e9, int e8,
      int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm512_set_epi32(e15, e14, e13, e12, e11, e10, e9, e8,
        e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> set1_ps(float e) noexcept {
    return _mm512_set1_ps(e);
  }
  static inline __m<V> set_ps(float e15, float e14, float e13, float e12,
      float e11, float e10, float e9, float e8, float e7, float e6, float e5,
      float e4, float e3, float e2, float e1, float e0) noexcept {
    return _mm512_set_ps(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4,
        e3, e2, e1, e0);
  }
  static inline __m<V> add_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_add_ps(op1, op2);
  }
  static inline __m<V> sub_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_sub_ps(op1, op2);
  }
  static inline __m<V> mul_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_mul_ps(op1, op2);
  }
  static inline __m<V> div_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_div_ps(op1, op2);
  }
  static inline __m<V> sqrt_ps(__m<V> op1) noexcept {
    return _mm512_sqrt_ps(op1);
  }
  static inline __m<V> fmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fmsub_ps(op1, op2, op3);
  }
  static inline __m<V> fnmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fnmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fnmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm512_fnmsub_ps(op1, op2, op3);
  }
  static inline __m<V> max_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_max_ps(op1, op2);
  }
  static inline __m<V> min_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_min_ps(op1, op2);
  }
  static inline float reduce_max_ps(__m<V> op1) noexcept {
    return _mm512_reduce_max_ps(op1);
  }
  static inline float reduce_min_ps(__m<V> op1) noexcept {
    return _mm512_reduce_min_ps(op1);
  }
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_xor_ps(op1, op2);
  }
  static inline __i<V> and_epi32(__i<V> op1, __i<V> op2) noexcept {
    return _mm512_and_epi32(op1, op2);
  }
  static inline __i<V> or_epi32(__i<V> op1, __i<V> op2) noexcept {
    return _mm512_or_epi32(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm512_broadcastss_ps(b);
  }
  static inline __i<V> setzero_epi32(void) noexcept {
    return _mm512_setzero_epi32();
  }
  static inline __i<V> load_epi32(void const *adrs) noexcept {
    return _mm512_load_epi32(adrs);
  }
  static inline void store_epi32(void *adrs, __i<V> m) noexcept {
    return _mm512_store_epi32(adrs, m);
  }
  static inline __i<V> set1_epi32(int e) noexcept {
    return _mm512_set1_epi32(e);
  }
  static inline __i<V> set1_epi16(short e) noexcept {
    return _mm512_set1_epi16(e);
  }
  static inline __i<V> adds_epi16(__i<V> m0, __i<V> m1) noexcept {
    return _mm512_adds_epi16(m0, m1);
  }
  static inline __i<V> maddubs_epi16(__i<V> m0, __i<V> m1) noexcept {
    return _mm512_maddubs_epi16(m0, m1);
  }
  static inline __i<V> madd_epi16(__i<V> m0, __i<V> m1) noexcept {
    return _mm512_madd_epi16(m0, m1);
  }
  static inline __i<V> add_epi32(__i<V> m0, __i<V> m1) noexcept {
    return _mm512_add_epi32(m0, m1);
  }
  static inline __m<V> cvtepi32_ps(__i<V> m) noexcept {
    return _mm512_cvtepi32_ps(m);
  }
  static inline __i<V> cvtepu8_epi32(__m128i m) noexcept {
    return _mm512_cvtepu8_epi32(m);
  }
  static inline __i<V> cvtepi8_epi32(__m128i m) noexcept {
    return _mm512_cvtepi8_epi32(m);
  }
  static inline __i<V> cvtps_epu32(__m<V> m) noexcept {
    return _mm512_cvtps_epu32(m);
  }
  static inline __i<V> cvt_roundps_epu32(__m<V> m, int imm8) noexcept {
    rounding_case(_mm512_cvt_roundps_epu32, __i<V>, imm8, m);
  }
  static inline __i<V> cvt_roundps_epi32(__m<V> m, int imm8) noexcept {
    rounding_case(_mm512_cvt_roundps_epi32, __i<V>, imm8, m);
  }
  static inline __m128i cvtusepi32_epi8(__i<V> m) noexcept {
    return _mm512_cvtusepi32_epi8(m);
  }
  static inline __m128i cvtsepi32_epi8(__i<V> m) noexcept {
    return _mm512_cvtsepi32_epi8(m);
  }
  static inline __m256i cvtps_ph(__m<V> a, int rounding) noexcept {
    rounding_case(_mm512_cvtps_ph, __m256i, rounding, a);
  }
  static inline __m<V> cvtph_ps(__m256i a) noexcept {
    return _mm512_cvtph_ps(a);
  }
  static inline __m<V> roundscale_ps(__m<V> m, int imm8) noexcept {
    rounding_case(_mm512_roundscale_ps, __m<V>, imm8, m);
  }
  static inline __m<V> range_ps(__m<V> m0, __m<V> m1, int imm8) noexcept {
    return _mm512_range_ps(m0, m1, imm8);
  }
  static inline __m512i cvtepi16_epi32(__m256i a) noexcept {
    return _mm512_cvtepi16_epi32(a);
  }
  static inline __m256i cvtepi32_epi16(__m512i a) noexcept {
    return _mm512_cvtepi32_epi16(a);
  }
  static inline __i<V> bsrli_epi128(__i<V> x, int imm8) {
  #undef imm8_case
  #define imm8_case(val) \
    case val: return _mm512_bsrli_epi128(x, val);

    switch (imm8) {
    imm8_case(0)
    imm8_case(1)
    imm8_case(2)
    imm8_case(3)
    imm8_case(4)
    imm8_case(5)
    imm8_case(6)
    imm8_case(7)
    imm8_case(8)
    imm8_case(9)
    imm8_case(10)
    imm8_case(11)
    imm8_case(12)
    imm8_case(13)
    imm8_case(14)
    imm8_case(15)
    default:
      return _mm512_bsrli_epi128(x, 15);
    };
  }
  static inline __i<V/2> cvt_f32_b16(__i<V> x) {
    x = _mm512_bsrli_epi128(x, 2);
    return _mm512_cvtepi32_epi16(x);
  }
};
#else
/* ICC Bug! */
template <> struct _mm<16> {
  static constexpr auto load_ps = _mm512_load_ps;
  static constexpr auto loadu_ps = _mm512_loadu_ps;
  static constexpr auto store_ps = _mm512_store_ps;
  static constexpr auto mask_store_ps = _mm512_mask_store_ps;
  static constexpr auto setzero_ps = _mm512_setzero_ps;
  static constexpr auto set1_ps = _mm512_set1_ps;
  static constexpr auto add_ps = _mm512_add_ps;
  static constexpr auto sub_ps = _mm512_sub_ps;
  static constexpr auto mul_ps = _mm512_mul_ps;
  static constexpr auto fmadd_ps = _mm512_fmadd_ps;
  static constexpr auto fmsub_ps = _mm512_fmsub_ps;
  static constexpr auto fnmadd_ps = _mm512_fnmadd_ps;
  static constexpr auto fnmsub_ps = _mm512_fnmsub_ps;
  static constexpr auto max_ps = _mm512_max_ps;
  static constexpr auto xor_ps = _mm512_xor_ps;
  static constexpr auto broadcastss_ps = _mm512_broadcastss_ps;
};
#endif
#endif

#ifdef __AVX2__
#if 1
template <> struct _mm<8> {
  static constexpr int V = 8;
  static inline __m<V> load_ps(float const *adrs) noexcept {
    return _mm256_load_ps(adrs);
  }
  static inline __m<V> loadu_ps(float const *adrs) noexcept {
    return _mm256_loadu_ps(adrs);
  }
  static inline void store_ps(float *adrs, __m<V> m) noexcept {
    _mm256_store_ps(adrs, m);
  }
  static inline void mask_store_ps(float *adrs, __mmask8 k, __m<V> m) noexcept {
    _mm256_mask_store_ps(adrs, k, m);
  }
  static inline void stream_ps(float *adrs, __m<V> m) noexcept {
    _mm256_stream_ps(adrs, m);
  }
  static inline void i32scatter_ps(void *adrs, __i<V> vidx,
      __m<V> m, int scale) noexcept {
    switch(scale) {
      case 1:
        _mm256_i32scatter_ps(adrs, vidx, m, 1);
      case 2:
        _mm256_i32scatter_ps(adrs, vidx, m, 2);
      case 4:
        _mm256_i32scatter_ps(adrs, vidx, m, 4);
      case 8:
        _mm256_i32scatter_ps(adrs, vidx, m, 8);
    }
  }
  static inline __m<V> i32gather_ps(__i<V> vidx, const float *adrs, int scale)
  noexcept {
    switch (scale) {
      case 1:
        return _mm256_i32gather_ps(adrs, vidx, 1);
      case 2:
        return _mm256_i32gather_ps(adrs, vidx, 2);
      case 4:
        return _mm256_i32gather_ps(adrs, vidx, 4);
      case 8:
        return _mm256_i32gather_ps(adrs, vidx, 8);
    }

    return _mm256_i32gather_ps(adrs, vidx, 1);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm256_setzero_ps();
  }
  static inline __m<V> set1_ps(float e) noexcept {
    return _mm256_set1_ps(e);
  }
  static inline __i<V> set_epi32(int, int, int, int, int, int, int, int,
      int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __i<V> set_epi32(int e7, int e6, int e5, int e4,
      int e3, int e2, int e1, int e0) noexcept {
    return _mm256_set_epi32(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> set_ps(float e7, float e6, float e5,
      float e4, float e3, float e2, float e1, float e0) noexcept {
    return _mm256_set_ps(e7, e6, e5, e4, e3, e2, e1, e0);
  }
  static inline __m<V> add_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_add_ps(op1, op2);
  }
  static inline __m<V> sub_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_sub_ps(op1, op2);
  }
  static inline __m<V> mul_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_mul_ps(op1, op2);
  }
  static inline __m<V> div_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_div_ps(op1, op2);
  }
  static inline __m<V> sqrt_ps(__m<V> op1) noexcept {
    return _mm256_sqrt_ps(op1);
  }
  static inline __m<V> fmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fmsub_ps(op1, op2, op3);
  }
  static inline __m<V> fnmadd_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fnmadd_ps(op1, op2, op3);
  }
  static inline __m<V> fnmsub_ps(__m<V> op1, __m<V> op2, __m<V> op3) noexcept {
    return _mm256_fnmsub_ps(op1, op2, op3);
  }
  static inline __m<V> max_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_max_ps(op1, op2);
  }
  static inline __m<V> min_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_min_ps(op1, op2);
  }
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_xor_ps(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm256_broadcastss_ps(b);
  }
  static inline __i<V> cvtps_epu32(__m<V> m) noexcept {
    return _mm256_cvtps_epu32(m);
  }
  // depends on rounding mode
  static inline __i<V> cvt_roundps_epu32(__m<V> m, int imm8) noexcept {
    return _mm256_cvtps_epu32(m);
  }
  static inline __i<V> cvt_roundps_epi32(__m<V> m, int imm8) noexcept {
    return _mm256_cvtps_epi32(m);
  }
  static inline __m128i cvtusepi32_epi8(__i<V> m) noexcept {
    return _mm256_cvtusepi32_epi8(m);
  }
  static inline __m128i cvtsepi32_epi8(__i<V> m) noexcept {
    return _mm256_cvtsepi32_epi8(m);
  }
  static inline __m<V> range_ps(__m<V> m0, __m<V> m1, int imm8) noexcept {
    return _mm256_range_ps(m0, m1, imm8);
  }
  static inline __m256i load_si256(__m256i const *a) noexcept {
    return _mm256_load_si256(a);
  }
  static inline void store_si256(__m256i *a, __m256i b) noexcept {
    return _mm256_store_si256(a, b);
  }
  static inline void stream_si256(__m256i *a, __m256i b) noexcept {
    return _mm256_stream_si256(a, b);
  }
};
#else
/* ICC Bug! */
template <> struct _mm<8> {
  static constexpr auto load_ps = _mm256_load_ps;
  static constexpr auto store_ps = _mm256_store_ps;
  static constexpr auto setzero_ps = _mm256_setzero_ps;
  static constexpr auto set1_ps = _mm256_set1_ps;
  static constexpr auto add_ps = _mm256_add_ps;
  static constexpr auto sub_ps = _mm256_sub_ps;
  static constexpr auto mul_ps = _mm256_mul_ps;
  static constexpr auto fmadd_ps = _mm256_fmadd_ps;
  static constexpr auto fmsub_ps = _mm256_fmsub_ps;
  static constexpr auto fnmadd_ps = _mm256_fnmadd_ps;
  static constexpr auto fnmsub_ps = _mm256_fnmsub_ps;
  static constexpr auto max_ps = _mm256_max_ps;
  static constexpr auto xor_ps = _mm256_xor_ps;
  static constexpr auto broadcastss_ps = _mm256_broadcastss_ps;
};
#endif
#endif
template <int V> inline __m<V> operator +(__m<V> op1, __m<V> op2) noexcept {
  return _mm<V>::add_ps(op1, op2);
}
template <int V> inline __m<V> operator -(__m<V> op1, __m<V> op2) noexcept {
  return _mm<V>::sub_ps(op1, op2);
}
template <int V> inline __m<V> operator *(__m<V> op1, __m<V> op2) noexcept {
  return _mm<V>::mul_ps(op1, op2);
}
template <int V> inline __m<V> operator ^(__m<V> op1, __m<V> op2) noexcept {
  return _mm<V>::xor_ps(op1, op2);
}
