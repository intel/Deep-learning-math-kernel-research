#pragma once
#include <x86intrin.h>

template <int V> struct _mm_traits {
  typedef void vector_type;
};

template <int V> struct _mm {
};

#ifdef __AVX512F__
template <> struct _mm_traits<16> {
  typedef __m512 vector_type;
};
#endif

#ifdef __AVX2__
template <> struct _mm_traits<8> {
  typedef __m256 vector_type;
};
#endif

template <> struct _mm_traits<4> {
  typedef __m128 vector_type;
};

template <int V> using __m = typename _mm_traits<V>::vector_type;

#ifdef __AVX512F__
#if 1
template <> struct _mm<16> {
  static constexpr int V = 16;
  static inline __m<V> load_ps(void const *adrs) noexcept {
    return _mm512_load_ps(adrs);
  }
  static inline void store_ps(void *adrs, __m<V> m) noexcept {
    _mm512_store_ps(adrs, m);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm512_setzero_ps();
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
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm512_xor_ps(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm512_broadcastss_ps(b);
  }
};
#else
/* ICC Bug! */
template <> struct _mm<16> {
  static constexpr auto load_ps = _mm512_load_ps;
  static constexpr auto store_ps = _mm512_store_ps;
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
  static inline void store_ps(float *adrs, __m<V> m) noexcept {
    _mm256_store_ps(adrs, m);
  }
  static inline __m<V> setzero_ps(void) noexcept {
    return _mm256_setzero_ps();
  }
  static inline __m<V> set1_ps(float e) noexcept {
    return _mm256_set1_ps(e);
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
  static inline __m<V> xor_ps(__m<V> op1, __m<V> op2) noexcept {
    return _mm256_xor_ps(op1, op2);
  }
  static inline __m<V> broadcastss_ps(__m128 b) noexcept {
    return _mm256_broadcastss_ps(b);
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
