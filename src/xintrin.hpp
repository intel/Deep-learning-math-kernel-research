#include <x86intrin.h>

template <int V> struct _mm_traits {
  typedef void vector_type;
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

template <int V> struct _mm {
};

#ifdef __AVX512F__
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

#ifdef __AVX2__
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
