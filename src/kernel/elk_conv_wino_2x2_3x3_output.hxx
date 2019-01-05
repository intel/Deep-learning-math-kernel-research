#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"
#include "elk_conv_wino_2x2_3x3_input.hxx"

namespace euler {

//   int _hOA_end, int _wOA_end
template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 4, 3>::__trans_output(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A][V], BiasType *bias,
    int hOA_end, int wOA_end)
{

  ENABLE_AVX512F();
  constexpr bool is_border = cd_traits<conditions...>::is_border;
  constexpr bool with_bias = cd_traits<conditions...>::with_bias;
  constexpr bool with_relu = cd_traits<conditions...>::with_relu;
  constexpr bool with_ip_sum = cd_traits<conditions...>::with_ip_sum;
  bool fuse_ip_sum = with_ip_sum && (wOA_end != -1);
  bool fuse_bias = with_bias && (bias != nullptr);
  bool fuse_relu = with_relu && (bias != nullptr);

  alignas(64) OutputType dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(OutputType, aoutput, output, A - K + 1, A - K + 1, V);
      return &md3(aoutput, _h, _w, 0);
    } else {
      MD3(OutputType, aoutput, output, xc.oh, xc.ow, V);
      if (is_border && (_h > hOA_end || _w > wOA_end))
        return dummy;
      else
        return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#undef OP
#undef FUSE_BIAS
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m<V> c0, c1, c2, c3, zero;
#define t(m, n) t##m##n
#define OP(m,n) __m<V> t(m,n) = _mm<V>::load_ps(T(m, n))
#define FUSE_BIAS(p)                                                           \
  if (std::is_same<BiasType, float>::value) {                                  \
    p = ADD(p, *(__m<V>*)bias);                                                \
  } else {                                                                     \
    auto f16v = _mm<V/2>::load_si256((__m256i *)bias);                         \
    p = ADD(p, _mm<V>::cvtph_ps(f16v));                                        \
  }

  MATRIX_DEF(4, 4);

  c0 = ADD(ADD(t10, t11), t12);
  c1 = ADD(ADD(t20, t21), t22);
  c2 = SUB(ADD(t12, t13), t11);
  c3 = SUB(ADD(t22, t23), t21);

  __m<V> p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
  if (fuse_bias) {FUSE_BIAS(p00)}
  if (fuse_ip_sum) p00 = ADD(p00, *(__m<V>*)P(0, 0));
  if (fuse_relu) {
    zero = XOR(zero, zero);
    p00 = MAX(p00, zero);
  }
  __m<V> p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
  if (fuse_bias) {FUSE_BIAS(p10)}
  if (fuse_ip_sum) p10 = ADD(p10, *(__m<V>*)P(1, 0));
  if (fuse_relu) p10 = MAX(p10, zero);
  __m<V> p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
  if (fuse_bias) {FUSE_BIAS(p01)}
  if (fuse_ip_sum) p01 = ADD(p01, *(__m<V>*)P(0, 1));
  if (fuse_relu) p01 = MAX(p01, zero);
  __m<V> p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);
  if (fuse_bias) {FUSE_BIAS(p11)}
  if (fuse_ip_sum) p11 = ADD(p11, *(__m<V>*)P(1, 1));
  if (fuse_relu) p11 = MAX(p11, zero);

#undef OP
#define p_(m, n) p##m##n
#define OP(m,n)                                         \
  if (std::is_same<OutputType, float>::value)           \
    _mm<V>::store_ps(P(m, n), p_(m, n));                \
  else {                                                \
    auto f16 = _mm<V>::cvtps_ph(p_(m, n),               \
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); \
    _mm<V/2>::store_si256((__m256i *)P(m, n), f16);     \
  }
  MATRIX_DEF(2, 2);
}


} // namespace euler
