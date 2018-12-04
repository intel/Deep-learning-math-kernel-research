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
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m<V> c0, c1, c2, c3, zero;
#define t(m, n) t##m##n
#define OP(m,n) __m<V> t(m,n) = _mm<V>::load_ps(T(m, n))
  MATRIX_DEF(4, 4);

  c0 = ADD(ADD(t10, t11), t12);
  c1 = ADD(ADD(t20, t21), t22);
  c2 = SUB(ADD(t12, t13), t11);
  c3 = SUB(ADD(t22, t23), t21);

  __m<V> p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
  if (fuse_bias) p00 = ADD(p00, *(__m<V>*)bias);
  if (fuse_ip_sum) p00 = ADD(p00, *(__m<V>*)P(0, 0));
  if (fuse_relu) {
    zero = XOR(zero, zero);
    p00 = MAX(p00, zero);
  }
  __m<V> p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
  if (fuse_bias) p10 = ADD(p10, *(__m<V>*)bias);
  if (fuse_ip_sum) p10 = ADD(p10, *(__m<V>*)P(1, 0));
  if (fuse_relu) p10 = MAX(p10, zero);
  __m<V> p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
  if (fuse_bias) p01 = ADD(p01, *(__m<V>*)bias);
  if (fuse_ip_sum) p01 = ADD(p01, *(__m<V>*)P(0, 1));
  if (fuse_relu) p01 = MAX(p01, zero);
  __m<V> p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);
  if (fuse_bias) p11 = ADD(p11, *(__m<V>*)bias);
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

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 4, 3>::__trans_outputa_th(elx_conv_t<UserTypes> &xc,
    TrOpType *toutputa, TrOpType *toutput, int Tz, bool stream_out)
{
  ENABLE_AVX512F();

  MD4(TrOpType, atoutput, toutput, A, xc.oc3 * xc.O2, Tz, V);
  MD2(TrOpType, atoutputa, toutputa, A - K + 1, V);

#undef P
#undef T
#define T(_h) &md4(atoutput, _h, 0, 0, 0)
#define P(_h) &md2(atoutputa, _h, 0)

  __m<V> t0 = _mm<V>::load_ps(T(0));
  __m<V> t1 = _mm<V>::load_ps(T(1));
  __m<V> t2 = _mm<V>::load_ps(T(2));
  __m<V> t3 = _mm<V>::load_ps(T(3));

  __m<V> p0 = ADD(ADD(t0, t1), t2);
  __m<V> p1 = SUB(ADD(t2, t3), t1);

  if (stream_out) {
    _mm<V>::stream_ps(P(1), p0);
    _mm<V>::stream_ps(P(1), p1);
  } else {
    _mm<V>::store_ps(P(0), p0);
    _mm<V>::store_ps(P(1), p1);
  }
}

template <typename UserTypes, typename TrOpType, int V>
template <bool... conditions>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 4, 3>::__trans_outputa_bh(elx_conv_t<UserTypes> &xc,
    OutputType *output, TrOpType atoutput[A][A - K + 1][V], BiasType *bias,
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

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    if (wOA_end == -1) {
      MD3(float, aoutput, output, A - K + 1, A - K + 1, 16);
      return &md3(aoutput,_h,_w, 0);
    } else {
    MD3(float, aoutput, output, xc.oh, xc.ow, 16);
    if (is_border && (_h > hOA_end || _w > wOA_end))
      return dummy;
    else
      return &md3(aoutput, _h, _w, 0);
    }
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

 // __m<V> c0, c1, c2, c3;

  __m<V> t0, t1, t2, t3, p0, p1, zero;
  t0 = _mm<V>::load_ps(T(0,0));
  t1 = _mm<V>::load_ps(T(0,1));
  t2 = _mm<V>::load_ps(T(0,2));
  t3 = _mm<V>::load_ps(T(0,3));

  p0 = ADD(ADD(t0, t1), t2);
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(0, 0));
  if (fuse_relu) {
    zero = XOR(zero, zero);
    p0 = MAX(p0, zero);
  }
  p1 = SUB(ADD(t2, t3), t1);
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(0, 1));
  if (fuse_relu) p1 = MAX(p1, zero);

  _mm<V>::store_ps(P(0,0), p0);
  _mm<V>::store_ps(P(0,1), p1);

  t0 = _mm<V>::load_ps(T(1,0));
  t1 = _mm<V>::load_ps(T(1,1));
  t2 = _mm<V>::load_ps(T(1,2));
  t3 = _mm<V>::load_ps(T(1,3));

  p0 = ADD(ADD(t0, t1), t2);
  if (fuse_bias) p0 = ADD(p0, *(__m<V>*)bias);
  if (fuse_ip_sum) p0 = ADD(p0, *(__m<V>*)P(1, 0));
  if (fuse_relu) p0 = MAX(p0, zero);
  p1 = SUB(ADD(t2, t3), t1);
  if (fuse_bias) p1 = ADD(p1, *(__m<V>*)bias);
  if (fuse_ip_sum) p1 = ADD(p1, *(__m<V>*)P(1, 1));
  if (fuse_relu) p1 = MAX(p1, zero);

  _mm<V>::store_ps(P(1,0), p0);
  _mm<V>::store_ps(P(1,1), p1);

}

} // namespace euler
