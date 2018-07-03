#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

#undef ADD
#undef SUB
#undef FMADD
#undef FMSUB
#undef MUL
#define ADD _mm512_add_ps
#define SUB _mm512_sub_ps
#define MUL _mm512_mul_ps
#define FMADD _mm512_fmadd_ps
#define FMSUB _mm512_fmsub_ps

namespace euler {

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::trans_output(elx_conv_t<Type> &xc, Type *output,
    Type atoutput[A][A][V], Type *bias, int _hOA_end, int _wOA_end)
{
  __trans_output(winograd_template_parameter_t<R_OUTPUT(
                     Type, A, K, V, I, is_border, with_bias)>(),
      xc, output, atoutput, bias, _hOA_end, _wOA_end);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::trans_outputa_th(elx_conv_t<Type> &xc, Type *toutputa,
    Type *toutput, int Tz)
{
  __trans_outputa_th(winograd_template_parameter_t<R_OUTPUT(
                     Type, A, K, V, I, is_border, with_bias)>(),
      xc, toutputa, toutput, Tz);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::trans_outputa_bh(elx_conv_t<Type> &xc, Type *output,
    Type atoutputa[A][A - K + 1][V], Type *bias, int _hOA_end, int _wOA_end)
{
  __trans_outputa_bh(winograd_template_parameter_t<R_OUTPUT(
                     Type, A, K, V, I, is_border, with_bias)>(),
      xc, output, atoutputa, bias, _hOA_end, _wOA_end);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 4,
                                    3, 16, ISA_GENERIC, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    MD(float, aoutput, [xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return &dummy[_V];
    else
      return &aoutput[_h][_w][_V];
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]
  float c0[16], c1[16], c2[16], c3[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(1,0) + T(1,1) + T(1,2);
    C(1) = T(2,0) + T(2,1) + T(2,2);
    C(2) = T(1,2) + T(1,3) - T(1,1);
    C(3) = T(2,2) + T(2,3) - T(2,1);

    P(0,0) = T(0,0) + T(0,1) + T(0,2) + C(0) + C(1);
    if (with_bias_) P(0, 0) += B;
    P(1,0) = C(1) - C(0) + T(3,0) + T(3,1) + T(3,2);
    if (with_bias_) P(1, 0) += B;
    P(0,1) = T(0,2) - T(0,1) + T(0,3) + C(2) + C(3);
    if (with_bias_) P(0, 1) += B;
    P(1,1) = C(3) - C(2) - T(3,1) + T(3,2) + T(3,3);
    if (with_bias_) P(1, 1) += B;
  }
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 4,
                                    3, 16, ISA_SKX_AVX512, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    MD(float, aoutput,[xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return dummy;
    else
      return aoutput[_h][_w];
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3;
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
  MATRIX_DEF(4, 4);

  c0 = ADD(ADD(t10, t11), t12);
  c1 = ADD(ADD(t20, t21), t22);
  c2 = SUB(ADD(t12, t13), t11);
  c3 = SUB(ADD(t22, t23), t21);

  __m512 p00 = ADD(ADD(ADD(ADD(t00, t01), t02), c0), c1);
  if (with_bias_) p00 = ADD(p00, *(__m512*)bias);
  __m512 p10 = ADD(ADD(ADD(SUB(c1, c0), t30), t31), t32);
  if (with_bias_) p10 = ADD(p10, *(__m512*)bias);
  __m512 p01 = ADD(ADD(ADD(SUB(t02, t01), t03), c2), c3);
  if (with_bias_) p01 = ADD(p01, *(__m512*)bias);
  __m512 p11 = ADD(ADD(SUB(SUB(c3, c2), t31), t32), t33);
  if (with_bias_) p11 = ADD(p11, *(__m512*)bias);

#undef OP
#define p_(m, n) p##m##n
#define OP(m,n) _mm512_store_ps(P(m, n), p_(m, n))
  MATRIX_DEF(2, 2);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 5,
                                    3, 16, ISA_GENERIC, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  float dummy[16];
  auto p_cb = [&](int _h, int _w, int _V) {
    MD(float, aoutput, [xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return &dummy[_V];
    else
      return &aoutput[_h][_w][_V];
  };

#undef T
#undef C
#undef P
#undef B
#define T(_hA, _wA) atoutput[_wA][_hA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p_cb(_h, _w, _V)
#define B bias[_V]
  float c0[16], c1[16], c2[16], c3[16], c4[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(0) = T(0, 0) + T(0, 1) + T(0, 2) + T(0, 3);
    C(1) = T(1, 0) + T(1, 1) + T(1, 2) + T(1, 3);
    C(2) = T(2, 0) + T(2, 1) + T(2, 2) + T(2, 3);
    C(3) = T(3, 0) + T(3, 1) + T(3, 2) + T(3, 3);
    C(4) = T(4, 0) + T(4, 1) + T(4, 2) + T(4, 3);
    P(0, 0) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 0) += B;
    P(1, 0) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 0) += B;
    P(2, 0) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 0) += B;

    C(0) = T(0, 2) - T(0, 1) + 2 * T(0, 3);
    C(1) = T(1, 2) - T(1, 1) + 2 * T(1, 3);
    C(2) = T(2, 2) - T(2, 1) + 2 * T(2, 3);
    C(3) = T(3, 2) - T(3, 1) + 2 * T(3, 3);
    C(4) = T(4, 2) - T(4, 1) + 2 * T(4, 3);
    P(0, 1) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 1) += B;
    P(1, 1) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 1) += B;
    P(2, 1) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 1) += B;

    C(0) = T(0, 1) + T(0, 2) + 4 * T(0, 3) + T(0, 4);
    C(1) = T(1, 1) + T(1, 2) + 4 * T(1, 3) + T(1, 4);
    C(2) = T(2, 1) + T(2, 2) + 4 * T(2, 3) + T(2, 4);
    C(3) = T(3, 1) + T(3, 2) + 4 * T(3, 3) + T(3, 4);
    C(4) = T(4, 1) + T(4, 2) + 4 * T(4, 3) + T(4, 4);
    P(0, 2) = C(0) + C(1) + C(2) + C(3);
    if (with_bias_) P(0, 2) += B;
    P(1, 2) = C(2) - C(1) + 2 * C(3);
    if (with_bias_) P(1, 2) += B;
    P(2, 2) = C(1) + C(2) + 4 * C(3) + C(4);
    if (with_bias_) P(2, 2) += B;
  }
}

#define AVX512_CALCULATE_C_0(z, n, nil)                                        \
  c##n = ADD(ADD(ADD(t##n##0, t##n##1), t##n##2), t##n##3);
#define AVX512_CALCULATE_C_1(z, n, nil)                                        \
  c##n = FMADD(z2, t##n##3, SUB(t##n##2, t##n##1));
#define AVX512_CALCULATE_C_2(z, n, nil)                                        \
  c##n = FMADD(z4, t##n##3, ADD(ADD(t##n##1, t##n##2), t##n##4));
#define AVX512_CALCULATE_P(n)                                                  \
  __m512 p0##n = ADD(ADD(ADD(c0, c1), c2), c3);                                \
  if (with_bias_)                                                              \
    p0##n = ADD(p0##n, *(__m512*)bias);                                        \
  __m512 p1##n = FMADD(z2, c3, SUB(c2, c1));                                   \
  if (with_bias_)                                                              \
    p1##n = ADD(p1##n, *(__m512*)bias);                                        \
  __m512 p2##n = FMADD(z4, c3, ADD(ADD(c1, c2), c4));                          \
  if (with_bias_)                                                              \
    p2##n = ADD(p2##n, *(__m512*)bias);

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 5,
                                    3, 16, ISA_SKX_AVX512, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    MD(float, aoutput,[xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return dummy;
    else
      return aoutput[_h][_w];
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3, c4;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

#undef t
#undef OP
#define t(m, n) t##m##n
#define OP(m,n) __m512 t(m,n) = _mm512_load_ps(T(m, n))
  MATRIX_DEF(5, 5);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_0, nil);
  AVX512_CALCULATE_P(0);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_1, nil);
  AVX512_CALCULATE_P(1);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_2, nil);
  AVX512_CALCULATE_P(2);

#undef OP
#define p(m, n) p##m##n
#define OP(m,n) _mm512_store_ps(P(m, n), p(m, n))
  MATRIX_DEF(3, 3);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_th(winograd_template_parameter_t<S_OUTPUT(float, 4, 3, 16,
                           ISA_GENERIC, false, false)>,
        elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_th(winograd_template_parameter_t<S_OUTPUT(float, 4, 3, 16,
                           ISA_SKX_AVX512, false, false)>,
        elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_th(winograd_template_parameter_t<S_OUTPUT(float, 5, 3, 16,
                           ISA_GENERIC, false, false)>,
        elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_th(winograd_template_parameter_t<S_OUTPUT(float, 5, 3, 16,
                           ISA_SKX_AVX512, false, false)>,
        elx_conv_t<float> &xc, float *toutputa, float *toutput, int Tz)
{
  ENABLE_AVX512F();

  MD(float, atoutput, [A][xc.oc3 * xc.O2][Tz][V], toutput);
  MD(float, atoutputa, [A - K + 1][V], toutputa);

#undef P
#undef T
#define T(_h) atoutput[_h][0][0]
#define P(_h) atoutputa[_h]

  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  __m512 t0 = _mm512_load_ps(T(0));
  __m512 t1 = _mm512_load_ps(T(1));
  __m512 t2 = _mm512_load_ps(T(2));
  __m512 t3 = _mm512_load_ps(T(3));
  __m512 t4 = _mm512_load_ps(T(4));

  __m512 p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  __m512 p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  __m512 p2 = ADD(ADD(ADD(MUL(z4, t3), t2), t1), t4);

  _mm512_store_ps(P(0), p0);
  _mm512_store_ps(P(1), p1);
  _mm512_store_ps(P(2), p2);
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_bh(winograd_template_parameter_t<S_OUTPUT(float, 4, 3, 16,
                           ISA_GENERIC, is_border_, with_bias_)>,
        elx_conv_t<float> &xc, float *output, float atoutput[A][A - K + 1][V],
        float *bias, int _hOA_end, int _wOA_end)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_bh(winograd_template_parameter_t<S_OUTPUT(float, 4, 3, 16,
                           ISA_SKX_AVX512, is_border_, with_bias_)>,
        elx_conv_t<float> &xc, float *output, float atoutput[A][A - K + 1][V],
        float *bias, int _hOA_end, int _wOA_end)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_bh(winograd_template_parameter_t<S_OUTPUT(float, 5, 3, 16,
                           ISA_GENERIC, is_border_, with_bias_)>,
        elx_conv_t<float> &xc, float *output, float atoutput[A][A - K + 1][V],
        float *bias, int _hOA_end, int _wOA_end)
{
  // TODO
}

template <D_OUTPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias)>
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(
    Type, A, K, V, I, is_border, with_bias)>::
    __trans_outputa_bh(winograd_template_parameter_t<S_OUTPUT(float, 5, 3, 16,
                           ISA_SKX_AVX512, is_border_, with_bias_)>,
        elx_conv_t<float> &xc, float *output, float atoutput[A][A - K + 1][V],
        float *bias, int _hOA_end, int _wOA_end)
{
  ENABLE_AVX512F();

  alignas(64) float dummy[16];
  auto p_cb = [&](int _h, int _w) {
    MD(float, aoutput,[xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return dummy;
    else
      return aoutput[_h][_w];
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_w][_h]
#define P(_h, _w) p_cb(_h, _w)

  __m512 c0, c1, c2, c3, c4;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  __m512 t0, t1, t2, t3, t4, p0, p1, p2;
  t0 = _mm512_load_ps(T(0,0));
  t1 = _mm512_load_ps(T(0,1));
  t2 = _mm512_load_ps(T(0,2));
  t3 = _mm512_load_ps(T(0,3));
  t4 = _mm512_load_ps(T(0,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(0,0), p0);
  _mm512_store_ps(P(0,1), p1);
  _mm512_store_ps(P(0,2), p2);

  t0 = _mm512_load_ps(T(1,0));
  t1 = _mm512_load_ps(T(1,1));
  t2 = _mm512_load_ps(T(1,2));
  t3 = _mm512_load_ps(T(1,3));
  t4 = _mm512_load_ps(T(1,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(1,0), p0);
  _mm512_store_ps(P(1,1), p1);
  _mm512_store_ps(P(1,2), p2);

  t0 = _mm512_load_ps(T(2,0));
  t1 = _mm512_load_ps(T(2,1));
  t2 = _mm512_load_ps(T(2,2));
  t3 = _mm512_load_ps(T(2,3));
  t4 = _mm512_load_ps(T(2,4));

  p0 = ADD(ADD(ADD(t0, t1), t2), t3);
  if (with_bias_) p0 = ADD(p0, *(__m512*)bias);
  p1 = SUB(ADD(MUL(z2, t3), t2), t1);
  if (with_bias_) p1 = ADD(p1, *(__m512*)bias);
  p2 = ADD(ADD(ADD(MUL(z4, t3), t1), t2), t4);
  if (with_bias_) p2 = ADD(p2, *(__m512*)bias);

  _mm512_store_ps(P(2,0), p0);
  _mm512_store_ps(P(2,1), p1);
  _mm512_store_ps(P(2,2), p2);
}

template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);

template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_outputa_th(elx_conv_t<float> &, float *,
    float *, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);

template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[4][4][16], float *, int, int);

template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_outputa_th(elx_conv_t<float> &, float *,
    float *, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 4, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[4][2][16], float *, int, int);

template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);

template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_outputa_th(elx_conv_t<float> &, float *,
    float *, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(false), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_GENERIC,
    BORDER(true), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);

template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(true))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(false))>::trans_output(elx_conv_t<float> &, float *,
    float[5][5][16], float *, int, int);

template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_outputa_th(elx_conv_t<float> &, float *,
    float *, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(true))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(false), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);
template void
convolution_winograd_kernel<S_OUTPUT(float, 5, 3, 16, ISA_SKX_AVX512,
    BORDER(true), BIAS(false))>::trans_outputa_bh(elx_conv_t<float> &, float *,
    float[5][3][16], float *, int, int);



} // namespace euler
