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
#define ADD _mm512_add_ps
#define SUB _mm512_sub_ps
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
template <const bool is_border_, const bool with_bias_>
void convolution_winograd_kernel<R_OUTPUT(Type, A, K, V, I, is_border,
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 4,
                                    3, 16, ISA_GENERIC, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  float dummy[16];
  auto p = [&](int _h, int _w, int _V) {
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
#define T(_hA, _wA) atoutput[_hA][_wA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p(_h, _w, _V)
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
    with_bias)>::__trans_output(winograd_template_parameter_t<S_OUTPUT(float, 5,
                                    3, 16, ISA_GENERIC, is_border_,
                                    with_bias_)>,
    elx_conv_t<float> &xc, float *output, float atoutput[A][A][V], float *bias,
    int _hOA_end, int _wOA_end)
{
  float dummy[16];
  auto p = [&](int _h, int _w, int _V) {
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
#define T(_hA, _wA) atoutput[_hA][_wA][_V]
#define C(n) c##n[_V]
#define P(_h, _w) *p(_h, _w, _V)
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

#define AVX512_LOAD0(z, n, nil) __m512 t0##n = _mm512_load_ps(T(0, n));
#define AVX512_LOAD1(z, n, nil) __m512 t1##n = _mm512_load_ps(T(1, n));
#define AVX512_LOAD2(z, n, nil) __m512 t2##n = _mm512_load_ps(T(2, n));
#define AVX512_LOAD3(z, n, nil) __m512 t3##n = _mm512_load_ps(T(3, n));
#define AVX512_LOAD4(z, n, nil) __m512 t4##n = _mm512_load_ps(T(4, n));
#define LOAD_ZMMS()                                                            \
  BOOST_PP_REPEAT(5, AVX512_LOAD0, nil);                                       \
  BOOST_PP_REPEAT(5, AVX512_LOAD1, nil);                                       \
  BOOST_PP_REPEAT(5, AVX512_LOAD2, nil);                                       \
  BOOST_PP_REPEAT(5, AVX512_LOAD3, nil);                                       \
  BOOST_PP_REPEAT(5, AVX512_LOAD4, nil);

#define AVX512_STORE0(z, n, nil) _mm512_store_ps(P(0, n), p0##n);
#define AVX512_STORE1(z, n, nil) _mm512_store_ps(P(1, n), p1##n);
#define AVX512_STORE2(z, n, nil) _mm512_store_ps(P(2, n), p2##n);
#define STORE_ZMMS()                                                           \
  BOOST_PP_REPEAT(3, AVX512_STORE0, nil);                                      \
  BOOST_PP_REPEAT(3, AVX512_STORE1, nil);                                      \
  BOOST_PP_REPEAT(3, AVX512_STORE2, nil);

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
  auto p = [&](int _h, int _w) {
    MD(float, aoutput,[xc.oh][xc.ow][16], output);
    if (is_border_ && (_h > _hOA_end || _w > _wOA_end))
      return dummy;
    else
      return aoutput[_h][_w];
  };

#undef P
#undef T
#define T(_h, _w) atoutput[_h][_w]
#define P(_h, _w) p(_h, _w)

  __m512 c0, c1, c2, c3, c4;
  __m512 z2 = _mm512_set_ps(IMM_BCAST16(2.0f));
  __m512 z4 = _mm512_set_ps(IMM_BCAST16(4.0f));

  LOAD_ZMMS();

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_0, nil);
  AVX512_CALCULATE_P(0);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_1, nil);
  AVX512_CALCULATE_P(1);

  BOOST_PP_REPEAT(5, AVX512_CALCULATE_C_2, nil);
  AVX512_CALCULATE_P(2);

  STORE_ZMMS();
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

} // namespace euler
