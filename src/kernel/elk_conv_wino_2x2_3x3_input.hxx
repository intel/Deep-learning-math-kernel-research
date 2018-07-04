#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <D_INPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border)>
template <const bool is_border_>
void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I,
    is_border)>::__trans_input(winograd_template_parameter_t<S_INPUT(float, 4,
                                   3, 16, ISA_GENERIC, is_border_)>,
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end)
{
  auto f_cb = [&](int _h, int _w, int _V) {
    MD(float, ainput, [xc.ih][xc.iw][16], input);
    if (is_border_
        && (_h < _hT_start || _w < _wT_start || _h > _hT_end || _w > _wT_end))
      return 0.0f;
    else
      return ainput[_h][_w][_V];
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w, _V)
#define C(n) C##n[_V]
#define T(_h, _w) atinput[_w][_h][_V]

  float C1[16], C2[16];
#pragma omp simd
  for (int _V = 0; _V < 16; ++_V) {
    C(1) = F(1,2) - F(1,0);
    C(2) = F(2,0) - F(2,2);
    T(0,0) = F(0,0) - F(0,2) - C(2);
    T(1,0) = C(1) + C(2);
    T(2,0) = C(2) - C(1);
    T(3,0) = C(1) + F(3,0) - F(3,2);

    C(1) = F(1,2) - F(1,1);
    C(2) = F(2,2) - F(2,1);
    T(0,1) = F(0,2) - F(0,1) - C(2); 
    T(1,1) = C(2) - C(1);
    T(2,1) = C(2) + C(1);
    T(3,1) = F(3,2) - F(3,1) - C(1);

    C(1) = F(1,1) + F(1,2);
    C(2) = F(2,1) + F(2,2);
    T(0,2) = F(0,1) + F(0,2) - C(2);
    T(1,2) = C(2) - C(1);
    T(2,2) = C(2) + C(1);
    T(3,2) = F(3,1) + F(3,2) - C(1);

    C(1) = F(1,1) - F(1,3);
    C(2) = F(2,3) - F(2,1);
    T(0,3) = F(0,3) - F(0,1) - C(2);
    T(1,3) = C(1) + C(2);
    T(2,3) = C(2) - C(1);
    T(3,3) = C(1) - F(3,1) + F(3,3);
  }
}

template <D_INPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border)>
template <const bool is_border_>
void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I,
    is_border)>::__trans_input(winograd_template_parameter_t<S_INPUT(float, 4,
                                    3, 16, ISA_SKX_AVX512, is_border_)>,
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end)
{
  ENABLE_AVX512F();

  // Cache
  __m512 c1, c2;
  // Outputs
  __m512 t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23,
      t30, t31, t32, t33;

  __m512 z0 = _mm512_setzero_ps();
  auto f_cb = [&](int _h, int _w) {
    MD(float, ainput, [xc.ih][xc.iw][16], input);
    if (is_border_
        && (_h < _hT_start || _w < _wT_start || _h > _hT_end || _w > _wT_end))
      return z0;
    else
      return _mm512_load_ps(ainput[_h][_w]);
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#define f(m, n) f##m##n
#define OP(m,n) __m512 f(m, n) = F(m, n)
  MATRIX_DEF(4, 4);

  c1 = SUB(f12, f10);
  c2 = SUB(f20, f22);
  t00 =  SUB(SUB(f00, f02), c2);
  _mm512_store_ps(T(0, 0), t00);
  t10 = ADD(c1, c2);
  _mm512_store_ps(T(1, 0), t10);
  t20 = SUB(c2, c1);
  _mm512_store_ps(T(2, 0), t20);
  t30 = ADD(c1, SUB(f30, f32));
  _mm512_store_ps(T(3, 0), t30);

  c1 = SUB(f12, f11);
  c2 = SUB(f22, f21);
  t01 = SUB(SUB(f02, f01), c2);
  _mm512_store_ps(T(0, 1), t01);
  t11 = SUB(c2, c1);
  _mm512_store_ps(T(1, 1), t11);
  t21 = ADD(c2, c1);
  _mm512_store_ps(T(2, 1), t21);
  t31 = SUB(SUB(f32, f31), c1);
  _mm512_store_ps(T(3, 1), t31);

  c1 = ADD(f11, f12);
  c2 = ADD(f21, f22);
  t02 = SUB(ADD(f01, f02), c2);
  _mm512_store_ps(T(0, 2), t02);
  t12 = SUB(c2, c1);
  _mm512_store_ps(T(1, 2), t12);
  t22 = ADD(c2, c1);
  _mm512_store_ps(T(2, 2), t22);
  t32 = SUB(ADD(f31, f32), c1);
  _mm512_store_ps(T(3, 2), t32);

  c1 = SUB(f11, f13);
  c2 = SUB(f23, f21);
  t03 = SUB(SUB(f03, f01), c2);
  _mm512_store_ps(T(0, 3), t03);
  t13 = ADD(c1, c2);
  _mm512_store_ps(T(1, 3), t13);
  t23 = SUB(c2, c1);
  _mm512_store_ps(T(2, 3), t23);
  t33 = ADD(SUB(c1, f31), f33);
  _mm512_store_ps(T(3, 3), t33);
}

template <D_INPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border)>
template <const bool is_border_>
void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I,
    is_border)>::__trans_inputa(winograd_template_parameter_t<S_INPUT(float, 4,
                                    3, 16, ISA_SKX_AVX512, is_border_)>,
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int _wA, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end)
{
  // TODO
}

template <D_INPUT(typename Type, const int A, const int K, const int V,
    const int I, const bool is_border)>
template <const bool is_border_>
void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I,
    is_border)>::__trans_inputa(winograd_template_parameter_t<S_INPUT(float, 4,
                                    3, 16, ISA_GENERIC, is_border_)>,
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int _wA, int _hT_start,
    int _hT_end, int _wT_start, int _wT_end)
{
  // TODO
}

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(false))>::trans_input(elx_conv_t<float> &, float[4][4][16], float *,
    int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16, ISA_GENERIC,
    BORDER(true))>::trans_input(elx_conv_t<float> &, float[4][4][16], float *,
    int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_SKX_AVX512, BORDER(false))>::trans_input(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_SKX_AVX512, BORDER(true))>::trans_input(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_GENERIC, BORDER(false))>::trans_inputa(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_GENERIC, BORDER(true))>::trans_inputa(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_SKX_AVX512, BORDER(false))>::trans_inputa(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int, int);

template void convolution_winograd_kernel<S_INPUT(float, 4, 3, 16,
    ISA_SKX_AVX512, BORDER(true))>::trans_inputa(elx_conv_t<float> &,
    float[4][4][16], float *, int, int, int, int, int);

}
