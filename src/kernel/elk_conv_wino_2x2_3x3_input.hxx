#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

// template <const bool is_border_>
// Params:
//    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//    int _hT_start, int _hT_end, int _wT_start, int _wT_end
template <>
class convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3> {
  template<typename Type, int ...configs>
    friend class convolution_winograd_kernel_base;
protected:
  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int V = 16;
  constexpr static int A = 4;
  constexpr static int K = 3;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end);

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int wA, int hA_start, int hA_end, int wA_start,
      int _wA_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<float> &xc, float *output,
      float atoutput[A][A][V], float *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<float> &xc, float *toutputa,
      float *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<float> &xc, float *output,
      float aoutputa[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]);
};


template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3>::__trans_input(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  ENABLE_AVX512F();

  // Inputs
    __m<V> f00, f01, f02, f03, f10, f11, f12, f13, f20, f21, f22, f23,
           f30, f31, f32, f33;
  // Cache
  __m<V> c1, c2;
  // Outputs
  __m<V> t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23,
      t30, t31, t32, t33;

  __m<V> z0 = _mm<V>::setzero_ps();
  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
    }
  };

#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[w][h]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)

  MATRIX_DEF(4, 4);

   c1 = SUB(f12, f10);
   c2 = SUB(f20, f22);
   t00 =  SUB(SUB(f00, f02), c2);
   _mm<V>::store_ps(T(0, 0), t00);
   t10 = ADD(c1, c2);
   _mm<V>::store_ps(T(1, 0), t10);
   t20 = SUB(c2, c1);
   _mm<V>::store_ps(T(2, 0), t20);
   t30 = ADD(c1, SUB(f30, f32));
   _mm<V>::store_ps(T(3, 0), t30);

   c1 = SUB(f12, f11);
   c2 = SUB(f22, f21);
   t01 = SUB(SUB(f02, f01), c2);
   _mm<V>::store_ps(T(0, 1), t01);
   t11 = SUB(c2, c1);
   _mm<V>::store_ps(T(1, 1), t11);
   t21 = ADD(c2, c1);
   _mm<V>::store_ps(T(2, 1), t21);
   t31 = SUB(SUB(f32, f31), c1);
   _mm<V>::store_ps(T(3, 1), t31);

   c1 = ADD(f11, f12);
   c2 = ADD(f21, f22);
   t02 = SUB(ADD(f01, f02), c2);
   _mm<V>::store_ps(T(0, 2), t02);
   t12 = SUB(c2, c1);
   _mm<V>::store_ps(T(1, 2), t12);
   t22 = ADD(c2, c1);
   _mm<V>::store_ps(T(2, 2), t22);
   t32 = SUB(ADD(f31, f32), c1);
   _mm<V>::store_ps(T(3, 2), t32);

   c1 = SUB(f11, f13);
   c2 = SUB(f23, f21);
   t03 = SUB(SUB(f03, f01), c2);
   _mm<V>::store_ps(T(0, 3), t03);
   t13 = ADD(c1, c2);
   _mm<V>::store_ps(T(1, 3), t13);
   t23 = SUB(c2, c1);
   _mm<V>::store_ps(T(2, 3), t23);
   t33 = ADD(SUB(c1, f31), f33);
   _mm<V>::store_ps(T(3, 3), t33);
}


// template <const bool is_border_>
// Params:
//   elx_conv_t<float> &xc, float atinput[A][A][V], float *input,
//   int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)
template <bool is_border>
inline void convolution_winograd_kernel_base<float, ISA_SKX_AVX512, 16, 4, 3>::
__trans_inputa(
    elx_conv_t<float> &xc, float atinput[A][A][V], float *input, int wA,
    int hT_start, int hT_end, int wT_start, int wT_end) {
  ENABLE_AVX512F();
  // Inputs
    __m<V> f00, f01, f02, f03, f10, f11, f12, f13, f20, f21, f22, f23,
           f30, f31, f32, f33;   
 // Cache
  __m<V> c1, c2;
  // Outputs
  __m<V> t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23,
      t30, t31, t32, t33;

  __m<V> z0 = _mm<V>::setzero_ps();
  auto f_cb = [&](int _h, int _w) {
    if (wT_end == -1) {
      MD3(float, ainput, input, A, A, 16);
      return _mm<V>::load_ps(&md3(ainput,_h, _w, 0));
    } else {
      MD3(float, ainput, input, xc.ih, xc.iw, 16);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else
        return _mm<V>::load_ps(&md3(ainput,_h, _w, 0));
    }
  };


#undef F
#undef C
#undef T
#define F(_h, _w) f_cb(_h, _w)
#define T(h, w) atinput[h][w]

#undef f
#undef OP
#define f(m, n) f##m##n
#define OP(m,n) f(m, n) = F(m, n)


  MATRIX_DEF(4, 4);

  switch(wA){
  case 0:
    c1 = SUB(f12, f10);
    c2 = SUB(f20, f22);
    t00 =  SUB(SUB(f00, f02), c2);
    _mm<V>::store_ps(T(0, 0), t00);
    t10 = ADD(c1, c2);
    _mm<V>::store_ps(T(1, 0), t10);
    t20 = SUB(c2, c1);
    _mm<V>::store_ps(T(2, 0), t20);
    t30 = ADD(c1, SUB(f30, f32));
    _mm<V>::store_ps(T(3, 0), t30);

    break;

  case 1:
    c1 = SUB(f12, f11);
    c2 = SUB(f22, f21);
    t01 = SUB(SUB(f02, f01), c2);
    _mm<V>::store_ps(T(0, 1), t01);
    t11 = SUB(c2, c1);
    _mm<V>::store_ps(T(1, 1), t11);
    t21 = ADD(c2, c1);
    _mm<V>::store_ps(T(2, 1), t21);
    t31 = SUB(SUB(f32, f31), c1);
    _mm<V>::store_ps(T(3, 1), t31);

    break;

  case 2:
    c1 = ADD(f11, f12);
    c2 = ADD(f21, f22);
    t02 = SUB(ADD(f01, f02), c2);
    _mm<V>::store_ps(T(0, 2), t02);
    t12 = SUB(c2, c1);
    _mm<V>::store_ps(T(1, 2), t12);
    t22 = ADD(c2, c1);
    _mm<V>::store_ps(T(2, 2), t22);
    t32 = SUB(ADD(f31, f32), c1);
    _mm<V>::store_ps(T(3, 2), t32);

    break;

  case 3:
    c1 = SUB(f11, f13);
    c2 = SUB(f23, f21);
    t03 = SUB(SUB(f03, f01), c2);
    _mm<V>::store_ps(T(0, 3), t03);
    t13 = ADD(c1, c2);
    _mm<V>::store_ps(T(1, 3), t13);
    t23 = SUB(c2, c1);
    _mm<V>::store_ps(T(2, 3), t23);
    t33 = ADD(SUB(c1, f31), f33);
    _mm<V>::store_ps(T(3, 3), t33);
    break;
  }
}


}//namespace euler
