#pragma once
#include <assert.h>
#include <x86intrin.h>
#include "elk_def.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename UserTypes, typename TrOpType, int v>
class convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, v, 4, 3> {
  protected:
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  constexpr static int I = ISA_SKX_AVX512;
  constexpr static int V = v;
  constexpr static int A = 4;
  constexpr static int K = 3;

  template <int input_format, bool is_border> static inline
  void __trans_input(elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V],
      InputType *input, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<UserTypes> &xc, OutputType *output,
      TrOpType atoutput[A][A][V], BiasType *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(
      TrOpType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]);
};


template <typename UserTypes, typename TrOpType, int V>
template <int input_format, bool is_border>
inline void convolution_winograd_kernel_base<UserTypes, TrOpType,
    ISA_SKX_AVX512, V, 4, 3>::__trans_input(
      elx_conv_t<UserTypes> &xc, TrOpType atinput[A][A][V], InputType *input,
      int hT_start, int hT_end, int wT_start, int wT_end)
{
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
    if (input_format == TKF_COMPACT) {// TODO: wT_end == -1
      MD3(InputType, ainput, input, A, A, V);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
        return _mm<V>::cvtph_ps(f16);
      }
    } else if (input_format == TKF_BLOCKED) {
      MD3(InputType, ainput, input, xc.ih, xc.iw, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md3(ainput, _h, _w, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md3(ainput, _h, _w, 0));
        return _mm<V>::cvtph_ps(f16);
      }
    } else { // TKF_NHWC
      MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.ic);
      // TODO: overflow on last V
      MD2(InputType, ainput1, &md3(ainput0, _h, _w, 0), xc.ic4 * xc.ic3 * xc.I2, V);
      if (is_border
          && (_h < hT_start || _w < wT_start || _h > hT_end
                 || _w > wT_end))
        return z0;
      else if (std::is_same<InputType, float>::value)
        return _mm<V>::load_ps(&md2(ainput1, 0, 0));
      else {
        auto f16 = _mm<V/2>::load_si256((__m256i *)&md2(ainput1, 0, 0));
        return _mm<V>::cvtph_ps(f16);
      }
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

}//namespace euler
