#pragma once

#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "el_stl.hpp"

// Type: data type
// S: stride
// O2: OC blocking
// T: tile blocking unit
// F: format
// V: vector size
// I: ISA
// is_Ir/Or/Tr: is the last IV, OV or T
// with_bias: has bias
// with_relu: with relu fusion
// with_sum: with sum fusion

namespace euler {

const int GKF_CCC = 0xccc;
const int GKF_CCD = 0xccd;
const int GKF_DDD = 0xddd;

template <typename Type, int V, int I, int F, int S, typename KP>
struct gemm_kernel_otj {
  static inline void execute(elx_conv_t<float> &xc, float *output, float *input,
      float *weights, float *bias, bool reset_output)
  {}
};

// O == 1: T + P <= 32
// O > 1: O (T + P) + 1 <= 32
template <int O, int T> struct para_traits {
  static constexpr int value
      = (O == 1 && T <= 28) || (O > 1 && (31 / O - T) >= 4)
      ? 4
      : (O == 1 && (T == 29 || T == 30)) || (O > 1 && (31 / O - T) >= 2) ? 2
                                                                         : 1;
};

template <int... Kp>
struct gemm_kernel_otj<float, 16, ISA_SKX_AVX512, GKF_CCC, 1,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 8,
      "Kernel parameters must be Type, V, I, F, S, O2, T, ...");

  constexpr static auto V = 16;
  constexpr static auto O2 = estl::get<0, int, kparams>();
  constexpr static auto T = estl::get<1, int, kparams>();
  constexpr static auto is_Ir = estl::get<2, bool, kparams>();
  constexpr static auto is_Or = estl::get<3, bool, kparams>();
  constexpr static auto is_Tr = estl::get<4, bool, kparams>();
  constexpr static auto with_bias = estl::get<5, bool, kparams>();
  constexpr static auto with_relu = estl::get<6, bool, kparams>();
  constexpr static auto with_sum = estl::get<7, bool, kparams>();

  template <int O, int P>
  static inline void op_fma(elx_conv_t<float> &xc, float *output,
      float *input, float *weights, float *bias, bool reset_output)
  {
    __m512 mmbcst, mmout[O][T], mmwei[O][P];

    MD3(float, aoutput3, output, O, T, V);
    MD4(float, ainput4, input, xc.I2, T, V/P, P);
    MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
    MD2(float, abias2, bias, O, V);

    if (P >= 2) {
      // preload weights
      for (int _O = 0; _O < O; ++_O) {
        mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, 0, 0, 0, _O, 0));
        if (P == 4)
          mmwei[_O][1] = _mm512_load_ps(&md5(aweights5, 0, 0, 1, _O, 0));
      }
    }
    if (reset_output) {
      if (with_bias) {
        // load bias
        for (int _O = 0; _O < O; ++_O) {
          __m512 tmp = _mm512_load_ps(&md2(abias2, _O, 0));
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m512 tmp = _mm512_setzero_ps();
        for (int _O = 0; _O < O; ++_O)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
    } else {
      // load output
      for (int _O = 0; _O < O; ++_O)
        for (int _T = 0; _T < T; ++_T)
          mmout[_O][_T] = _mm512_load_ps(&md3(aoutput3, _O, _T, 0));
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
      for (int _V = 0; _V < V / P; ++_V) {
        for (int _P = 0; _P < P; ++_P) {
          // load weights
          for (int _O = 0; _O < O; ++_O) {
            const int p = (_P + P/2) % P;
            const int v = _P >= P/2 ? _V + 1 : _V;
            mmwei[_O][p] = _mm512_load_ps(&md5(aweights5, _I2, v, p, _O, 0));
          }
          // FMA
          for (int _T = 0; _T < T; ++_T) {
            mmbcst = _mm512_broadcastss_ps(
                *(__m128 *)&md4(ainput4, _I2, _T, _V, _P));
            for (int _O = 0; _O < O; ++_O)
              mmout[_O][_T]
                  = _mm512_fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
          }
        }
      }
    }

    // store output
    for (int _O = 0; _O < O; ++_O)
      for (int _T = 0; _T < T; ++_T)
        _mm512_store_ps(&md3(aoutput3, _O, _T, 0), mmout[_O][_T]);
  }

  template <int O2 = O2, int T = T>
  static inline
      typename std::enable_if<(O2 == 1 && T < 32) || (O2 == 2 && T < 15)
              || (O2 == 3 && T < 10) || (O2 == 4 && T < 7) || (O2 == 5 && T < 6)
              || (O2 == 6 && T < 5) || (O2 == 7 && T < 4) || (O2 == 8 && T < 3),
          void>::type
      execute(elx_conv_t<float> &xc, float *output, float *input,
          float *weights, float *bias, bool reset_output)
  {
    op_fma<O2, para_traits<O2, T>::value>(
        xc, output, input, weights, bias, reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline
      typename std::enable_if<O2 == 3 && (T >= 10 && T < 15), void>::type
      execute(elx_conv_t<float> &xc, float *output, float *input,
          float *weights, float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    op_fma<2, para_traits<2, T>::value>(
        xc, output, input, weights, bias, reset_output);
    op_fma<1, para_traits<1, T>::value>(xc, &md2(aoutput, 2, 0), input,
        &md4(aweights, 0, 0, 2, 0), &md2(abias, 2, 0), reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline
      typename std::enable_if<O2 == 4 && (T >= 7 && T < 15), void>::type
      execute(elx_conv_t<float> &xc, float *output, float *input,
          float *weights, float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    op_fma<2, para_traits<2, T>::value>(
        xc, output, input, weights, bias, reset_output);
    op_fma<2, para_traits<2, T>::value>(xc, &md2(aoutput, 2, 0), input,
        &md4(aweights, 0, 0, 2, 0), &md2(abias, 2, 0), reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline
      typename std::enable_if<O2 == 8 && (T >= 3 && T < 6), void>::type
      execute(elx_conv_t<float> &xc, float *output, float *input,
          float *weights, float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    op_fma<5, para_traits<5, T>::value>(
        xc, output, input, weights, bias, reset_output);
    op_fma<3, para_traits<3, T>::value>(xc, &md2(aoutput, 5, 0), input,
        &md4(aweights, 0, 0, 5, 0), &md2(abias, 5, 0), reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline typename std::enable_if<O2 == 8 && T == 6, void>::type execute(
      elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    op_fma<4, para_traits<4, T>::value>(
        xc, output, input, weights, bias, reset_output);
    op_fma<4, para_traits<4, T>::value>(xc, &md2(aoutput, 4, 0), input,
        &md4(aweights, 0, 0, 4, 0), &md2(abias, 4, 0), reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline
      typename std::enable_if<O2 == 8 && (T == 7 || T == 8), void>::type
      execute(elx_conv_t<float> &xc, float *output, float *input,
          float *weights, float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    op_fma<3, para_traits<3, T>::value>(
        xc, output, input, weights, bias, reset_output);
    op_fma<3, para_traits<3, T>::value>(xc, &md2(aoutput, 3, 0), input,
        &md4(aweights, 0, 0, 3, 0), &md2(abias, 3, 0), reset_output);
    op_fma<2, para_traits<2, T>::value>(xc, &md2(aoutput, 6, 0), input,
        &md4(aweights, 0, 0, 6, 0), &md2(abias, 6, 0), reset_output);
  }
};


} // namespace euler
