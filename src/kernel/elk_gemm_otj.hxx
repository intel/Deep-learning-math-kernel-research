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
struct gemm_kernel_otj {};

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

  // O (T + P) + 1 <= 32
  template <int O> static inline constexpr typename
  std::enable_if<(32 / O - T) >= 4, int>::type getP() {
    return 4;
  }
  template <int O> static inline constexpr typename
  std::enable_if<((32 / O - T) >= 2) && ((32 / O - T) < 4), int>::type getP() {
    return 2;
  }
  template <int O> static inline constexpr typename
  std::enable_if<(32 / O - T) == 1, int>::type getP() {
    return 1;
  }
  template <int O> static inline constexpr typename
  std::enable_if<(32 / O - T) < 1, int>::type getP() {
    // Wrong path to make compiler happy
    return 0xd;
  }

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

  static inline void execute(elx_conv_t<float> &xc, float *output, float *input,
      float *weights, float *bias, bool reset_output)
  {
    MD2(float, aoutput, output, O2, T *V);
    MD4(float, aweights, weights, xc.I2, V, O2, V);
    MD2(float, abias, bias, O2, V);

    if constexpr(O2 * T < 32)
      op_fma<O2, getP<O2>()>(xc, output, input, weights, bias, reset_output);
    else if constexpr(O2 == 3 && T < 15) {
      op_fma<2, getP<2>()>(xc, output, input, weights, bias, reset_output);
      op_fma<1, getP<1>()>(xc, &md2(aoutput, 2, 0), input,
          &md4(aweights, 0, 0, 2, 0), &md2(abias, 2, 0), reset_output);
    } else if constexpr(O2 == 4 && T < 15) {
      op_fma<2, getP<2>()>(xc, output, input, weights, bias, reset_output);
      op_fma<2, getP<2>()>(xc, &md2(aoutput, 2, 0), input,
          &md4(aweights, 0, 0, 2, 0), &md2(abias, 2, 0), reset_output);
    } else if constexpr(O2 == 8 && T < 6) {
      op_fma<5, getP<5>()>(xc, output, input, weights, bias, reset_output);
      op_fma<3, getP<3>()>(xc, &md2(aoutput, 5, 0), input,
          &md4(aweights, 0, 0, 5, 0), &md2(abias, 5, 0), reset_output);
    } else if constexpr(O2 == 8 && T == 6) {
      op_fma<4, getP<4>()>(xc, output, input, weights, bias, reset_output);
      op_fma<4, getP<4>()>(xc, &md2(aoutput, 4, 0), input,
          &md4(aweights, 0, 0, 4, 0), &md2(abias, 4, 0), reset_output);
    } else if constexpr(O2 == 8 && T < 9) {
      op_fma<3, getP<3>()>(xc, output, input, weights, bias, reset_output);
      op_fma<3, getP<3>()>(xc, &md2(aoutput, 3, 0), input,
          &md4(aweights, 0, 0, 3, 0), &md2(abias, 3, 0), reset_output);
      op_fma<2, getP<2>()>(xc, &md2(aoutput, 6, 0), input,
          &md4(aweights, 0, 0, 6, 0), &md2(abias, 6, 0), reset_output);
    }
  }
};


}
