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
// has_Ir/Or/Tr: is the last IV, OV or T
// with_bias: has bias
// with_relu: with relu fusion
// with_sum: with sum fusion

namespace euler {

const int GKF_CCC = 0xccc;
const int GKF_CCD = 0xccd;
const int GKF_DCD = 0xdcd;
const int GKF_DDD = 0xddd;

// Parallel level
template <int O, int T, bool has_Ir, typename C = void> struct P_traits {};

// O == 1: T + P <= 32
// O > 1: O (T + P) + 1 <= 32
template <int T>
struct P_traits<1, T, false, typename std::enable_if<(T <= 28)>::type> {
  static constexpr int P = 4;
};

template <int T>
struct P_traits<1, T, false,
    typename std::enable_if<(T == 29 || T == 30)>::type> {
  static constexpr int P = 2;
};

template <int T, bool has_Ir>
struct P_traits<1, T, has_Ir, typename std::enable_if<(T >= 31)>::type> {
  static constexpr int P = 1;
};

template <int O, int T>
struct P_traits<O, T, false,
    typename std::enable_if<(O > 1 && (31 / O - T) >= 4)>::type> {
  static constexpr int P = 4;
};

template <int O, int T>
struct P_traits<O, T, false,
    typename std::enable_if<(
        O > 1 && (31 / O - T == 2 || 31 / O - T == 3))>::type> {
  static constexpr int P = 2;
};

template <int O, int T, bool has_Ir>
struct P_traits<O, T, has_Ir,
    typename std::enable_if<(O > 1 && (31 / O - T) == 1)>::type> {
  static constexpr int P = 1;
};

// Jamming
template <int O, int T, bool has_Ir, typename C = void> struct J_traits {};

template <int T, bool has_Ir>
struct J_traits<8, T, has_Ir, typename std::enable_if<T == 6>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir>
struct J_traits<8, T, has_Ir,
    typename std::enable_if<T == 7 || T == 8, void>::type> {
  static constexpr int J = 3;
  static constexpr int O0 = 3;
  static constexpr int O1 = 3;
  static constexpr int O2 = 2;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir>::P;
  static constexpr int P2 = P_traits<O2, T, has_Ir>::P;
};

template <int T, bool has_Ir>
struct J_traits<8, T, has_Ir,
    typename std::enable_if<(T >= 3 && T < 6), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir>
struct J_traits<4, T, has_Ir, typename std::enable_if<(T >= 7 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 2;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir>
struct J_traits<3, T, has_Ir,
    typename std::enable_if<(T >= 10 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 1;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir>::P;
  static constexpr int P2 = 0;
};

template <int O, int T, bool has_Ir>
struct J_traits<O, T, has_Ir,
    typename std::enable_if<(O == 1 && T < 32) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int F>
struct F_traits {
  static constexpr bool is_compact_input = (F & 0xF00) == 0xC00;
  static constexpr bool is_compact_weights = (F & 0xF0) == 0xC0;
  static constexpr bool is_compact_output = (F & 0xF) == 0xC;
};

template <typename Type, int V, int I, typename KP>
struct gemm_kernel_otj {
  static inline void execute(
      elx_conv_t<float> &, float *, float *, float *, float *, bool)
  {}
};

template <int... Kp>
struct gemm_kernel_otj<float, 16, ISA_SKX_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 8,
      "Kernel parameters must be Type, V, I, <S, F, O2, T, ...");

  constexpr static auto V = 16;
  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O2 = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();
  constexpr static auto has_Ir = estl::get<4, bool, kparams>();
  constexpr static auto with_bias = estl::get<5, bool, kparams>();
  constexpr static auto with_relu = estl::get<6, bool, kparams>();
  constexpr static auto with_sum = estl::get<7, bool, kparams>();

  // Jamming components
  constexpr static int J = J_traits<O2, T, has_Ir>::J;
  constexpr static int JO0 = J_traits<O2, T, has_Ir>::O0;
  constexpr static int JP0 = J_traits<O2, T, has_Ir>::P0;
  constexpr static int JO1 = J_traits<O2, T, has_Ir>::O1;
  constexpr static int JP1 = J_traits<O2, T, has_Ir>::P1;
  constexpr static int JO2 = J_traits<O2, T, has_Ir>::O2;
  constexpr static int JP2 = J_traits<O2, T, has_Ir>::P2;

  template <int O, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == false), void>::type
  op_fma(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    __m512 mmout[O][T], mmwei[O][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, O, V);

    if (reset_output) {
      if (with_bias) {
        // load bias
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          __m512 tmp = _mm512_load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m512 tmp = _mm512_setzero_ps();
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
    } else {
      // load output
#pragma unroll(O)
      for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm512_load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, _I2, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O) {
            MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
            __m512 mmbcst = _mm512_set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }
    }

    // store output
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        _mm512_store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int O, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == true), void>::type
  op_fma(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    __m512 mmout[O][T], mmwei[O][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, O, V);

    if (reset_output) {
      if (with_bias) {
        // load bias
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          __m512 tmp = _mm512_load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m512 tmp = _mm512_setzero_ps();
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
    } else {
      // load output
#pragma unroll(O)
      for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm512_load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2 - 1; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V; ++_V) {
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V, 1, O2, V);
            mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, _I2, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V, 1, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O) {
            MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V, 1);
            __m512 mmbcst = _mm512_set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }
    }
    // Ir
    {
#pragma nounroll
      for (int _V = 0; _V < xc.Ir; ++_V) {
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V, 1, O2, V);
            mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, _I2, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V, 1, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O) {
            MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V, 1);
            __m512 mmbcst = _mm512_set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }

    }

    // store output
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        _mm512_store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }


  template <int O, int P>
  static inline typename std::enable_if<P == 2, void>::type op_fma(
      elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    __m512 mmout[O][T], mmwei[O][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, O, V);

    // preload weights
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
      if (F_traits<F>::is_compact_weights) {
        MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
        mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, 0, 0, 0, _O, 0));
      } else {
        MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
        mmwei[_O][0] = _mm512_load_ps(&md6(aweights6, _O, 0, 0, 0, 0, 0));
      }
    }

    if (reset_output) {
      if (with_bias) {
        // load bias
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          __m512 tmp = _mm512_load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m512 tmp = _mm512_setzero_ps();
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
    } else {
      // load output
#pragma unroll(O)
      for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm512_load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][1] = _mm512_load_ps(&md5(aweights5, _I2, _V, 1, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][1]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 1, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 0));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
        }
        // _P = 1
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md5(aweights5, _I2, _V + 1, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 1));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][1], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        _mm512_store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int O, int P>
  static inline typename std::enable_if<P == 4, void>::type op_fma(
      elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    __m512 mmout[O][T], mmwei[O][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, O, V);

    // preload weights
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
      if (F_traits<F>::is_compact_weights) {
        MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
        mmwei[_O][0] = _mm512_load_ps(&md5(aweights5, 0, 0, 0, _O, 0));
        mmwei[_O][1] = _mm512_load_ps(&md5(aweights5, 0, 0, 1, _O, 0));
      } else {
        MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
        mmwei[_O][0] = _mm512_load_ps(&md6(aweights6, _O, 0, 0, 0, 0, 0));
        mmwei[_O][1] = _mm512_load_ps(&md6(aweights6, _O, 0, 0, 0, 1, 0));
      }
    }
    if (reset_output) {
      if (with_bias) {
        // load bias
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          __m512 tmp = _mm512_load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m512 tmp = _mm512_setzero_ps();
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
    } else {
      // load output
#pragma unroll(O)
      for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm512_load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][2] = _mm512_load_ps(&md5(aweights5, _I2, _V, 2, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][2]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 2, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 0));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
        }
#pragma unroll(O)
        // _P = 1
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][3] = _mm512_load_ps(&md5(aweights5, _I2, _V, 3, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][3]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V, 3, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 1));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][1], mmbcst, mmout[_O][_T]);
        }
        // _P = 2
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md5(aweights5, _I2, _V + 1, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0,  _V, 2));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][2], mmbcst, mmout[_O][_T]);
        }
        // _P = 3
#pragma unroll(O)
        for (int _O = 0; _O < O; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O2, V);
            mmwei[_O][1]
                = _mm512_load_ps(&md5(aweights5, _I2, _V + 1, 1, _O, 0));
          } else {
            MD6(float, aweights6, weights, O, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][1]
                = _mm512_load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 1, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m512 mmbcst
              = _mm512_broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 3));
#pragma unroll(O)
          for (int _O = 0; _O < O; ++_O)
            mmout[_O][_T]
                = _mm512_fmadd_ps(mmwei[_O][3], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
#pragma unroll(O)
    for (int _O = 0; _O < O; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        _mm512_store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int O2 = O2, int T = T>
  static inline typename std::enable_if<(J_traits<O2, T, has_Ir>::J == 1)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    op_fma<JO0, JP0>(xc, output, input, weights, bias, reset_output);
  }

  template <int O2 = O2, int T = T>
  static inline typename std::enable_if<(J_traits<O2, T, has_Ir>::J == 2)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    const int O2_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O2, O2_stride);
    MD2(float, abias, bias, O2, V);

    op_fma<JO0, JP0>(xc, output, input, weights, bias, reset_output);

    if (F_traits<F>::is_compact_weights) {
      MD4(float, aweights, weights, xc.I2, V, O2, V);
      op_fma<JO1, JP1>(xc, &md2(aoutput, JO0, 0), input,
          &md4(aweights, 0, 0, JO0, 0), &md2(abias, JO0, 0), reset_output);
    } else {
      MD3(float, aweights, weights, O2, xc.ic34, xc.I2 *V *V);
      op_fma<JO1, JP1>(xc, &md2(aoutput, JO0, 0), input,
          &md3(aweights, JO0, 0, 0), &md2(abias, JO0, 0), reset_output);
    }
  }

  template <int O2 = O2, int T = T>
  static inline typename std::enable_if<(J_traits<O2, T, has_Ir>::J == 3)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, bool reset_output)
  {
    const int O2_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, O2, O2_stride);
    MD2(float, abias, bias, O2, V);

    op_fma<JO0, JP0>(xc, output, input, weights, bias, reset_output);

    if (F_traits<F>::is_compact_weights) {
      MD4(float, aweights, weights, xc.I2, V, O2, V);
      op_fma<JO1, JP1>(xc, &md2(aoutput, JO0, 0), input,
          &md4(aweights, 0, 0, JO0, 0), &md2(abias, JO0, 0), reset_output);
    } else {
      MD3(float, aweights, weights, O2, xc.ic34, xc.I2 *V *V);
      op_fma<JO1, JP1>(xc, &md2(aoutput, JO0, 0), input,
          &md3(aweights, JO0, 0, 0), &md2(abias, JO0, 0), reset_output);
    }

    if (F_traits<F>::is_compact_weights) {
      MD4(float, aweights, weights, xc.I2, V, O2, V);
      op_fma<JO2, JP2>(xc, &md2(aoutput, JO0 + JO1, 0), input,
          &md4(aweights, 0, 0, JO0 + JO1, 0), &md2(abias, JO0 + JO1, 0),
          reset_output);
    } else {
      MD3(float, aweights, weights, O2, xc.ic34, xc.I2 *V *V);
      op_fma<JO2, JP2>(xc, &md2(aoutput, JO0 + JO1, 0), input,
          &md3(aweights, JO0 + JO1, 0, 0), &md2(abias, JO0 + JO1, 0),
          reset_output);
    }
  }
};

} // namespace euler
