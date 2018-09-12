#pragma once

#include "el_intrin.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_stl.hpp"
#include "elx_conv.hpp"

// Type: data type
// S: stride
// O: OC blocking unit
// T: tile blocking unit
// F: format
// V: vector size
// I: ISA
// has_Ir: has tailing ic

namespace euler {

// GEMM kernel format
// Input - weights - outputt
// C: compact
//    Input: I2, T, S, V
//    Weights: O1, I2, V, O, V
//    Output: O1, O, T, V
// D: discrete
//    Input: I2, ih, iw, V
//    Weights: O1, O, ic2, V, V
//    Output: O1, O, oh, ow, V
const int GKF_CCC = 0xccc;
const int GKF_CCD = 0xccd;
const int GKF_DCD = 0xdcd;
const int GKF_DDD = 0xddd;

// (weights) pipeline length
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

template <int T>
struct P_traits<1, T, false, typename std::enable_if<(T >= 31)>::type> {
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

template <int O, int T>
struct P_traits<O, T, false,
    typename std::enable_if<(O > 1 && (31 / O - T) == 1)>::type> {
  static constexpr int P = 1;
};

template <int O, int T> struct P_traits<O, T, true> {
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
      elx_conv_t<float> &, float *, float *, float *, float *, int)
  {}
};

template <int V, int... Kp>
struct gemm_kernel_otj<float, V, ISA_SKX_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 5,
      "Kernel parameters must be Type, V, I, <S, F, O, T, has_Ir>");

  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();
  constexpr static auto has_Ir = estl::get<4, bool, kparams>();

  // Jamming components
  constexpr static int J = J_traits<O, T, has_Ir>::J;
  constexpr static int JO0 = J_traits<O, T, has_Ir>::O0;
  constexpr static int JP0 = J_traits<O, T, has_Ir>::P0;
  constexpr static int JO1 = J_traits<O, T, has_Ir>::O1;
  constexpr static int JP1 = J_traits<O, T, has_Ir>::P1;
  constexpr static int JO2 = J_traits<O, T, has_Ir>::O2;
  constexpr static int JP2 = J_traits<O, T, has_Ir>::P2;

  template <int JO, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == false), void>::type
  op_fma(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    __m<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, JO, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, JO, V);

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
        // load bias
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          __m<V> tmp = _mm<V>::load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (get_attr(attr, ip_sum_idx)) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T) {
            MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
            mmout[_O][_T] = _mm<V>::add_ps(mmout[_O][_T],
                _mm<V>::load_ps(&md2(aoutput2, _T, 0)));
          }
        }
      }
    } else {
      // load output
#pragma unroll(JO)
      for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm<V>::load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][0] = _mm<V>::load_ps(&md5(aweights5, _I2, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O) {
            MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
            __m<V> mmbcst = _mm<V>::set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }
    }

    // store output
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        if (get_attr(attr, relu_idx)) {
          __m<V> zero = _mm<V>::setzero_ps();
          mmout[_O][_T] = _mm<V>::max_ps(mmout[_O][_T], zero);
        }
        if (get_attr(attr, s_output_idx))
          _mm<V>::stream_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
        else
          _mm<V>::store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == true), void>::type
  op_fma(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    __m<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, JO, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, JO, V);

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
        // load bias
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          __m<V> tmp = _mm<V>::load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (get_attr(attr, ip_sum_idx)) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T) {
            MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
            mmout[_O][_T] = _mm<V>::add_ps(mmout[_O][_T],
                _mm<V>::load_ps(&md2(aoutput2, _T, 0)));
          }
        }
      }
    } else {
      // load output
#pragma unroll(JO)
      for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm<V>::load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2 - 1; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V; ++_V) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V, 1, O, V);
            mmwei[_O][0] = _mm<V>::load_ps(&md5(aweights5, _I2, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V, 1, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O) {
            MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V, 1);
            __m<V> mmbcst = _mm<V>::set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }
    }
    // Ir
    {
#pragma nounroll
      for (int _V = 0; _V < xc.Ir; ++_V) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V, 1, O, V);
            mmwei[_O][0] = _mm<V>::load_ps(&md5(aweights5, xc.I2 - 1, _V, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V, 1, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, xc.I2 - 1, _V, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O) {
            MD4(float, ainput4, &md2(ainput, xc.I2 - 1, 0), T, S, V, 1);
            __m<V> mmbcst = _mm<V>::set1_ps(md4(ainput4, _T, 0, _V, 0));
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      }

    }

    // store output
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        if (get_attr(attr, relu_idx)) {
          __m<V> zero = _mm<V>::setzero_ps();
          mmout[_O][_T] = _mm<V>::max_ps(mmout[_O][_T], zero);
        }
        if (get_attr(attr, s_output_idx))
          _mm<V>::stream_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
        else
          _mm<V>::store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }


  template <int JO, int P>
  static inline typename std::enable_if<P == 2, void>::type op_fma(
      elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    __m<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, JO, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, JO, V);

    // preload weights
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
      if (F_traits<F>::is_compact_weights) {
        MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
        mmwei[_O][0] = _mm<V>::load_ps(&md5(aweights5, 0, 0, 0, _O, 0));
      } else {
        MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
        mmwei[_O][0] = _mm<V>::load_ps(&md6(aweights6, _O, 0, 0, 0, 0, 0));
      }
    }

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
        // load bias
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          __m<V> tmp = _mm<V>::load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (get_attr(attr, ip_sum_idx)) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T) {
            MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
            mmout[_O][_T] = _mm<V>::add_ps(mmout[_O][_T],
                _mm<V>::load_ps(&md2(aoutput2, _T, 0)));
          }
        }
      }
    } else {
      // load output
#pragma unroll(JO)
      for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm<V>::load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][1] = _mm<V>::load_ps(&md5(aweights5, _I2, _V, 1, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][1]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, 1, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 0));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
        }
        // _P = 1
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md5(aweights5, _I2, _V + 1, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 1));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][1], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        if (get_attr(attr, relu_idx)) {
          __m<V> zero = _mm<V>::setzero_ps();
          mmout[_O][_T] = _mm<V>::max_ps(mmout[_O][_T], zero);
        }
        if (get_attr(attr, s_output_idx))
          _mm<V>::stream_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
        else
          _mm<V>::store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 4, void>::type op_fma(
      elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    __m<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aoutput, output, JO, O_stride);
    MD2(float, ainput, input, xc.I2, I2_stride);
    MD2(float, abias2, bias, JO, V);

    // preload weights
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
      if (F_traits<F>::is_compact_weights) {
        MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
        mmwei[_O][0] = _mm<V>::load_ps(&md5(aweights5, 0, 0, 0, _O, 0));
        mmwei[_O][1] = _mm<V>::load_ps(&md5(aweights5, 0, 0, 1, _O, 0));
      } else {
        MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
        mmwei[_O][0] = _mm<V>::load_ps(&md6(aweights6, _O, 0, 0, 0, 0, 0));
        mmwei[_O][1] = _mm<V>::load_ps(&md6(aweights6, _O, 0, 0, 0, 1, 0));
      }
    }
    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
        // load bias
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          __m<V> tmp = _mm<V>::load_ps(&md2(abias2, _O, 0));
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O)
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (get_attr(attr, ip_sum_idx)) {
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
          for (int _T = 0; _T < T; ++_T) {
            MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
            mmout[_O][_T] = _mm<V>::add_ps(mmout[_O][_T],
                _mm<V>::load_ps(&md2(aoutput2, _T, 0)));
          }
        }
      }
    } else {
      // load output
#pragma unroll(JO)
      for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
          mmout[_O][_T] = _mm<V>::load_ps(&md2(aoutput2, _T, 0));
        }
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][2] = _mm<V>::load_ps(&md5(aweights5, _I2, _V, 2, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][2]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, 2, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 0));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
        }
#pragma unroll(JO)
        // _P = 1
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][3] = _mm<V>::load_ps(&md5(aweights5, _I2, _V, 3, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][3]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, 3, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 1));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][1], mmbcst, mmout[_O][_T]);
        }
        // _P = 2
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md5(aweights5, _I2, _V + 1, 0, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][0]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 0, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0,  _V, 2));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][2], mmbcst, mmout[_O][_T]);
        }
        // _P = 3
#pragma unroll(JO)
        for (int _O = 0; _O < JO; ++_O) {
          if (F_traits<F>::is_compact_weights) {
            MD5(float, aweights5, weights, xc.I2, V / P, P, O, V);
            mmwei[_O][1]
                = _mm<V>::load_ps(&md5(aweights5, _I2, _V + 1, 1, _O, 0));
          } else {
            MD6(float, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
            mmwei[_O][1]
                = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V + 1, 1, 0));
          }
        }
#pragma unroll(T)
        for (int _T = 0; _T < T; ++_T) {
          MD4(float, ainput4, &md2(ainput, _I2, 0), T, S, V / P, P);
          __m<V> mmbcst
              = _mm<V>::broadcastss_ps(*(__m128 *)&md4(ainput4, _T, 0, _V, 3));
#pragma unroll(JO)
          for (int _O = 0; _O < JO; ++_O)
            mmout[_O][_T]
                = _mm<V>::fmadd_ps(mmwei[_O][3], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
#pragma unroll(JO)
    for (int _O = 0; _O < JO; ++_O) {
#pragma unroll(T)
      for (int _T = 0; _T < T; ++_T) {
        MD2(float, aoutput2, &md2(aoutput, _O, 0), T, V);
        if (get_attr(attr, relu_idx)) {
          __m<V> zero = _mm<V>::setzero_ps();
          mmout[_O][_T] = _mm<V>::max_ps(mmout[_O][_T], zero);
        }
        if (get_attr(attr, s_output_idx))
          _mm<V>::stream_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
        else
          _mm<V>::store_ps(&md2(aoutput2, _T, 0), mmout[_O][_T]);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 1)
      && (F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aweights, weights, xc.O1, xc.I2 *V *O *V);
    MD2(float, aoutput, output, xc.O1, O * O_stride);
    MD2(float, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md2(aoutput, _O1, 0), input, &md2(aweights, _O1, 0),
          &md2(abias, _O1, 0), attr);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 1)
      && !(F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(float, aweights, weights, xc.O1, O * xc.IC * V);
    MD2(float, aoutput, output, xc.O1, O * O_stride);
    MD2(float, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md2(aoutput, _O1, 0), input, &md2(aweights, _O1, 0),
          &md2(abias, _O1, 0), attr);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 2)
      && (F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD4(float, aweights, weights, xc.O1, xc.I2 * V, O, V);
    MD3(float, aoutput, output, xc.O1, O, O_stride);
    MD3(float, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md4(aweights, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0), attr);
      op_fma<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md4(aweights, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0), attr);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 2)
      && !(F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD3(float, aweights, weights, xc.O1, O, xc.IC * V);
    MD3(float, aoutput, output, xc.O1, O, O_stride);
    MD3(float, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md3(aweights, _O1, 0, 0), &md3(abias, _O1, 0, 0), attr);
      op_fma<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md3(aweights, _O1, JO0, 0), &md3(abias, _O1, JO0, 0), attr);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 3)
      && (F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD4(float, aweights, weights, xc.O1, xc.I2 * V, O, V);
    MD3(float, aoutput, output, xc.O1, O, O_stride);
    MD3(float, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md4(aweights, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0), attr);
      op_fma<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md4(aweights, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0), attr);
      op_fma<JO2, JP2>(xc, &md3(aoutput, _O1, JO0 + JO1, 0), input,
          &md4(aweights, _O1, 0, JO0 + JO1, 0),
          &md3(abias, _O1, JO0 + JO1, 0), attr);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir>::J == 3)
      && !(F_traits<F>::is_compact_weights)>::type
  execute(elx_conv_t<float> &xc, float *output, float *input, float *weights,
      float *bias, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD3(float, aweights, weights, xc.O1, O, xc.IC * V);
    MD3(float, aoutput, output, xc.O1, O, O_stride);
    MD3(float, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_fma<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md3(aweights, _O1, 0, 0), &md3(abias, _O1, 0, 0), attr);
      op_fma<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md3(aweights, _O1, JO0, 0), &md3(abias, _O1, JO0, 0), attr);
      op_fma<JO2, JP2>(xc, &md3(aoutput, _O1, JO0 + JO1, 0), input,
          &md3(aweights, _O1, JO0 + JO1, 0), &md3(abias, _O1, JO0 + JO1, 0),
          attr);
    }
  }
};

struct gemm_kernel_binder {
  template <typename Type, int V, int I, int... Kp>
  using gemm_ker_cls = typename euler::gemm_kernel_otj<Type, V, I,
      estl::integer_sequence<Kp...>>;
  using ker = decltype(gemm_ker_cls<int, 1, 1, 1, 1, 1, 1, false>::execute);

  template <typename Type, int V, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, ker **func)
  {
    switch (O) {
    case 1:
      LOOP_FROM_TO(_T, 1, 32, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 1, _T, has_Ir>::execute);
      });
      if (T >= 32)
        el_error("gemm_kernel: O = 1, T >= 32 not supported");
      break;
    case 2:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 2, _T, has_Ir>::execute);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 2, T >= 15 not supported");
      break;
    case 3:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 3, _T, has_Ir>::execute);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 3, T >= 15 not supported");
      break;
    case 4:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 4, _T, has_Ir>::execute);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 4, T >= 15 not supported");
      break;
    case 5:
      LOOP_FROM_TO(_T, 1, 6, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 5, _T, has_Ir>::execute);
      });
      if (T >= 6)
        el_error("gemm_kernel: O = 5, T >= 6 not supported");
      break;
    case 6:
      LOOP_FROM_TO(_T, 1, 5, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 6, _T, has_Ir>::execute);
      });
      if (T >= 5)
        el_error("gemm_kernel: O = 6, T >= 5 not supported");
      break;
    case 7:
      LOOP_FROM_TO(_T, 1, 4, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 7, _T, has_Ir>::execute);
      });
      if (T >= 4)
        el_error("gemm_kernel: O = 7, T >= 4 not supported");
      break;
    case 8:
      LOOP_FROM_TO(_T, 1, 9, {
        if (T == _T)
          (*func = gemm_ker_cls<Type, V, I, S, F, 8, _T, has_Ir>::execute);
      });
      if (T >= 9)
        el_error("gemm_kernel: O = 8, T >= 9 not supported");
      break;
    default:
      el_error("gemm_kenrel: O > 8 unsupported");
    }
  }
};

} // namespace euler
