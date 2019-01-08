#pragma once

#include "el_intrin.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_stl.hpp"
#include "elx_conv.hpp"

// S: stride
// O: OC blocking unit
// T: tile blocking unit
// F: format
// V: vector size
// Vx: packed size of data with InputType
// I: ISA
// has_Ir: has tailing ic

namespace euler {

// GEMM kernel format
// Input - weights - output
// C: compact
//    Input: I2, T, S, V, Vx
//    Weights: O1, I2, V, O, V, Vx
//    Output: O1, O, T, V
//    factor: O1, O, V
//    weights_scale: O1, O, V
// D: discrete
//    Input: I2, ih, iw, V, Vx
//    Weights: O1, O, ic2, V, V, Vx
//    Output: O1, O, oh, ow, V
//    factor: O1, O, V
//    weights_scale: O1, O, V
// E: discrete (for nchw input)
//    Input: I2, V, ih, iw
const int GKF_CCC = 0xccc;
const int GKF_CCD = 0xccd;
const int GKF_DCD = 0xdcd;
const int GKF_DDD = 0xddd;
const int GKF_ECD = 0xecd;

#define IF
#define THEN ?
#define ELSE :

// (weights) pipeline length
//
// Wtype = fp32 || fp16:
//   O == 1: T + P <= 32
//   O > 1: O (T + P) + 1 <= 32
template <int O, int T, bool has_Ir, typename Wtype>
struct P_traits {
  static constexpr int P = IF (has_Ir) THEN (1) ELSE (
    IF (O == 1) THEN (
      IF (T <= 28) THEN (4) ELSE (
        IF (T == 29 || T == 30) THEN (2) ELSE (1)
      )
    ) ELSE (
      IF (O > 1 && (31 / O - T) >= 4) THEN (4) ELSE (
        IF (O > 1 && (31 / O - T == 2 || 31 / O - T == 3)) THEN (2) ELSE (1)
      )
    )
  );
};

// Wtype = int8_t:
//   O == 1: T + P + 1(one) + 1(t0) <= 32
//   O > 1: O (T + P) + 1(bcast) + 1(one) + 1(t0) <= 32
template <int O, int T, bool has_Ir>
struct P_traits<O, T, has_Ir, int8_t> {
  static constexpr int P = IF (has_Ir) THEN (1) ELSE (
    IF (O == 1) THEN (
      IF (T <= 26) THEN (4) ELSE (
        IF (T == 27 || T == 28) THEN (2) ELSE (1)
      )
    ) ELSE (
      IF (O > 1 && (29 / O - T) >= 4) THEN (4) ELSE (
        IF (O > 1 && (29 / O - T == 2 || 29 / O - T == 3)) THEN (2) ELSE (1)
      )
    )
  );
};

// Jamming
template <int O, int T, bool has_Ir, typename Wtype = float, typename C = void>
struct J_traits {};

template <int T, bool has_Ir, typename Wtype>
struct J_traits<8, T, has_Ir, Wtype, typename std::enable_if<T == 6>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir, typename Wtype>
struct J_traits<8, T, has_Ir, Wtype,
    typename std::enable_if<T == 7 || T == 8, void>::type> {
  static constexpr int J = 3;
  static constexpr int O0 = 3;
  static constexpr int O1 = 3;
  static constexpr int O2 = 2;
  static constexpr int P0 = P_traits<O0, T, has_Ir, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir, Wtype>::P;
  static constexpr int P2 = P_traits<O2, T, has_Ir, Wtype>::P;
};

template <int T, bool has_Ir, typename Wtype>
struct J_traits<8, T, has_Ir, Wtype,
    typename std::enable_if<(T >= 3 && T < 6), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 4;
  static constexpr int O1 = 4;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir, typename Wtype>
struct J_traits<4, T, has_Ir, Wtype,
    typename std::enable_if<(T >= 7 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 2;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int T, bool has_Ir, typename Wtype>
struct J_traits<3, T, has_Ir, Wtype,
    typename std::enable_if<(T >= 10 && T < 15), void>::type> {
  static constexpr int J = 2;
  static constexpr int O0 = 2;
  static constexpr int O1 = 1;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, Wtype>::P;
  static constexpr int P1 = P_traits<O1, T, has_Ir, Wtype>::P;
  static constexpr int P2 = 0;
};

template <int O, int T, bool has_Ir>
struct J_traits<O, T, has_Ir, float,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, float>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T, bool has_Ir>
struct J_traits<O, T, has_Ir, float16,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, float16>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int O, int T, bool has_Ir>
struct J_traits<O, T, has_Ir, int8_t,
    typename std::enable_if<((O == 1 && T < 32)) || (O == 2 && T < 15)
        || (O == 3 && T < 10) || (O == 4 && T < 7) || (O == 5 && T < 6)
        || (O == 6 && T < 5) || (O == 7 && T < 4) || (O == 8 && T < 3)>::type> {
  static constexpr int J = 1;
  static constexpr int O0 = O;
  static constexpr int O1 = 0;
  static constexpr int O2 = 0;
  static constexpr int P0 = P_traits<O0, T, has_Ir, int8_t>::P;
  static constexpr int P1 = 0;
  static constexpr int P2 = 0;
};

template <int F>
struct F_traits {
  static constexpr bool is_compact_input = (F & 0xF00) == 0xC00;
  static constexpr bool is_blocked_input = (F & 0xF00) == 0xD00;
  static constexpr bool is_nchw_input = (F & 0xF00) == 0xE00;
  static constexpr bool is_compact_weights = (F & 0xF0) == 0xC0;
  static constexpr bool is_compact_output = (F & 0xF) == 0xC;
};

template <typename GarrayTypes, int V, int Vx, int I, typename KP>
struct gemm_kernel_otj {
  static inline void gemm(
      elx_conv_params_t &, typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *, int,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *) {}

  static inline void conv(
      elx_conv_params_t &,
      typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *,
      int, int, int, int, int, int) {}
};

template <typename GarrayTypes, int V, int Vx, int ...Kp>
struct gemm_kernel_otj<GarrayTypes, V, Vx, ISA_SKX_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 5,
      "Kernel parameters must be GarrayTypes, V, Vx, I, <S, F, O, T, has_Ir>");

  using InputType = typename GarrayTypes::InputType;
  using WeightsType = typename GarrayTypes::WeightsType;
  using OutputType = typename GarrayTypes::OutputType;
  using BiasType = typename GarrayTypes::BiasType;
  using ScaleType = typename GarrayTypes::ScaleType;

  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();
  constexpr static auto K = estl::get<4, int, kparams>(); // reuse, conv
  // TODO: disable P=2/4 for K>3 due to ICC build crash
  constexpr static auto has_Ir = K == 3 ? false : estl::get<4, bool, kparams>(); // reuse, gemm

  // Jamming components
  constexpr static int J = J_traits<O, T, has_Ir, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, has_Ir, WeightsType>::O0;
  constexpr static int JP0 = J_traits<O, T, has_Ir, WeightsType>::P0;
  constexpr static int JO1 = J_traits<O, T, has_Ir, WeightsType>::O1;
  constexpr static int JP1 = J_traits<O, T, has_Ir, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, has_Ir, WeightsType>::O2;
  constexpr static int JP2 = J_traits<O, T, has_Ir, WeightsType>::P2;


  // FP32 gemm kernel
  //
  template <int JO>
  static inline __m<V> op_load_bias(BiasType *bias, const int _O)
  {
    __m<V> res;
    MD2(BiasType, abias2, bias, JO, V);
    if (std::is_same<BiasType, float>::value) {
      res = _mm<V>::load_ps(&md2(abias2, _O, 0));
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(abias2, _O, 0));
      res = _mm<V>::cvtph_ps(fp16v);
    }
    return res;
  }

  template <int JO>
  static inline __m<V> op_load_output(OutputType *output, const int _T)
  {
    __m<V> res;
    MD2(OutputType, aoutput2, output, T, V);
    if (std::is_same<OutputType, float>::value) {
      res = _mm<V>::load_ps(&md2(aoutput2, _T, 0));
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(aoutput2, _T, 0));
      res = _mm<V>::cvtph_ps(fp16v);
    }
    return res;
  }

  template <int JO, int P>
  static inline __m<V> op_load_weights(elx_conv_params_t &xc,
      WeightsType *weights, const int _I2, const int _V, const int _P, const int _O)
  {
    __m<V> res;
    if (F_traits<F>::is_compact_weights) {
      MD5(WeightsType, aweights5, weights, xc.I2, V / P, P, O, V);
      if (std::is_same<WeightsType, float>::value) {
        res = _mm<V>::load_ps(&md5(aweights5, _I2, _V, _P, _O, 0));
      } else {
        auto fp16v = _mm<V / 2>::load_si256(
            (__m256i *)&md5(aweights5, _I2, _V, _P, _O, 0));
        res = _mm<V>::cvtph_ps(fp16v);
      }
    } else {
      MD6(WeightsType, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V);
      if (std::is_same<WeightsType, float>::value) {
        res = _mm<V>::load_ps(&md6(aweights6, _O, 0, _I2, _V, _P, 0));
      } else {
        auto fp16v = _mm<V / 2>::load_si256(
            (__m256i *)&md6(aweights6, _O, 0, _I2, _V, _P, 0));
        res = _mm<V>::cvtph_ps(fp16v);
      }
    }
    return res;
  }

  template <int P>
  static inline __m<V> op_load_input(elx_conv_params_t &xc, InputType *input,
      const int _V, const int _P, const int _T)
  {
    // *Note*: xc.T vs. T:
    // T is not real T in border of direct-conv. It works okay only as
    // leading dim.
    if (F_traits<F>::is_nchw_input) {
      MD4(InputType, ainput4, input, V / P, P, xc.ih, xc.iw);
      MD3(InputType, ainput3, &md4(ainput4, _V, _P, 0, 0), xc.wt, xc.T, S);
      return _mm<V>::set1_ps(md3(ainput3, 0, _T, 0));
    } else {
      MD4(InputType, ainput4, input, T, S, V / P, P);
      return _mm<V>::set1_ps(md4(ainput4, _T, 0, _V, _P));
    }
  }

  static inline void op_store_output(
      OutputType *output, __m<V> res, const int _T, const int attr)
  {
    MD2(OutputType, aoutput2, output, T, V);

    if (get_attr(attr, relu_idx)) {
      __m<V> zero = _mm<V>::setzero_ps();
      res = _mm<V>::max_ps(res, zero);
    }
    if (get_attr(attr, s_output_idx)) {
      if (std::is_same<OutputType, float>::value) {
        _mm<V>::stream_ps(&md2(aoutput2, _T, 0), res);
      } else {
        auto fp16v = _mm<V>::cvtps_ph(
            res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::stream_si256((__m256i *)&md2(aoutput2, _T, 0), fp16v);
      }
    } else {
      if (std::is_same<OutputType, float>::value) {
        _mm<V>::store_ps(&md2(aoutput2, _T, 0), res);
      } else {
        auto fp16v = _mm<V>::cvtps_ph(
            res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)&md2(aoutput2, _T, 0), fp16v);
      }
    }
  }

  static inline void __type_check_fp32_fp16(
      OutputType *, InputType *, WeightsType *, BiasType *)
  {
    static_assert(
        std::is_same<InputType, float>::value, "only fp32 input type");
    static_assert(std::is_same<WeightsType, float>::value
            || std::is_same<WeightsType, float16>::value,
        "only fp32/fp16 weights type");
    static_assert(std::is_same<OutputType, float>::value
            || std::is_same<OutputType, float16>::value,
        "only fp32/fp16 output type");
    static_assert(std::is_same<BiasType, float>::value
            || std::is_same<BiasType, float16>::value,
        "only fp32/fp16 bias type");
  }

  template <int JO>
  static inline void op_setup_output(elx_conv_params_t &xc, OutputType *output,
      BiasType *bias, __m<V> mmout[JO][T], int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
        // load bias
        unroll_for (_O, JO) {
          unroll_for (_T, T)
            mmout[_O][_T] = op_load_bias<JO>(bias, _O);
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
        unroll_for (_O, JO)
          unroll_for (_T, T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (get_attr(attr, ip_sum_idx)) {
        unroll_for (_O, JO) {
          unroll_for (_T, T)
            mmout[_O][_T] += op_load_output<JO>(&md2(aoutput, _O, 0), _T);
        }
      }
    } else {
      // load output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(&md2(aoutput, _O, 0), _T);
      }
    }
 
  }

  template <int JO, int P>
  static inline typename std::enable_if<
      !std::is_same<InputType, uint8_t>::value
      && (P == 1 && has_Ir == false), void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias,
      int attr, ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __type_check_fp32_fp16(output, input, weights, bias);

    __m<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(InputType, ainput, input, xc.I2, I2_stride);
    MD2(BiasType, abias2, bias, JO, V);

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[0][_O], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<
      !std::is_same<InputType, uint8_t>::value
      && (P == 1 && has_Ir == true), void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias,
      int attr, ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __type_check_fp32_fp16(output, input, weights, bias);

    __m<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(InputType, ainput, input, xc.I2, I2_stride);
    MD2(BiasType, abias2, bias, JO, V);

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _I2 = 0; _I2 < xc.I2 - 1; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[0][_O], mmbcst, mmout[_O][_T]);
        }
      }
    }
    // Ir
    {
#pragma nounroll
      for (int _V = 0; _V < xc.Ir; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, xc.I2 - 1, _V, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, xc.I2 - 1, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[0][_O], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }


  template <int JO, int P>
  static inline typename std::enable_if<
      !std::is_same<InputType, uint8_t>::value && P == 2, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias,
      int attr, ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __type_check_fp32_fp16(output, input, weights, bias);

    __m<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(InputType, ainput, input, xc.I2, I2_stride);
    MD2(BiasType, abias2, bias, JO, V);

    // preload weights
    unroll_for (_O, JO)
      mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
        unroll_for (_O, JO)
          mmwei[1][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 1, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[0][_O], mmbcst, mmout[_O][_T]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[1][_O], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<
      !std::is_same<InputType, uint8_t>::value && P == 4, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias,
      int attr, ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __type_check_fp32_fp16(output, input, weights, bias);

    __m<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V : xc.ih * xc.iw * V;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(InputType, ainput, input, xc.I2, I2_stride);
    MD2(BiasType, abias2, bias, JO, V);

    // preload weights
    unroll_for (_O, JO) {
      mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);
      mmwei[1][_O] = op_load_weights<JO, P>(xc, weights, 0, 0, 1, _O);
    }

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
        unroll_for (_O, JO)
          mmwei[2][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 2, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[0][_O], mmbcst, mmout[_O][_T]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[3][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 3, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[1][_O], mmbcst, mmout[_O][_T]);
        }
        // _P = 2
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 2, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[2][_O], mmbcst, mmout[_O][_T]);
        }
        // _P = 3
        unroll_for (_O, JO)
          mmwei[1][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 1, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, &md2(ainput, _I2, 0), _V, 3, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[3][_O], mmbcst, mmout[_O][_T]);
        }
      }
    }
    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  // INT8 gemm kernel
  //
  static inline __i<V> op_int8_fma(__i<V>& out, __i<V>& a, __i<V>& b) {
    // TODO: check ISA
#if defined(WITH_VNNI)
    out = _mm512_dpbusds_epi32(out, a, b);
#else
    __i<V> one = _mm<V>::set1_epi16(1);
    __i<V> t0 = _mm<V>::maddubs_epi16(a, b);
    t0 = _mm<V>::madd_epi16(t0, one);
    out = _mm<V>::add_epi32(t0, out);
#endif
    return out;
  }

  static inline __i<V> op_int8_load_output(OutputType *output, const int _T)
  {
    MD2(OutputType, aoutput2, output, T, V);
    if (std::is_same<OutputType, float>::value) {
      return _mm<V>::load_epi32((__i<V> *)&md2(aoutput2, _T, 0));
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__i<V/2> *)&md2(aoutput2, _T, 0));
      return _mm<V>::cvtepi16_epi32(fp16v);
    }
  }

  template <const int P>
  static inline __i<V> op_int8_load_input(
      uint8_t *input, const int _V, const int _P, const int _T)
  {
    MD5(uint8_t, ainput5, input, T, S, V / P, P, Vx);
    return _mm<V>::set1_epi32(*(int32_t*)&md5(ainput5, _T, 0, _V, _P, 0));
  }

  template <const int JO, const int P>
  static inline __i<V> op_int8_load_weights(elx_conv_params_t &xc,
      int8_t *weights, const int _I2, const int _V, const int _P, const int _O)
  {
    __i<V> res;
    if (F_traits<F>::is_compact_weights) {
      MD5(int8_t, aweights5, weights, xc.I2, V / P, P, O, V * Vx);
      res = _mm<V>::load_epi32(&md5(aweights5, _I2, _V, _P, _O, 0));
    } else {
      MD6(int8_t, aweights6, weights, JO, xc.ic34, xc.I2, V / P, P, V * Vx);
      res = _mm<V>::load_epi32(&md6(aweights6, _O, 0, _I2, _V, _P, 0));
    }
    return res;
  }

  static inline void op_int8_store_output(
      OutputType *output, __i<V> res, const int _T)
  {
    if (std::is_same<OutputType, float>::value) {
      MD2(int, aoutput2, output, T, V);
      _mm<V>::store_epi32(&md2(aoutput2, _T, 0), res);
    } else {
      MD2(OutputType, aoutput2, output, T, V);
      _mm<V / 2>::store_si256(
          (__i<V / 2> *)&md2(aoutput2, _T, 0), _mm<V>::cvtepi32_epi16(res));
    }
  }

  template <const int JO>
  static inline void op_int8_restore_output(elx_conv_params_t &xc,
      OutputType *output, BiasType *bias, __i<V> res, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale,
      ScaleType *weights_factor, const int _O1, const int _O0, const int _O,
      const int _T, const int attr)
  {
    MD2(OutputType, aoutput2, output, T, V);
    MD3(float, aweights_scale3, weights_scale, xc.O1, O, V);
    MD2(float, aweights_scale, &md3(aweights_scale3, _O1, _O0, 0), JO, V);
    MD3(float, aweights_factor3, weights_factor, xc.O1, O, V);
    MD2(float, aweights_factor, &md3(aweights_factor3, _O1, _O0, 0), JO, V);

    __m<V> coeffi = _mm<V>::broadcastss_ps(*(__m128 *)&src_scale[_T]);
    coeffi = _mm<V>::mul_ps(*(__m<V> *)&md2(aweights_scale, _O, 0), coeffi);
    __m<V> ffactor = _mm<V>::broadcastss_ps(*(__m128 *)&src_factor[_T]);
    ffactor = _mm<V>::mul_ps(ffactor, *(__m<V> *)&md2(aweights_factor, _O, 0));
    __m<V> fout = _mm<V>::cvtepi32_ps(res);
    fout = _mm<V>::fmadd_ps(fout, coeffi, ffactor);

    // toutput lazy accumulation
    if (!get_attr(attr, r_output_idx) && get_attr(attr, l_output_idx)) {
      if (std::is_same<OutputType, float>::value)
        fout = _mm<V>::add_ps(fout, _mm<V>::load_ps(&md2(aoutput2, _T, 0)));
      else {
        auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(aoutput2, _T, 0));
        fout = _mm<V>::add_ps(fout, _mm<V>::cvtph_ps(fp16v));
      }
    }
    // 1. add bias (direct conv 1x1)
    if (get_attr(attr, bias_idx)) {
      MD2(BiasType, abias2, bias, JO, V);
      if (std::is_same<BiasType, float>::value) {
        fout = _mm<V>::add_ps(fout, _mm<V>::load_ps(&md2(abias2, _O, 0)));
      } else {
        auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(abias2, _O, 0));
        fout = _mm<V>::add_ps(fout, _mm<V>::cvtph_ps(fp16v));
      }
    }
    // 2. fuse relu (direct conv 1x1)
    if (get_attr(attr, relu_idx)) {
      fout = _mm<V>::max_ps(fout, _mm<V>::setzero_ps());
    }
    // 3. store output
    if (std::is_same<OutputType, float>::value)
      _mm<V>::store_ps(&md2(aoutput2, _T, 0), fout);
    else {
      auto fp16v = _mm<V>::cvtps_ph(
          fout, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      _mm<V / 2>::store_si256((__m256i *)&md2(aoutput2, _T, 0), fp16v);
    }
  }

  // u8s8f32 fma
  template <int JO, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == false), void>::type
  op_gemm(elx_conv_params_t &xc, OutputType *output, uint8_t *input,
      int8_t *weights, BiasType *bias, int attr, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale,
      ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx : xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    if (get_attr(attr, r_output_idx) || get_attr(attr, l_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
        unroll_for (_T, T)
          mmout[_O][_T] = tmp;
    } else {
      // load output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_int8_load_output(&md2(aoutput, _O, 0), _T);
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[0][_O]);
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_restore_output<JO>(xc, &md2(aoutput, _O, 0), bias,
              mmout[_O][_T], src_scale, src_factor, weights_scale,
              weights_factor, _O1, _O0, _O, _T, attr);
      }
    } else {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T);
      }
    }
  }

  // TODO: handling V and Vx tail
  template <int JO, int P>
  static inline typename std::enable_if<(P == 1 && has_Ir == true), void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx: xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    if (get_attr(attr, r_output_idx) || get_attr(attr, l_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else {
      // load output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_int8_load_output(&md2(aoutput, _O, 0), _T);
      }
    }

    for (int _I2 = 0; _I2 < xc.I2 - 1; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[0][_O]);
        }
      }
    }
    // Ir
    {
#pragma nounroll
      for (int _V = 0; _V < xc.Ir; ++_V) {
        unroll_for (_O, JO)
          mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, xc.I2 - 1, _V, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, xc.I2 - 1, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[0][_O]);
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_restore_output<JO>(xc, &md2(aoutput, _O, 0), bias,
              mmout[_O][_T], src_scale, src_factor, weights_scale,
              weights_factor, _O1, _O0, _O, _T, attr);
      }
    } else {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T);
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 2, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx: xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    // preload weights
    unroll_for (_O, JO)
      mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);

    if (get_attr(attr, r_output_idx) || get_attr(attr, l_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else {
      // load output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_int8_load_output(&md2(aoutput, _O, 0), _T);
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
        unroll_for (_O, JO)
          mmwei[1][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 1, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[0][_O]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[1][_O]);
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_restore_output<JO>(xc, &md2(aoutput, _O, 0), bias,
              mmout[_O][_T], src_scale, src_factor, weights_scale,
              weights_factor, _O1, _O0, _O, _T, attr);
      }
    } else {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T);
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 4, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[P][JO];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx: xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    // preload weights
    unroll_for (_O, JO) {
      mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);
      mmwei[1][_O] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 1, _O);
    }

    if (get_attr(attr, r_output_idx) || get_attr(attr, l_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else {
      // load output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_int8_load_output(&md2(aoutput, _O, 0), _T);
      }
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        // _P = 0
        unroll_for (_O, JO)
          mmwei[2][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 2, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[0][_O]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[3][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 3, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO) {
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[1][_O]);
          }
        }
        // _P = 2
        unroll_for (_O, JO)
          mmwei[0][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 2, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[2][_O]);
        }
        // _P = 3
        unroll_for (_O, JO)
          mmwei[1][_O] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 1, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 3, _T);
          unroll_for (_O, JO) {
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[3][_O]);
          }
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
        unroll_for (_T, T) {
          op_int8_restore_output<JO>(xc, &md2(aoutput, _O, 0), bias,
              mmout[_O][_T], src_scale, src_factor, weights_scale,
              weights_factor, _O1, _O0, _O, _T, attr);
        }
      }
    } else {
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          op_int8_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 1)
      && (F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(WeightsType, aweights, weights, xc.O1, xc.I2 * V * O * V * Vx);
    MD2(OutputType, aoutput, output, xc.O1, O * O_stride);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md2(aoutput, _O1, 0), input, &md2(aweights, _O1, 0),
          &md2(abias, _O1, 0), attr, src_scale, src_factor,
          weights_scale, weights_factor, _O1, 0);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 1)
      && !(F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(WeightsType, aweights, weights, xc.O1, O * xc.IC * V);
    MD2(OutputType, aoutput, output, xc.O1, O * O_stride);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md2(aoutput, _O1, 0), input, &md2(aweights, _O1, 0),
          &md2(abias, _O1, 0), attr, src_scale, src_factor,
          weights_scale, weights_factor, _O1, 0);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 2)
      && (F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD4(WeightsType, aweights, weights, xc.O1, xc.I2 * V, O, V * Vx);
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md4(aweights, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, 0);
      op_gemm<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md4(aweights, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 2)
      && !(F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD3(WeightsType, aweights, weights, xc.O1, O, xc.IC * V);
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md3(aweights, _O1, 0, 0), &md3(abias, _O1, 0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, 0);
      op_gemm<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md3(aweights, _O1, JO0, 0), &md3(abias, _O1, JO0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 3)
      && (F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD4(WeightsType, aweights, weights, xc.O1, xc.I2 * V, O, V * Vx);
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md4(aweights, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, 0);
      op_gemm<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md4(aweights, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0);
      op_gemm<JO2, JP2>(xc, &md3(aoutput, _O1, JO0 + JO1, 0), input,
          &md4(aweights, _O1, 0, JO0 + JO1, 0),
          &md3(abias, _O1, JO0 + JO1, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0 + JO1);
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 3)
      && !(F_traits<F>::is_compact_weights)>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD3(WeightsType, aweights, weights, xc.O1, O, xc.IC * V);
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_gemm<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md3(aweights, _O1, 0, 0), &md3(abias, _O1, 0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, 0);
      op_gemm<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md3(aweights, _O1, JO0, 0), &md3(abias, _O1, JO0, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0);
      op_gemm<JO2, JP2>(xc, &md3(aoutput, _O1, JO0 + JO1, 0), input,
          &md3(aweights, _O1, JO0 + JO1, 0), &md3(abias, _O1, JO0 + JO1, 0),
          attr, src_scale, src_factor, weights_scale, weights_factor, _O1, JO0 + JO1);
    }
  }

  // FP32 convolution kernel

  // overload load_input: support only nchw or blocked format
  template <int P>
  static inline __m<V> op_load_input(elx_conv_params_t &xc, InputType *input,
      const int _ih, const int _iw, const int _V, const int _P, const int _T)
  {
    if (F_traits<F>::is_nchw_input) {
      MD4(InputType, ainput4, input, V / P, P, xc.ih, xc.iw);
      MD3(InputType, ainput3, &md4(ainput4, _V, _P, _ih, _iw), xc.wt, T, S);
      return _mm<V>::set1_ps(md3(ainput3, 0, _T, 0));
    } else {
      MD3(InputType, ainput3, input, xc.ih, xc.iw, V); // blocked
      MD4(InputType, ainput4, &md3(ainput3, _ih, _iw, 0), T, S, V / P, P);
      return _mm<V>::set1_ps(md4(ainput4, _T, 0, _V, _P));
    }
  }

  template <int JO, int P, int FROM, int TO>
  static inline typename std::enable_if<P == 1, void>::type op_gemm_OVT(
      elx_conv_params_t &xc, InputType *input, WeightsType *weights,
      __m<V> mmout[JO][T], __m<V> mmwei[JO], int V_, int _kh, int _kw,
      int _I2)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

#pragma nounroll
    for (int _V = 0; _V < V_; ++_V) {
      unroll_for (_O, JO)
        mmwei[_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
      unroll_from_to (_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 0, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[_O] * mmbcst;
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 1, void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias, int _wt,
      int khs, int khe, int kws, int kwe, int attr)
  {
    constexpr int AKW = K / 2;
    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    // TODO: non-blocked handling
    MD2(InputType, ainput, input, xc.I2, xc.ih * xc.iw * V); // blocked or nchw
    MD3(WeightsType, aweights, weights, xc.kh, xc.kw, xc.oc3 * xc.ic3 * xc.O1 * xc.I2 * V * O * V); // compact
    MD2(OutputType, aoutput, output, JO, xc.oh * xc.ow * V); // blocked

    __m<V> mmout[JO][T], mmwei[JO];

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _kh = khs; _kh < khe; ++_kh) {
      for (int _I2 = 0; _I2 < I2; ++_I2) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          op_gemm_OVT<JO, P, 0, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // left
        if (_wt == 0) {
          int _kw = 0; // K = 3, 5, 7
          op_gemm_OVT<JO, P, AKW, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          if (K > 3) {
            _kw = 1; // K = 5, 7
            op_gemm_OVT<JO, P, AKW-1, T>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = 2; // K = 7
            op_gemm_OVT<JO, P, AKW-2, T>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          }
        }
        // right
        if (_wt == xc.wt - 1) {
          int _kw = K - 1; // K = 3, 5, 7
          op_gemm_OVT<JO, P, 0, T-AKW>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          if (K > 3) {
            _kw = K - 2; // K = 5, 7
            op_gemm_OVT<JO, P, 0, T-(AKW-1)>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = K - 3; // K = 7
            op_gemm_OVT<JO, P, 0, T-(AKW-2)>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
          }
        }
      } // I2 loop

      // Ir
      if (Ir > 0) {
        int _I2 = xc.I2 - 1;
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          op_gemm_OVT<JO, P, 0, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
        }
        // left
        if (_wt == 0) {
          int _kw = 0; // K = 3, 5, 7
          op_gemm_OVT<JO, P, AKW, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          if (K > 3) {
            _kw = 1; // K = 5, 7
            op_gemm_OVT<JO, P, AKW-1, T>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = 2; // K = 7
            op_gemm_OVT<JO, P, AKW-2, T>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          }
        }
        // right
        if (_wt == xc.wt - 1) {
          int _kw = K - 1; // K = 3, 5, 7
          op_gemm_OVT<JO, P, 0, T-AKW>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          if (K > 3) {
            _kw = K - 2; // K = 5, 7
            op_gemm_OVT<JO, P, 0, T-(AKW-1)>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = K - 3; // K = 7
            op_gemm_OVT<JO, P, 0, T-(AKW-2)>(xc, &md2(ainput, _I2, 0),
                &md3(aweights, _kh, _kw, 0), mmout, mmwei, Ir, _kh, _kw, _I2);
          }
        }
      }
    }

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  template <int JO, int P, int FROM, int TO>
  static inline typename std::enable_if<P == 2, void>::type op_gemm_OVT(
      elx_conv_params_t &xc, InputType *input, WeightsType *weights,
      __m<V> mmout[JO][T], __m<V> mmwei[P][JO], int V_, int _kh, int _kw,
      int _I2)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

#pragma nounroll
    for (int _V = 0; _V < V_ / P; ++_V) {
      // _P = 0
      unroll_for (_O, JO)
        mmwei[1][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 1, _O);
      unroll_from_to(_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 0, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[0][_O] * mmbcst;
      }
      // _P = 1
      unroll_for (_O, JO)
        mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
      unroll_from_to(_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 1, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[1][_O] * mmbcst;
      }
    }
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 2, void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias, int _wt,
      int khs, int khe, int kws, int kwe, int attr)
  {
    // 3x3 conv
    constexpr int AKW = K / 2;
    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    MD2(InputType, ainput, input, xc.I2, xc.ih * xc.iw * V); // blocked or nchw
    MD3(WeightsType, aweights, weights, xc.kh, xc.kw, xc.oc3 * xc.ic3 * xc.O1 * xc.I2 * V * O * V); // compact
    MD2(OutputType, aoutput, output, JO, xc.oh * xc.ow * V); // blocked

    __m<V> mmout[JO][T], mmwei[P][JO];

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _kh = khs; _kh < khe; ++_kh) {
      // mid
      for (int _kw = kws; _kw < kwe; ++_kw) {
        // preload weights
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, 0, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // Ir
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, 0, T>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // _kw loop, mid

      // left
      if (_wt == 0) {
        int _kw = 0;
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, AKW, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // Ir
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, AKW, T>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // left

      // right
      if (_wt == xc.wt - 1) {
        int _kw = 2;
        unroll_for (_O, JO)
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, 0, T - AKW>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // Ir
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, 0, T - AKW>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // right
    } // _kh loop

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  template <int JO, int P, int FROM, int TO>
  static inline typename std::enable_if<P == 4, void>::type op_gemm_OVT(
      elx_conv_params_t &xc, InputType *input, WeightsType *weights,
      __m<V> mmout[JO][T], __m<V> mmwei[P][JO], int V_, int _kh, int _kw,
      int _I2)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

#pragma nounroll
    for (int _V = 0; _V < V_ / P; ++_V) {
      // _P = 0
      unroll_for (_O, JO)
        mmwei[2][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 2, _O);
      unroll_from_to (_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 0, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[0][_O] * mmbcst;
      }
      // _P = 1
      unroll_for (_O, JO)
        mmwei[3][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V, 3, _O);
      unroll_from_to (_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 1, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[1][_O] * mmbcst;
      }
      // _P = 2
      unroll_for (_O, JO)
        mmwei[0][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
      unroll_from_to (_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 2, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[2][_O] * mmbcst;
      }
      // _P = 3
      unroll_for (_O, JO)
        mmwei[1][_O] = op_load_weights<JO, P>(xc, weights, _I2, _V + 1, 1, _O);
      unroll_from_to (_T, FROM, TO) {
        __m<V> mmbcst
            = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _V, 3, _T);
        unroll_for (_O, JO)
          mmout[_O][_T] += mmwei[3][_O] * mmbcst;
      }
    } // _V
  }

  template <int JO, int P>
  static inline typename std::enable_if<P == 4, void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias, int _wt,
      int khs, int khe, int kws, int kwe, int attr)
  {
    // 3x3 conv
    constexpr int AKW = K / 2;
    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    MD2(InputType, ainput, input, xc.I2, xc.ih * xc.iw * V); // blocked or nchw
    MD3(WeightsType, aweights, weights, xc.kh, xc.kw, xc.oc3 * xc.ic3 * xc.O1 * xc.I2 * V * O * V); // compact
    MD2(OutputType, aoutput, output, JO, xc.oh * xc.ow * V); // blocked

    __m<V> mmout[JO][T], mmwei[P][JO];

    op_setup_output<JO>(xc, output, bias, mmout, attr);

    for (int _kh = khs; _kh < khe; ++_kh) {
      // mid
      for (int _kw = kws; _kw < kwe; ++_kw) {
        // preload weights
        unroll_for (_O, JO) {
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
          mmwei[1][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 1, _O);
        }
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, 0, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // Ir
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, 0, T>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // _kw loop, mid

      // left
      if (_wt == 0) {
        int _kw = 0;
        unroll_for (_O, JO) {
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
          mmwei[1][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 1, _O);
        }
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, AKW, T>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        // Ir
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, AKW, T>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // left

      // right
      if (_wt == xc.wt - 1) {
        int _kw = 2;
        unroll_for (_O, JO) {
          mmwei[0][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 0, _O);
          mmwei[1][_O] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, 1, _O);
        }
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          op_gemm_OVT<JO, P, 0, T - AKW>(xc, &md2(ainput, _I2, 0),
              &md3(aweights, _kh, _kw, 0), mmout, mmwei, V, _kh, _kw, _I2);
        }
        int _I2 = xc.I2 - 1;
        op_gemm_OVT<JO, 1, 0, T - AKW>(xc, &md2(ainput, _I2, 0),
            &md3(aweights, _kh, _kw, 0), mmout, mmwei[0], Ir, _kh, _kw, _I2);
      } // right
    } // _kh loop

    // store output
    unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T, attr);
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 1)
          && (F_traits<F>::is_compact_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int _wt, int khs, int khe,
          int kws, int kwe, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD3(WeightsType, aweights, weights, xc.kh * xc.kw * xc.oc3 * xc.ic3, xc.O1,
        xc.I2 * V * O * V * Vx); // compact
    MD2(OutputType, aoutput, output, xc.O1, O * O_stride);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_conv<JO0, JP0>(xc, &md2(aoutput, _O1, 0), input,
          &md3(aweights, 0, _O1, 0), &md2(abias, _O1, 0), _wt, khs,
          khe, kws, kwe, attr);
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 2)
          && (F_traits<F>::is_compact_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int _wt, int khs, int khe,
          int kws, int kwe, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD5(WeightsType, aweights, weights, xc.kh * xc.kw * xc.oc3 * xc.ic3, xc.O1,
        xc.I2 * V, O, V * Vx); // compact
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_conv<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md5(aweights, 0, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0), _wt, khs,
          khe, kws, kwe, attr);
      op_conv<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0), _wt, khs,
          khe, kws, kwe, attr);
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, has_Ir, WeightsType>::J == 3)
          && (F_traits<F>::is_compact_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int _wt, int khs, int khe,
          int kws, int kwe, int attr)
  {
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD5(WeightsType, aweights, weights, xc.kh * xc.kw * xc.oc3 * xc.ic3, xc.O1,
        xc.I2 * V, O, V * Vx); // compact
    MD3(OutputType, aoutput, output, xc.O1, O, O_stride);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      op_conv<JO0, JP0>(xc, &md3(aoutput, _O1, 0, 0), input,
          &md5(aweights, 0, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0), _wt, khs,
          khe, kws, kwe, attr);
      op_conv<JO1, JP1>(xc, &md3(aoutput, _O1, JO0, 0), input,
          &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0), _wt, khs,
          khe, kws, kwe, attr);
      op_conv<JO2, JP2>(xc, &md3(aoutput, _O1, JO0 + JO1, 0), input,
          &md5(aweights, 0, _O1, 0, JO0 + JO1, 0), &md3(abias, _O1, JO0 + JO1, 0),
          _wt, khs, khe, kws, kwe, attr);
    }
  }

};


} // namespace euler
