#pragma once

#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "el_stl.hpp"
#include "elx_conv.hpp"
#include "elk_gemm_traits.hxx"

// S: stride
// O: OC blocking unit
// T: tile blocking unit
// F: format
// V: vector size
// Vx: packed size of data with InputType
// I: ISA
// K: kernel size

namespace euler {

template <typename GarrayTypes, int V, int Vx, int I, typename KP>
struct gemm_kernel {
  static inline void gemm(
      elx_param_t &, typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *, int) {}
};

template <typename GarrayTypes, int V, int Vx, int ...Kp>
struct gemm_kernel<GarrayTypes, V, Vx, ISA_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 5,
      "Kernel parameters must be GarrayTypes, V, Vx, I, <S, F, O, T, K>");

  using InputType = typename GarrayTypes::InputType;
  using WeightsType = typename GarrayTypes::WeightsType;
  using OutputType = typename GarrayTypes::OutputType;
  using BiasType = typename GarrayTypes::BiasType;

  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();

  // Loop splitting
  constexpr static int J   = J_traits<O, T, K_GEMM, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, K_GEMM, WeightsType>::O0;
  constexpr static int JP0 = J_traits<O, T, K_GEMM, WeightsType>::P0;
  constexpr static int JO1 = J_traits<O, T, K_GEMM, WeightsType>::O1;
  constexpr static int JP1 = J_traits<O, T, K_GEMM, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, K_GEMM, WeightsType>::O2;
  constexpr static int JP2 = J_traits<O, T, K_GEMM, WeightsType>::P2;


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
  static inline __m<V> op_load_bias(BiasType *bias, __mmask16 k, const int _O)
  {
    __m<V> res;
    MD2(BiasType, abias2, bias, JO, V);
    assert(F_traits<F>::is_nhwc_output);
    if (std::is_same<BiasType, float>::value) {
      res = _mm512_maskz_load_ps(k, &md2(abias2, _O, 0));
    } else {
      // TODO: fp16 Or
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(abias2, _O, 0));
      res = _mm<V>::cvtph_ps(fp16v);
    }
    return res;
  }

  template <int JO>
  static inline __m<V> op_load_output(elx_param_t &ep, OutputType *output,
                                      const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, ep.oh * ep.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, ep.g, ep.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), ep.O4 * ep.O3 * ep.O1, ep.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);
    __m<V> res;
    if (std::is_same<OutputType, float>::value) {
      res = _mm<V>::load_ps(aout);
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)aout);
      res = _mm<V>::cvtph_ps(fp16v);
    }
    return res;
  }

  template <int JO>
  static inline __m<V> op_load_output(elx_param_t &ep, OutputType *output,
                                      __mmask16 k, const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, ep.oh * ep.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, ep.g, ep.oc);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0), ep.O4 * ep.O3 * ep.O1, ep.O, V);
    assert(F_traits<F>::is_nhwc_output);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);
    __m<V> res;
    if (std::is_same<OutputType, float>::value) {
      res = _mm512_maskz_load_ps(k, aout);
    } else {
      // TODO
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)aout);
      res = _mm<V>::cvtph_ps(fp16v);
    }
    return res;
  }

  // Compiler warning when use #unroll(n), use #unroll as workaround
  template <int JO, int P>
  static inline __m<V> op_load_weights(elx_param_t &ep,
      WeightsType *weights, const int _I2, const int _V, const int _P, const int _O)
  {
    __m<V> res;
    if (F_traits<F>::is_compact_weights) {
      MD5(WeightsType, aweights5, weights, ep.I2, V / P, P, O, V);
      if (std::is_same<WeightsType, float>::value) {
        res = _mm<V>::load_ps(&md5(aweights5, _I2, _V, _P, _O, 0));
      } else {
        if (O == 2) { // bf16 type weights
          res = (_O == 0)
              ? _mm<V>::load_ps(&md5(aweights5, _I2, _V, _P, 0, 0))
              : _mm<V>::loadu_ps(&md5(aweights5, _I2, _V, _P, 0, 0) - 1);
        } else {      // fp16 type weights
          auto fp16v = _mm<V / 2>::load_si256(
              (__m256i *)&md5(aweights5, _I2, _V, _P, _O, 0));
          res = _mm<V>::cvtph_ps(fp16v);
        }
      }
    } else {
      MD6(WeightsType, aweights6, weights, JO, ep.ic34, ep.I2, V / P, P, V);
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
  static inline __m<V> op_load_input(elx_param_t &ep, InputType *input,
      const int _I2, const int _V, const int _P, const int _T)
  {
    // For bf16 type, considering performance, also load 32 bits
    // while the low 16 bits are neighbor values instead of zeros

    if (F_traits<F>::is_compact_input) {
      MD5(InputType, ainput0, input, ep.I2, T, S, V / P, P);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::set1_ps(md5(ainput0, _I2, _T, 0, _V, _P));
      else
        return _mm<V>::set1_ps(*(float *)(&md5(ainput0, _I2, _T, 0, _V, _P) - 1));
    } else if (F_traits<F>::is_nhwc_input) {
      MD5(InputType, ainput0, input, ep.wt, T, S, ep.g, ep.ic);
      MD5(InputType, ainput1, &md5(ainput0, 0, _T, 0, 0, 0), ep.I4, ep.I3, ep.I2, V/P, P);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::set1_ps(md5(ainput1, 0, 0, _I2, _V, _P));
      else
        return _mm<V>::set1_ps(*(float *)(&md5(ainput1, 0, 0, _I2, _V, _P) - 1));
    } else if (F_traits<F>::is_nchw_input) {
      // *Note*: ep.T vs. T:
      // T is not real T in border of direct-conv. It works okay only as
      // leading dim.
      MD5(InputType, ainput0, input, ep.I2, V / P, P, ep.ih, ep.iw);
      MD3(InputType, ainput1, &md5(ainput0, _I2, _V, _P, 0, 0), ep.wt, ep.T, S);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::set1_ps(md3(ainput1, 0, _T, 0));
      else
        return _mm<V>::set1_ps(*(float *)(&md3(ainput1, 0, _T, 0) - 1));
    } else { // blocked
      MD3(InputType, ainput0, input, ep.I2, ep.ih * ep.iw, V);
      MD4(InputType, ainput1, &md3(ainput0, _I2, 0, 0), T, S, V / P, P);
      if (std::is_same<InputType, float>::value)
        return _mm<V>::set1_ps(md4(ainput1, _T, 0, _V, _P));
      else
        return _mm<V>::set1_ps(*(float *)(&md4(ainput1, _T, 0, _V, _P) - 1));
    }
  }

  template <int JO>
  static inline void op_store_output(elx_param_t &ep,
      OutputType *output, __m<V> res, const int _O, const int _T, const int attr)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, ep.oh * ep.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, ep.g, ep.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), ep.O4 * ep.O3 * ep.O1, ep.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);

    if (test_bit(attr, AT_RELU_MASK)) {
      auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
      res = _mm<V>::max_ps(res, lower);
      res = _mm<V>::min_ps(res, upper);
    }
    if (test_bit(attr, AT_STREAMING_OUTPUT_MASK)) {
      if (std::is_same<OutputType, float>::value) {
        _mm<V>::stream_ps(aout, res);
      } else {
        auto fp16v = _mm<V>::cvtps_ph(
            res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::stream_si256((__m256i *)aout, fp16v);
      }
    } else {
      if (std::is_same<OutputType, float>::value) {
        _mm<V>::store_ps(aout, res);
      } else {
        auto fp16v = _mm<V>::cvtps_ph(
            res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)aout, fp16v);
      }
    }
  }

  template <int JO>
  static inline void op_store_output(elx_param_t &ep, OutputType *output,
      __m<V> res, __mmask16 k, const int _O, const int _T, const int attr)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, ep.oh * ep.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, ep.g, ep.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), ep.O4 * ep.O3 * ep.O1, ep.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);
    assert(F_traits<F>::is_nhwc_output);

    if (test_bit(attr, AT_RELU_MASK)) {
      auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
      res = _mm<V>::max_ps(res, lower);
      res = _mm<V>::min_ps(res, upper);
    }
    if (std::is_same<OutputType, float>::value) {
      _mm512_mask_store_ps(aout, k, res);
    } else {
      // TODO: maskstore
      auto fp16v = _mm<V>::cvtps_ph(
          res, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      _mm<V / 2>::store_si256((__m256i *)aout, fp16v);
    }
  }

  static inline void __type_check_fp32_fp16(
      OutputType *, InputType *, WeightsType *, BiasType *)
  {
    static_assert(std::is_same<InputType, float>::value
            || std::is_same<InputType, float16>::value,
        "only fp32/bf16 input type");
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

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<(P == 1 || P == 2 || P == 4), void>::type
  op_gemm(elx_param_t &ep,
      OutputType *output, InputType *input, WeightsType *weights, BiasType *bias,
      int attr, int _O1, int _O0)
  {
    __type_check_fp32_fp16(output, input, weights, bias);

    __m<V> mmout[JO][T], mmwei[JO][P];

    int I2 = ep.I2, Ir = 0;
    if (test_bit(attr, AT_Ir_MASK)) {
      I2 = ep.I2 - 1;
      Ir = ep.Ir;
    }

    // preload weights
    unroll_for (_P, P) {
      unroll_auto (_O, JO) {
        mmwei[_O][_P] = op_load_weights<JO, P>(ep, weights, 0, 0, _P, _O);
      }
    }

    __mmask16 k = _cvtu32_mask16(ep.ormask);

    if (test_bit(attr, AT_CLEAR_OUTPUT_MASK)) {
      if (test_bit(attr, AT_BIAS_MASK)) {
        // load bias
        unroll_for (_O, JO - 1) {
          unroll_for (_T, T)
            mmout[_O][_T] = op_load_bias<JO>(bias, _O);
        }
        if (has_Or) {
          unroll_for (_T, T)
            mmout[JO - 1][_T] = op_load_bias<JO>(bias, k, JO - 1);
        } else {
          unroll_for (_T, T)
            mmout[JO - 1][_T] = op_load_bias<JO>(bias, JO - 1);
        }
      } else {
        // clear output
        __m<V> tmp = _mm<V>::setzero_ps();
        unroll_for (_O, JO)
          unroll_for (_T, T)
            mmout[_O][_T] = tmp;
      }
      // load output
      if (test_bit(attr, AT_INP_SUM_MASK)) {
        unroll_for (_O, JO - 1) {
          unroll_for (_T, T)
            mmout[_O][_T] += op_load_output<JO>(ep, output, _O, _T);
        }
        if (has_Or) {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(ep, output, k, JO - 1, _T);
        } else {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(ep, output, JO - 1, _T);
        }
      }
    } else {
      // load output
      unroll_for (_O, JO - 1) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(ep, output, _O, _T);
      }
      if (has_Or) {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(ep, output, k, JO - 1, _T);
      } else {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(ep, output, JO - 1, _T);
      }
    }

    for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
      for (int _V = 0; _V < V / P; ++_V) {
        unroll_for(_P, P) {
          unroll_for(_T, T) {
            __m<V> mmbcst = op_load_input<P>(ep, input, _I2, _V, _P, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
          }
          unroll_auto(_O, JO) mmwei[_O][_P] =
              op_load_weights<JO, P>(ep, weights, _I2, _V + 1, _P, _O);
        }
      }
    }
    // Ir
    if (Ir > 0) {
#pragma nounroll
      for (int _V = 0; _V < ep.Ir; ++_V) {
        unroll_auto (_O, JO)
          mmwei[_O][0] = op_load_weights<JO, 1>(ep, weights, ep.I2 - 1, _V, 0, _O);
        unroll_for (_T, T) {
          __m<V> mmbcst = op_load_input<1>(ep, input, ep.I2 - 1, _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
        }
      }
    }

    // store output
    unroll_for (_O, JO - 1) {
      unroll_for (_T, T) {
        op_store_output<JO>(ep, output, mmout[_O][_T], _O, _T, attr);
      }
    }
    if (has_Or) {
      unroll_for (_T, T) {
        op_store_output<JO>(ep, output, mmout[JO - 1][_T], k, JO - 1, _T, attr);
      }
    } else {
      unroll_for (_T, T) {
        op_store_output<JO>(ep, output, mmout[JO - 1][_T], JO - 1, _T, attr);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 1>::type
  gemm(elx_param_t &ep, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr)
  {
    const int W_stride = F_traits<F>::is_compact_weights
                         ? ep.I2 * V * O * V : O * ep.IC * V;

    MD2(OutputType, aoutput_compact, output, ep.O1, O * T * V);
    MD2(OutputType, aoutput_blocked, output, ep.O1, O * ep.oh * ep.ow * V);
    MD5(OutputType, aoutput_nhwc, output, ep.oh * ep.ow, ep.g, ep.O4 * ep.O3, ep.O1, O * V);

    MD2(WeightsType, aweights, weights, ep.O1, W_stride);
    MD2(BiasType, abias, bias, ep.O1, O * V);

    for (int _O1 = 0; _O1 < ep.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, 0, _O1, 0)
          : F_traits<F>::is_compact_output ? &md2(aoutput_compact, _O1, 0)
                                           : &md2(aoutput_blocked, _O1, 0);
      if (F_traits<F>::is_nhwc_output && test_bit(attr, AT_Or_MASK)
          && _O1 == ep.O1 - 1) {
        op_gemm<JO0, JP0, true>(ep, aout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, _O1, 0);
      } else {
        op_gemm<JO0, JP0, false>(ep, aout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, _O1, 0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 2>::type
  gemm(elx_param_t &ep, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? ep.I2 * V : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V : ep.IC * V;

    MD3(OutputType, aoutput_compact, output, ep.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, ep.O1, O, ep.oh * ep.ow * V);
    MD6(OutputType, aoutput_nhwc, output, ep.oh * ep.ow, ep.g, ep.O4 * ep.O3, ep.O1, O, V);

    MD4(WeightsType, aweights, weights, ep.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, ep.O1, O, V);

    for (int _O1 = 0; _O1 < ep.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, false>(ep, aout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      if (F_traits<F>::is_nhwc_output && test_bit(attr, AT_Or_MASK)
          && _O1 == ep.O1 - 1) {
        op_gemm<JO1, JP1, true>(ep, aout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, _O1, JO0);
      } else {
        op_gemm<JO1, JP1, false>(ep, aout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, _O1, JO0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 3>::type
  gemm(elx_param_t &ep, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? ep.I2 * V : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V : ep.IC * V;

    MD3(OutputType, aoutput_compact, output, ep.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, ep.O1, O, ep.oh * ep.ow * V);
    MD6(OutputType, aoutput_nhwc, output, ep.oh * ep.ow, ep.g, ep.O4 * ep.O3, ep.O1, O, V);

    MD4(WeightsType, aweights, weights, ep.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, ep.O1, O, V);

    for (int _O1 = 0; _O1 < ep.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, false>(ep, aout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      op_gemm<JO1, JP1, false>(ep, aout, input, &md4(aweights, _O1, 0, JO0, 0),
          &md3(abias, _O1, JO0, 0), attr, _O1, JO0);
      aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0 + JO1, 0)
          : F_traits<F>::is_compact_output
              ? &md3(aoutput_compact, _O1, JO0 + JO1, 0)
              : &md3(aoutput_blocked, _O1, JO0 + JO1, 0);
      if (F_traits<F>::is_nhwc_output && test_bit(attr, AT_Or_MASK)
          && _O1 == ep.O1 - 1) {
        op_gemm<JO2, JP2, true>(ep, aout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, _O1, JO0 + JO1);
      } else {
        op_gemm<JO2, JP2, false>(ep, aout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, _O1, JO0 + JO1);
      }
    }
  }
};

} // namespace euler
