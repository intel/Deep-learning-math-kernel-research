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

// conv kernel with vector multi-group
// G: number of vmg
// C: ic/oc per group
// For V=16,
//    G = 1 (normal, see conv_kernel)
//    G = 2, C = 8
//    G = 4, C = 4
//    G = 8, C = 2
//    G = 16, C = 1 (depth-wise)
template <typename GarrayTypes, int V, int Vx, int I, typename KP>
struct vmg_conv_kernel {
  static inline void conv(
      elx_param_t &,
      typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *,
      int, int, int, int, int, int) {}
};

#define _MIN(a, b) ((a) <= (b) ? (a) : (b))

template <typename GarrayTypes, int V, int Vx, int ...Kp>
struct vmg_conv_kernel<GarrayTypes, V, Vx, ISA_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 6,
      "Kernel parameters must be GarrayTypes, V, Vx, I, <S, F, O, T, K, G>");

  using InputType = typename GarrayTypes::InputType;
  using WeightsType = typename GarrayTypes::WeightsType;
  using OutputType = typename GarrayTypes::OutputType;
  using BiasType = typename GarrayTypes::BiasType;

  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();
  constexpr static auto K = estl::get<4, int, kparams>();
  constexpr static auto G = estl::get<5, int, kparams>();

  static_assert(O == 1, "KMG conv: O > 1 not support");
  static_assert(G <= V, "KMG conv: G > V not support");

  // ic/oc per group
  constexpr static int C = V / G;

  // Loop splitting
  constexpr static int J = 1;
  constexpr static int JO0 = 1;
  constexpr static int JP0 =
      _MIN(C, ((K > 3 || F_traits<F>::is_compact_ir_weights)
                   ? 1
                   : J_traits<O, T, K_CONV, WeightsType>::P0));
  constexpr static int JO1 = 1;
  constexpr static int JP1 =
      _MIN(C, ((K > 3 || F_traits<F>::is_compact_ir_weights)
                   ? 1
                   : J_traits<O, T, K_CONV, WeightsType>::P1));
  constexpr static int JO2 = 1;
  constexpr static int JP2 =
      _MIN(C, ((K > 3 || F_traits<F>::is_compact_ir_weights)
                   ? 1
                   : J_traits<O, T, K_CONV, WeightsType>::P2));

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
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), ep.O4 * ep.O3 * ep.O1, ep.O, V);
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

  template <int JO, int P>
  static inline __m<V> op_load_weights(elx_param_t &ep,
      WeightsType *weights, const int _I2, const int _V, const int _P, const int _O)
  {
    __m<V> res;
    if (F_traits<F>::is_compact_ir_weights) {
      MD4(WeightsType, aweights, weights, ep.I2, ep.Ir, O, V);
      if (std::is_same<WeightsType, float>::value) {
        res = _mm<V>::load_ps(&md4(aweights, _I2, _V, _O, 0));
      } else {
        if (O == 2) { // bf16 type weights
          res = (_O == 0)
              ? _mm<V>::load_ps(&md4(aweights, _I2, _V, 0, 0))
              : _mm<V>::loadu_ps(&md4(aweights, _I2, _V, 0, 0) - 1);
        } else {      // fp16 type weights
          auto fp16v = _mm<V / 2>::load_si256(
              (__m256i *)&md4(aweights, _I2, _V, _O, 0));
          res = _mm<V>::cvtph_ps(fp16v);
        }
      }
    } else if (F_traits<F>::is_compact_weights) {
      MD5(WeightsType, aweights5, weights, ep.I2, C / P, P, O, V);
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
    } else { // blocked, TODO: consider remove
      MD6(WeightsType, aweights6, weights, JO, ep.ic34, ep.I2, C / P, P, V);
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

  // overload load_input: support nchw, nhwc or blocked format
  static inline __m<V> op_load_input(elx_param_t &ep, InputType *input,
      const int _ih, const int _iw, const int _I2, const int _V, const int _T)
  {
    __m<V> vin;
    if (F_traits<F>::is_nhwc_input) {
      MD3(InputType, ainput0, input, ep.ih, ep.iw, ep.g * ep.ic);
      MD5(InputType, ainput1, &md3(ainput0, _ih, _iw, 0), ep.wt, T, S, ep.g, ep.ic);
      MD4(InputType, ainput2, &md5(ainput1, 0, _T, 0, 0, 0), ep.I4, ep.I3, ep.I2, V);
      vin = _mm<V>::load_ps(&md4(ainput2, 0, 0, _I2, 0));
    } else { // blocked
      MD4(InputType, ainput0, input, ep.I2, ep.ih, ep.iw, V);
      MD3(InputType, ainput1, &md4(ainput0, _I2, _ih, _iw, 0), T, S, V);
      vin = _mm<V>::load_ps(&md3(ainput1, _T, 0, 0));
    }

    if (V == 16) {
      if (G == 1) {
        // _V = 0..15
        __m<V> tmp =
            _V < 4 ? _mm512_shuffle_f32x4(vin, vin, 0x00)
                   : _V < 8 ? _mm512_shuffle_f32x4(vin, vin, 0x55)
                            : _V < 12 ? _mm512_shuffle_f32x4(vin, vin, 0xAA)
                                      : _mm512_shuffle_f32x4(vin, vin, 0xFF);
        const int __V = _V & 0x3;
        return __V == 3 ? _mm512_permute_ps(tmp, 0xFF)
                        : __V == 2 ? _mm512_permute_ps(tmp, 0xAA)
                                   : __V == 1 ? _mm512_permute_ps(tmp, 0x55)
                                              : _mm512_permute_ps(tmp, 0x0);
      } else if (G == 2) {
        // _V = 0..7
        __m<V> tmp = _V < 4 ? _mm512_shuffle_f32x4(vin, vin, 0xA0)
                            : _mm512_shuffle_f32x4(vin, vin, 0xF5);
        const int __V = _V & 0x3;
        return __V == 3 ? _mm512_permute_ps(tmp, 0xFF)
                        : __V == 2 ? _mm512_permute_ps(tmp, 0xAA)
                                   : __V == 1 ? _mm512_permute_ps(tmp, 0x55)
                                              : _mm512_permute_ps(tmp, 0x0);
      } else if (G == 4) {
        // _V = 0..3
        return _V == 3 ? _mm512_permute_ps(vin, 0xFF)
                       : _V == 2 ? _mm512_permute_ps(vin, 0xAA)
                                 : _V == 1 ? _mm512_permute_ps(vin, 0x55)
                                           : _mm512_permute_ps(vin, 0x0);
      } else if (G == 8) {
        // _V = 0,1
        //int imm8 = _V == 1 ? 0x5 : 0x8;
        //return _mm512_permute_ps(vin, imm8);
        return _V == 0 ? _mm512_moveldup_ps(vin) : _mm512_movehdup_ps(vin);
      } else if (G == 16) {
        // _V = 0
        return vin;
      }
    } else {
      el_error("V != 16 not support");
    }
    return vin;
  }

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 1, void>::type
  op_conv(elx_param_t &ep, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias, int _wt,
      int khs, int khe, int kws, int kwe, int attr)
  {
    const int AKH = ep.kh / 2;
    constexpr int AKW = K / 2;

    //int I2 = ep.I2, Ir = 0;
    //if (test_bit(attr, AT_Ir_MASK)) {
    //  I2 = ep.I2 - 1;
    //  Ir = ep.Ir;
    //}
    int I2 = ep.I2;

    //int Vr = F_traits<F>::is_compact_ir_weights ? ep.Ir : V;
    MD3(WeightsType, aweights, weights, ep.kh, ep.kw, ep.O1 * I2 * C * O * V); // compact

    __m<V> mmout[JO][T], mmwei[JO][P];
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

    auto gemm_OVT = [&](InputType *input_, WeightsType *weights_,
                        int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxT = [&](InputType *input_, WeightsType *weights_,
                         int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxxT = [&](InputType *input_, WeightsType *weights_,
                          int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW - 1 + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxxxT = [&](InputType *input_, WeightsType *weights_,
                           int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW - 2 + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTx = [&](InputType *input_, WeightsType *weights_,
                         int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - AKW/S) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTxx = [&](InputType *input_, WeightsType *weights_,
                          int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - (AKW - 1)/S) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTxxx = [&](InputType *input_, WeightsType *weights_,
                           int _kh, int _kw, int _I2) {
      unroll_for(_V, C) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(ep, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - (AKW - 2)/S) {
          __m<V> mmbcst = op_load_input(ep, input_, _kh - AKH, _kw - AKW, _I2, _V, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };

    for (int _kh = khs; _kh < khe; ++_kh) {
      for (int _I2 = 0; _I2 < I2; ++_I2) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          gemm_OVT(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
        }
        // left
        if (_wt == 0) {
          int _kw = 0; // K = 3, 5, 7
          gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          if (K > 3) {
            _kw = 1; // K = 5, 7
            gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = 2; // K = 7
            gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          }
        }
        // right
        if (_wt == ep.wt - 1) {
          int _kw = K - 1; // K = 3, 5, 7
          gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          if (K > 3) {
            _kw = K - 2; // K = 5, 7
            gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = K - 3; // K = 7
            gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), _kh, _kw, _I2);
          }
        }
      } // I2 loop

#if 0
      // Ir
      if (Ir > 0) {
        int _I2 = ep.I2 - 1;
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          gemm_OVT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
        }
        // left
        if (_wt == 0) {
          int _kw = 0; // K = 3, 5, 7
          gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          if (K > 3) {
            _kw = 1; // K = 5, 7
            gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = 2; // K = 7
            gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
        }
        // right
        if (_wt == ep.wt - 1) {
          int _kw = K - 1; // K = 3, 5, 7
          gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          if (K > 3) {
            _kw = K - 2; // K = 5, 7
            gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
          if (K > 5) {
            _kw = K - 3; // K = 7
            gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
        }
      }
#endif
    }

    // store output
    unroll_for (_O, JO - 1) {
      unroll_for (_T, T)
        op_store_output<JO>(ep, output, mmout[_O][_T], _O, _T, attr);
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

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<(P == 2 || P == 4), void>::type
  op_conv(elx_param_t &ep, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias, int _wt,
      int khs, int khe, int kws, int kwe, int attr)
  {
    // 3x3 conv
    constexpr int AKH = 3 / 2;
    constexpr int AKW = 3 / 2;

    //int I2 = ep.I2, Ir = 0;
    //if (test_bit(attr, AT_Ir_MASK)) {
    //  I2 = ep.I2 - 1;
    //  Ir = ep.Ir;
    //}
    int I2 = ep.I2;

    MD3(WeightsType, aweights, weights, ep.kh, ep.kw, ep.O1 * ep.I2 * C * O * V); // compact

    __m<V> mmout[JO][T], mmwei[JO][P];
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

    for (int _kh = khs; _kh < khe; ++_kh) {
      // mid
      for (int _kw = kws; _kw < kwe; ++_kw) {
        // preload weights
        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              ep, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          unroll_for(_V, C / P) {
            unroll_auto(_P, P) {
              unroll_for(_T, T) {
                __m<V> mmbcst = op_load_input(ep, input, _kh - AKH, _kw - AKW, _I2, _V * P + _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_for(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  ep, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);
            }
          } // _V
        }
#if 0
        // Ir
        int _I2 = ep.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              ep, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_for(_T, T) {
            __m<V> mmbcst = op_load_input<1>(
                ep, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
#endif
      } // _kw loop, mid

      // left
      if (_wt == 0) {
        constexpr int _kw = 0;

        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              ep, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          unroll_for(_V, C / P) {
            unroll_auto(_P, P) {
              unroll_from_to(_T, 1, T) {
                __m<V> mmbcst = op_load_input(ep, input, _kh - AKH, _kw - AKW, _I2, _V * P + _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_for(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  ep, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);
            }
          } // _V
        }
#if 0
        // Ir
        int _I2 = ep.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              ep, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_from_to(_T, 1, T) {
            __m<V> mmbcst = op_load_input<1>(
                ep, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
#endif
      } // left

      // right
      if (_wt == ep.wt - 1) {
        constexpr int _kw = 2;

        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              ep, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          unroll_for(_V, C / P) {
            unroll_auto(_P, P) {
              unroll_for(_T, T - 1 + S - 1) {
                __m<V> mmbcst = op_load_input(ep, input, _kh - AKH, _kw - AKW, _I2, _V * P + _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_for(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  ep, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);

            }
          } // _V
        }
#if 0
        int _I2 = ep.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              ep, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_for(_T, T - 1 + S - 1) {
            __m<V> mmbcst = op_load_input<1>(
                ep, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
#endif
      } // right
    } // _kh loop

    // store output
    unroll_for (_O, JO - 1) {
      unroll_for (_T, T)
        op_store_output<JO>(ep, output, mmout[_O][_T], _O, _T, attr);
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

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 1) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_param_t &ep, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int _wt, int khs, int khe,
          int kws, int kwe, int attr)
  {
    int Vr = F_traits<F>::is_compact_ir_weights ? ep.Ir : C;
    MD3(WeightsType, aweights, weights, ep.kh * ep.kw, ep.O1,
        ep.I2 * Vr * O * V); // compact
    MD2(OutputType, aoutput_blocked, output, ep.O1, O * ep.oh * ep.ow * V);
    MD5(OutputType, aoutput_nhwc, output, ep.oh * ep.ow, ep.g, ep.O4 * ep.O3, ep.O1, O * V);
    MD2(BiasType, abias, bias, ep.O1, O * V);

    for (int _O1 = 0; _O1 < ep.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, 0, _O1, 0)
                                              : &md2(aoutput_blocked, _O1, 0);
      if (F_traits<F>::is_nhwc_output && test_bit(attr, AT_Or_MASK)
          && _O1 == ep.O1 - 1) {
        op_conv<JO0, JP0, true>(ep, aout, input, &md3(aweights, 0, _O1, 0),
            &md2(abias, _O1, 0), _wt, khs, khe, kws, kwe, attr);
      } else {
        op_conv<JO0, JP0, false>(ep, aout, input, &md3(aweights, 0, _O1, 0),
            &md2(abias, _O1, 0), _wt, khs, khe, kws, kwe, attr);
      }
    }
  }

};

} // namespace euler
