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
struct conv_kernel_otj {
  static inline void conv(
      elx_conv_params_t &,
      typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *,
      int, int, int, int, int, int, int) {}
};

template <typename GarrayTypes, int V, int Vx, int ...Kp>
struct conv_kernel_otj<GarrayTypes, V, Vx, ISA_SKX_AVX512,
    estl::integer_sequence<Kp...>> {
  using kparams = estl::integer_sequence<Kp...>;
  static_assert(sizeof...(Kp) == 5,
      "Kernel parameters must be GarrayTypes, V, Vx, I, <S, F, O, T, K>");

  using InputType = typename GarrayTypes::InputType;
  using WeightsType = typename GarrayTypes::WeightsType;
  using OutputType = typename GarrayTypes::OutputType;
  using BiasType = typename GarrayTypes::BiasType;
  using ScaleType = typename GarrayTypes::ScaleType;

  constexpr static auto S = estl::get<0, int, kparams>();
  constexpr static auto F = estl::get<1, int, kparams>();
  constexpr static auto O = estl::get<2, int, kparams>();
  constexpr static auto T = estl::get<3, int, kparams>();
  constexpr static auto K = estl::get<4, int, kparams>();

  // Jamming components
  constexpr static int J = J_traits<O, T, K_CONV, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, K_CONV, WeightsType>::O0;
  constexpr static int JP0 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                                 ? 1
                                 : J_traits<O, T, K_CONV, WeightsType>::P0;
  constexpr static int JO1 = J_traits<O, T, K_CONV, WeightsType>::O1;
  constexpr static int JP1 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                                 ? 1
                                 : J_traits<O, T, K_CONV, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, K_CONV, WeightsType>::O2;
  constexpr static int JP2 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                                 ? 1
                                 : J_traits<O, T, K_CONV, WeightsType>::P2;

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
  static inline __m<V> op_load_output(elx_conv_params_t &xc, OutputType *output,
                                      const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, xc.g, xc.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

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
  static inline __m<V> op_load_output(elx_conv_params_t &xc, OutputType *output,
                                      __mmask16 k, const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, xc.g, xc.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);
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
  static inline __m<V> op_load_weights(elx_conv_params_t &xc,
      WeightsType *weights, const int _I2, const int _V, const int _P, const int _O)
  {
    __m<V> res;
    if (F_traits<F>::is_compact_ir_weights) {
      MD4(WeightsType, aweights, weights, xc.I2, xc.Ir, O, V);
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
      MD5(WeightsType, aweights5, weights, xc.I2, V / P, P, O, V);
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

  template <int JO>
  static inline void op_store_output(elx_conv_params_t &xc,
      OutputType *output, __m<V> res, const int _O, const int _T, const int attr)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, xc.g, xc.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);

    if (get_attr(attr, relu_idx)) {
      auto lower = *(__m<V> *)(xc.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(xc.relu_bound_upper_vec);
      res = _mm<V>::max_ps(res, lower);
      res = _mm<V>::min_ps(res, upper);
    }
    if (get_attr(attr, s_output_idx)) {
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
  static inline void op_store_output(elx_conv_params_t &xc, OutputType *output,
      __m<V> res, __mmask16 k, const int _O, const int _T, const int attr)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD3(OutputType, aoutput_nhwc0, output, T, xc.g, xc.oc);
    MD3(OutputType, aoutput_nhwc1, &md3(aoutput_nhwc0, _T, 0, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);
    assert(F_traits<F>::is_nhwc_output);

    if (get_attr(attr, relu_idx)) {
      auto lower = *(__m<V> *)(xc.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(xc.relu_bound_upper_vec);
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
  template <int P>
  static inline __m<V> op_load_input(elx_conv_params_t &xc, InputType *input,
      const int _ih, const int _iw, const int _I2, const int _V, const int _P, const int _T)
  {
    if (F_traits<F>::is_nchw_input) {
      MD2(InputType, ainput0, input, xc.I2, xc.ih * xc.iw * V);
      MD4(InputType, ainput1, &md2(ainput0, _I2, 0), V / P, P, xc.ih, xc.iw);
      MD3(InputType, ainput2, &md4(ainput1, _V, _P, _ih, _iw), xc.wt, T, S);
      return _mm<V>::set1_ps(md3(ainput2, 0, _T, 0));
    } else if (F_traits<F>::is_nhwc_input) {
      MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.g * xc.ic);
      MD5(InputType, ainput1, &md3(ainput0, _ih, _iw, 0), xc.wt, T, S, xc.g, xc.ic);
      MD5(InputType, ainput2, &md5(ainput1, 0, _T, 0, 0, 0), xc.ic4, xc.ic3, xc.I2, V/P, P);
      return _mm<V>::set1_ps(md5(ainput2, 0, 0, _I2, _V, _P));
    } else { // blocked
      MD4(InputType, ainput0, input, xc.I2, xc.ih, xc.iw, V);
      MD4(InputType, ainput1, &md4(ainput0, _I2, _ih, _iw, 0), T, S, V / P, P);
      return _mm<V>::set1_ps(md4(ainput1, _T, 0, _V, _P));
    }
  }

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 1, void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias,
      int khs, int khe, int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    int Vr = F_traits<F>::is_compact_ir_weights ? xc.Ir : V;
    MD3(WeightsType, aweights, weights, xc.kh, xc.kw, xc.O1 * xc.I2 * Vr * O * V); // compact

    __m<V> mmout[JO][T], mmwei[JO][P];
    __mmask16 k = _cvtu32_mask16(xc.ormask);

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
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
      if (get_attr(attr, ip_sum_idx)) {
        unroll_for (_O, JO - 1) {
          unroll_for (_T, T)
            mmout[_O][_T] += op_load_output<JO>(xc, output, _O, _T);
        }
        if (has_Or) {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(xc, output, k, JO - 1, _T);
        } else {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(xc, output, JO - 1, _T);
        }
      }
    } else {
      // load output
      unroll_for (_O, JO - 1) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(xc, output, _O, _T);
      }
      if (has_Or) {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(xc, output, k, JO - 1, _T);
      } else {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(xc, output, JO - 1, _T);
      }
    }

    auto gemm_OVT = [&](InputType *input_, WeightsType *weights_,
                        int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxT = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxxT = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW - 1 + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVxxxT = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_from_to(_T, (AKW - 2 + S - 1)/S, T) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTx = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - AKW/S) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTxx = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - (AKW - 1)/S) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };
    auto gemm_OVTxxx = [&](InputType *input_, WeightsType *weights_,
                         int V_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V = 0; _V < V_; ++_V) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V, 0, _O);
        unroll_for(_T, T - (AKW - 2)/S) {
          __m<V> mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
          unroll_for(_O, JO) mmout[_O][_T] += mmwei[_O][0] * mmbcst;
        }
      }
    };

    for (int _kh = khs; _kh < khe; ++_kh) {
      for (int _I2 = 0; _I2 < I2; ++_I2) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          gemm_OVT(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
        }
        // left
        if (pad_l) {
          int _kw = 0; // K = 3, 5, 7
          gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          if (pad_l > 1 && K > 3) {
            _kw = 1; // K = 5, 7
            gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          }
          if (pad_l > 2 && K > 5) {
            _kw = 2; // K = 7
            gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          }
        }
        // right
        if (pad_r) {
          int _kw = K - 1; // K = 3, 5, 7
          gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          if (pad_r > 1 && K > 3) {
            _kw = K - 2; // K = 5, 7
            gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          }
          if (pad_r > 2 && K > 5) {
            _kw = K - 3; // K = 7
            gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), V, _kh, _kw, _I2);
          }
        }
      } // I2 loop

      // Ir
      if (Ir > 0) {
        int _I2 = xc.I2 - 1;
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          gemm_OVT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
        }
        // left
        if (pad_l) {
          int _kw = 0; // K = 3, 5, 7
          gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          if (pad_l > 1 && K > 3) {
            _kw = 1; // K = 5, 7
            gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
          if (pad_l > 2 && K > 5) {
            _kw = 2; // K = 7
            gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
        }
        // right
        if (pad_r) {
          int _kw = K - 1; // K = 3, 5, 7
          gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          if (pad_r > 1 && K > 3) {
            _kw = K - 2; // K = 5, 7
            gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
          if (pad_r > 2 && K > 5) {
            _kw = K - 3; // K = 7
            gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), Ir, _kh, _kw, _I2);
          }
        }
      }
    }

    // store output
    unroll_for (_O, JO - 1) {
      unroll_for (_T, T)
        op_store_output<JO>(xc, output, mmout[_O][_T], _O, _T, attr);
    }
    if (has_Or) {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[JO - 1][_T], k, JO - 1, _T, attr);
      }
    } else {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[JO - 1][_T], JO - 1, _T, attr);
      }
    }
  }

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<(P == 2 || P == 4), void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output,
      InputType *input, WeightsType *weights, BiasType *bias,
      int khs, int khe, int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    // 3x3 conv
    constexpr int AKH = 3 / 2;
    constexpr int AKW = 3 / 2;

    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    MD3(WeightsType, aweights, weights, xc.kh, xc.kw, xc.O1 * xc.I2 * V * O * V); // compact

    __m<V> mmout[JO][T], mmwei[JO][P];
    __mmask16 k = _cvtu32_mask16(xc.ormask);

    if (get_attr(attr, r_output_idx)) {
      if (get_attr(attr, bias_idx)) {
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
      if (get_attr(attr, ip_sum_idx)) {
        unroll_for (_O, JO - 1) {
          unroll_for (_T, T)
            mmout[_O][_T] += op_load_output<JO>(xc, output, _O, _T);
        }
        if (has_Or) {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(xc, output, k, JO - 1, _T);
        } else {
          unroll_for (_T, T)
            mmout[JO - 1][_T] += op_load_output<JO>(xc, output, JO - 1, _T);
        }
      }
    } else {
      // load output
      unroll_for (_O, JO - 1) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(xc, output, _O, _T);
      }
      if (has_Or) {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(xc, output, k, JO - 1, _T);
      } else {
        unroll_for (_T, T)
          mmout[JO - 1][_T] = op_load_output<JO>(xc, output, JO - 1, _T);
      }
    }

    for (int _kh = khs; _kh < khe; ++_kh) {
      // mid
      for (int _kw = kws; _kw < kwe; ++_kw) {
        // preload weights
        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V = 0; _V < V / P; ++_V) {
            unroll_for(_P, P) {
              unroll_for(_T, T) {
                __m<V> mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V, _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);
            }
          } // _V
        }
        // Ir
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_for(_T, T) {
            __m<V> mmbcst = op_load_input<1>(
                xc, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      } // _kw loop, mid

      // left
      if (pad_l) {
        constexpr int _kw = 0;

        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V = 0; _V < V / P; ++_V) {
            unroll_for(_P, P) {
              unroll_from_to(_T, 1, T) {
                __m<V> mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V, _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);
            }
          } // _V
        }
        // Ir
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_from_to(_T, 1, T) {
            __m<V> mmbcst = op_load_input<1>(
                xc, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      } // left

      // right
      if (pad_r) {
        constexpr int _kw = 2;

        unroll_for(_P, P) {
        unroll_auto(_O, JO) {
          mmwei[_O][_P] = op_load_weights<JO, P>(
              xc, &md3(aweights, _kh, _kw, 0), 0, 0, _P, _O);
        }}
        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V = 0; _V < V / P; ++_V) {
            unroll_for(_P, P) {
              unroll_for(_T, T - 1 + S - 1) {
                __m<V> mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V, _P, _T);
                unroll_for(_O, JO) mmout[_O][_T] =
                    _mm<V>::fmadd_ps(mmwei[_O][_P], mmbcst, mmout[_O][_T]);
              }
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V + 1, _P, _O);

            }
          } // _V
        }
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V = 0; _V < Ir; ++_V) {
          unroll_auto(_O, JO)
            mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V, 0, _O);
          unroll_for(_T, T - 1 + S - 1) {
            __m<V> mmbcst = op_load_input<1>(
                xc, input, _kh - AKH, _kw - AKW, _I2, _V, 0, _T);
            unroll_for(_O, JO) mmout[_O][_T] =
                _mm<V>::fmadd_ps(mmwei[_O][0], mmbcst, mmout[_O][_T]);
          }
        }
      } // right
    } // _kh loop

    // store output
    unroll_for (_O, JO - 1) {
      unroll_for (_T, T)
        op_store_output<JO>(xc, output, mmout[_O][_T], _O, _T, attr);
    }
    if (has_Or) {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[JO - 1][_T], k, JO - 1, _T, attr);
      }
    } else {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[JO - 1][_T], JO - 1, _T, attr);
      }
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 1) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int khs, int khe,
          int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    int Vr = F_traits<F>::is_compact_ir_weights ? xc.Ir : V;
    MD3(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1,
        xc.I2 * Vr * O * V); // compact
    MD2(OutputType, aoutput_blocked, output, xc.O1, O * xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.g, xc.oc4 * xc.oc3, xc.O1, O * V);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, 0, _O1, 0)
                                              : &md2(aoutput_blocked, _O1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_conv<JO0, JP0, true>(xc, aout, input, &md3(aweights, 0, _O1, 0),
            &md2(abias, _O1, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
      } else {
        op_conv<JO0, JP0, false>(xc, aout, input, &md3(aweights, 0, _O1, 0),
            &md2(abias, _O1, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
      }
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 2) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int khs, int khe,
          int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    int Vr = F_traits<F>::is_compact_ir_weights ? xc.Ir : V;
    MD5(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1,
        xc.I2 * Vr, O, V); // compact
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD6(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.g, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, 0, 0)
          : &md3(aoutput_blocked, _O1, 0, 0);
      op_conv<JO0, JP0, false>(xc, aout, input, &md5(aweights, 0, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
      aout = F_traits<F>::is_nhwc_output ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0, 0)
                                         : &md3(aoutput_blocked, _O1, JO0, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_conv<JO1, JP1, true>(xc, aout, input,
            &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
            khs, khe, kws, kwe, pad_l, pad_r, attr);
      } else {
        op_conv<JO1, JP1, false>(xc, aout, input,
            &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
            khs, khe, kws, kwe, pad_l, pad_r, attr);
      }
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 3) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, InputType *input,
          WeightsType *weights, BiasType *bias, int khs, int khe,
          int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    int Vr = F_traits<F>::is_compact_ir_weights ? xc.Ir : V;
    MD5(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1,
        xc.I2 * Vr, O, V); // compact
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD6(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.g, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, 0, 0)
          : &md3(aoutput_blocked, _O1, 0, 0);
      op_conv<JO0, JP0, false>(xc, aout, input, &md5(aweights, 0, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
      aout = F_traits<F>::is_nhwc_output ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0, 0)
                                         : &md3(aoutput_blocked, _O1, JO0, 0);
      op_conv<JO1, JP1, false>(xc, aout, input, &md5(aweights, 0, _O1, 0, JO0, 0),
          &md3(abias, _O1, JO0, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
      aout = F_traits<F>::is_nhwc_output
          ? &md6(aoutput_nhwc, 0, 0, 0, _O1, JO0 + JO1, 0)
          : &md3(aoutput_blocked, _O1, JO0 + JO1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_conv<JO2, JP2, true>(xc, aout, input,
            &md5(aweights, 0, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), khs, khe, kws, kwe,
            pad_l, pad_r, attr);
      } else {
        op_conv<JO2, JP2, false>(xc, aout, input,
            &md5(aweights, 0, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), khs, khe, kws, kwe,
            pad_l, pad_r, attr);
      }
    }
  }

};

} // namespace euler
