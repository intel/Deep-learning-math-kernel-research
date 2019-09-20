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

template <typename GarrayTypes, typename RoutputType, int V, int Vx, int I, typename KP>
struct u8s8_conv_kernel_otj {
  static inline void conv(
      elx_conv_params_t &,
      typename GarrayTypes::OutputType *,
      RoutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      int, int, int, int, int, int, int) {}
};

template <typename GarrayTypes, typename RoutputType, int V, int Vx, int ...Kp>
struct u8s8_conv_kernel_otj<GarrayTypes, RoutputType, V, Vx, ISA_SKX_AVX512,
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
  constexpr static int J   = J_traits<O, T, K_CONV, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, K_CONV, WeightsType>::O0;
  constexpr static int JO1 = J_traits<O, T, K_CONV, WeightsType>::O1;
  constexpr static int JO2 = J_traits<O, T, K_CONV, WeightsType>::O2;

  constexpr static int V1 = V / Vx;

  constexpr static int JP0 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                           ? 1
                           : J_traits<O, T, K_CONV, WeightsType>::P0;
  constexpr static int JP1 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                           ? 1
                           : J_traits<O, T, K_CONV, WeightsType>::P1;
  constexpr static int JP2 = (K > 3 || F_traits<F>::is_compact_ir_weights)
                           ? 1
                           : J_traits<O, T, K_CONV, WeightsType>::P2;

  // INT8 gemm kernel
#if !defined(WITH_VNNI)
  static inline __i<V> op_int8_fma_opt(
      __i<V>& out, __i<V>& a1, __i<V>& a2, __i<V>&b1, __i<V>& b2) {
    auto one = _mm<V>::set1_epi16(1);
    auto t1 = _mm<V>::maddubs_epi16(a1, b1);
    auto t2 = _mm<V>::maddubs_epi16(a2, b2);
    t1 = _mm<V>::adds_epi16(t1, t2);
    t1 = _mm<V>::madd_epi16(t1, one);
    out = _mm<V>::add_epi32(t1, out);
    return out;
  }
#endif

  static inline __i<V> op_int8_fma(__i<V>& out, __i<V>& a, __i<V>& b) {
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

  template <int JO>
  static inline __m<V> op_load_bias(BiasType *bias, const int _O)
  {
    MD2(BiasType, abias2, bias, JO, V);
    if (std::is_same<BiasType, float>::value) {
      return _mm<V>::load_ps(&md2(abias2, _O, 0));
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(abias2, _O, 0));
      return _mm<V>::cvtph_ps(fp16v);
    }
  }

  template <int JO>
  static inline __i<V> op_load_output(elx_conv_params_t &xc, OutputType *output,
                                      const int _O, const int _T)
  {
    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);
    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0),
        xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_blocked_output ? &md2(aoutput_blocked1, _T, 0)
                                               : &md3(aoutput_nhwc1, 0, _O, 0);
    __i<V> res;
    if (std::is_same<OutputType, float>::value) {
      res = _mm<V>::load_epi32((__i<V> *)aout);
    } else {
      el_error("load output in conv kernel: not supported output type");
    }
    return res;
  }

  template <int P>
  static inline __i<V> op_load_input(elx_conv_params_t &xc, InputType *input,
      const int _ih, const int _iw, const int _I2, const int _V1,
      const int _P, const int _T)
  {
    if (F_traits<F>::is_nhwc_input && F_traits<F>::is_compact_ir_weights) {
      MD3(InputType, ainput0, input, xc.ih, xc.iw, 3);
      MD3(InputType, ainput1, &md3(ainput0, _ih, _iw, 0), T, S, 3);
      return _mm<V>::set1_epi32(*(int32_t*)&md3(ainput1, _T, 0, 0));
    } else if (F_traits<F>::is_nhwc_input) {
      MD3(InputType, ainput0, input, xc.ih, xc.iw, xc.ic);
      MD4(InputType, ainput1, &md3(ainput0, _ih, _iw, 0), xc.wt, T, S, xc.ic);
      MD6(InputType, ainput2, &md4(ainput1, 0, _T, 0, 0), xc.ic4, xc.ic3,
          xc.I2, V1 / P, P, Vx);
      return _mm<V>::set1_epi32(*(int32_t*)&md6(ainput2, 0, 0, _I2, _V1, _P, 0));
    } else { // blocked
      MD4(InputType, ainput0, input, xc.I2, xc.ih, xc.iw, V);
      MD5(InputType, ainput1, &md4(ainput0, _I2, _ih, _iw, 0), T, S, V1 / P, P, Vx);
      return _mm<V>::set1_epi32(*(int32_t*)&md5(ainput1, _T, 0, _V1, _P, 0));
    }
  }

  template <int JO, int P>
  static inline __i<V> op_load_weights(elx_conv_params_t &xc,
      WeightsType *weights, const int _I2, const int _V1, const int _P, const int _O)
  {
    __i<V> res;
    if (F_traits<F>::is_compact_ir_weights) {
      MD4(int8_t, aweights5, weights, xc.I2, xc.Ir, O, V * Vx);
      res = _mm<V>::load_epi32(&md4(aweights5, _I2, _V1, _O, 0));
    } else if (F_traits<F>::is_compact_weights) {
      MD5(int8_t, aweights5, weights, xc.I2, V1 / P, P, O, V * Vx);
      res = _mm<V>::load_epi32(&md5(aweights5, _I2, _V1, _P, _O, 0));
    } else {
      el_error("load weights in conv kernel: not supported weights format");
    }
    return res;
  }

  template <int JO>
  static inline void op_store_output(elx_conv_params_t &xc,
      OutputType *output, __i<V> res, const int _O, const int _T)
  {
    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);
    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0),
        xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_blocked_output ? &md2(aoutput_blocked1, _T, 0)
                                               : &md3(aoutput_nhwc1, 0, _O, 0);

    if (std::is_same<OutputType, float>::value) {
      _mm<V>::store_epi32(aout, res);
    } else {
      el_error("store output in conv kernel: not supported output type");
    }
  }

  template <int JO>
  static inline void op_restore_output(elx_conv_params_t &xc, OutputType *output,
      RoutputType *routput, BiasType *bias, __i<V> res, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      const int _O1, const int _O0, const int _O, const int _T, const int attr)
  {
    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);
    MD2(RoutputType, aroutput_blocked0, routput, JO, xc.oh * xc.ow * V);
    MD2(RoutputType, aroutput_blocked1, &md2(aroutput_blocked0, _O, 0), T, V);

    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0),
        xc.oc4 * xc.oc3 * xc.O1, xc.O, V);
    MD2(RoutputType, aroutput_nhwc0, routput, T, xc.oc);
    MD3(RoutputType, aroutput_nhwc1, &md2(aroutput_nhwc0, _T, 0),
        xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    MD3(float, aweights_scale3, weights_scale, xc.O1, O, V);
    MD2(float, aweights_scale, &md3(aweights_scale3, _O1, _O0, 0), JO, V);
    MD3(float, aweights_factor3, weights_factor, xc.O1, O, T * V);
    MD3(float, aweights_factor, &md3(aweights_factor3, _O1, _O0, 0), JO, T, V);

    auto aout = F_traits<F>::is_blocked_output ? &md2(aoutput_blocked1, _T, 0)
                                               : &md3(aoutput_nhwc1, 0, _O, 0);
    auto rout = F_traits<F>::is_blocked_output ? &md2(aroutput_blocked1, _T, 0)
                                               : &md3(aroutput_nhwc1, 0, _O, 0);

    __m<V> fout = _mm<V>::cvtepi32_ps(res);

    // restore and requantization
    auto scale = *(__m<V> *)&md2(aweights_scale, _O, 0);
    auto factor = *(__m<V> *)&md3(aweights_factor, _O, _T, 0);
    fout = fout * scale + factor;

    // fuse relu
    if (get_attr(attr, relu_idx)) {
      auto lower = *(__m<V> *)(xc.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(xc.relu_bound_upper_vec);
      fout = _mm<V>::max_ps(fout, lower);
      fout = _mm<V>::min_ps(fout, upper);
    }
    // store output
    if (std::is_same<RoutputType, uint8_t>::value
        || std::is_same<RoutputType, int8_t>::value) {
      __i<V> s32 = _mm<V>::cvt_roundps_epi32(
          fout, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
      __m128i x8;
      if (std::is_same<RoutputType, int8_t>::value)
        x8 = _mm<V>::cvtsepi32_epi8(s32);
      else
        x8 = _mm<V>::cvtusepi32_epi8(s32);
      _mm_store_si128((__m128i *)rout, x8);
    } else {
      if (std::is_same<OutputType, float>::value)
        _mm<V>::store_ps(aout, fout);
      else {
        auto fp16v = _mm<V>::cvtps_ph(
            fout, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)aout, fp16v);
      }
    }
  }

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 1, void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
      uint8_t *input, int8_t *weights, BiasType *bias, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      int khs, int khe, int kws, int kwe, int pad_l, int pad_r, int attr, int _O1, int _O0)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

#if defined(WITH_VNNI)
    __i<V> mmout[JO][T], mmwei[JO][1];
#else
    __i<V> mmout[JO][T], mmwei[JO][2];
#endif

    if (get_attr(attr, r_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else {
      // load accumulated s32 output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(xc, output, _O, _T);
      }
    }

    auto gemm_OVT = [&](InputType *input_, WeightsType *weights_,
                        int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_for(_T, T) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVxT = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_from_to(_T, (AKW + S - 1)/S, T) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVxxT = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_from_to(_T, (AKW - 1 + S - 1)/S, T) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVxxxT = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_from_to(_T, (AKW - 2 + S - 1)/S, T) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                         _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVTx = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_for(_T, T - AKW/S) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVTxx = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_for(_T, T - (AKW - 1)/S) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };
    auto gemm_OVTxxx = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
      for (int _V1 = 0; _V1 < V1_; ++_V1) {
        unroll_auto(_O, JO)
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
        unroll_for(_T, T - (AKW - 2)/S) {
          auto mmbcst = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
        }
      }
    };

#if !defined(WITH_VNNI)
    auto gemm_OVT_opt = [&](InputType *input_, WeightsType *weights_,
                        int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_for(_T, T) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVxT_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_from_to(_T, (AKW + S - 1)/S, T) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVxxT_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_from_to(_T, (AKW - 1 + S - 1)/S, T) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVxxxT_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_from_to(_T, (AKW - 2 + S - 1)/S, T) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVTx_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_for(_T, T - AKW/S) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVTxx_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_for(_T, T - (AKW - 1)/S) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
    auto gemm_OVTxxx_opt = [&](InputType *input_, WeightsType *weights_,
                         int V1_, int _kh, int _kw, int _I2) {
#pragma nounroll
      for (int _V1 = 0; _V1 < V1_; _V1 += 2) {
        unroll_auto(_O, JO) {
          mmwei[_O][0] = op_load_weights<JO, P>(xc, weights_, _I2, _V1, 0, _O);
          mmwei[_O][1] = op_load_weights<JO, P>(xc, weights_, _I2, _V1 + 1, 0, _O);
        }
        unroll_for(_T, T - (AKW - 2)/S) {
          auto mmbcst0 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1, 0, _T);
          auto mmbcst1 = op_load_input<P>(xc, input_, _kh - AKH, _kw - AKW,
                                          _I2, _V1 + 1, 0, _T);
          unroll_for(_O, JO)
            mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst0, mmbcst1,
                                            mmwei[_O][0], mmwei[_O][1]);
        }
      }
    };
#endif

    if (F_traits<F>::is_compact_ir_weights) {
      MD3(int8_t, aweights, weights, xc.kh, xc.kw, xc.O1 * O * V * Vx); // compact
      for (int _kh = khs; _kh < khe; ++_kh) {
        // mid
        for (int _kw = kws; _kw < kwe; ++_kw) {
          gemm_OVT(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
        }
        // left
        if (pad_l) {
          int _kw = 0; // K = 3, 5, 7
          gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          if (pad_l > 1 && K > 3) {
            _kw = 1; // K = 5, 7
            gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          }
          if (pad_l > 2 && K > 5) {
            _kw = 2; // K = 7
            gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          }
        }
        // right
        if (pad_r) {
          int _kw = K - 1; // K = 3, 5, 7
          gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          if (pad_r > 1 && K > 3) {
            _kw = K - 2; // K = 5, 7
            gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          }
          if (pad_r > 2 && K > 5) {
            _kw = K - 3; // K = 7
            gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), 1, _kh, _kw, 0);
          }
        }
      }
    } else {
      int I2 = xc.I2, Ir = 0;
      if (get_attr(attr, has_Ir_idx)) {
        I2 = xc.I2 - 1;
        Ir = xc.Ir;
      }
      MD3(int8_t, aweights, weights, xc.kh, xc.kw, xc.O1 * xc.I2 * V1 * O * V * Vx); // compact

      for (int _kh = khs; _kh < khe; ++_kh) {
        for (int _I2 = 0; _I2 < I2; ++_I2) {
          // mid
          for (int _kw = kws; _kw < kwe; ++_kw) {
#if defined(WITH_VNNI)
            gemm_OVT(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
            gemm_OVT_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
          }
          // left
          if (pad_l) {
            int _kw = 0; // K = 3, 5, 7
#if defined(WITH_VNNI)
            gemm_OVxT(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
            gemm_OVxT_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            if (pad_l > 1 && K > 3) {
              _kw = 1; // K = 5, 7
#if defined(WITH_VNNI)
              gemm_OVxxT(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
              gemm_OVxxT_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            }
            if (pad_l > 2 && K > 5) {
              _kw = 2; // K = 7
#if defined(WITH_VNNI)
              gemm_OVxxxT(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
              gemm_OVxxxT_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            }
          }
          // right
          if (pad_r) {
            int _kw = K - 1; // K = 3, 5, 7
#if defined(WITH_VNNI)
            gemm_OVTx(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
            gemm_OVTx_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            if (pad_r > 1 && K > 3) {
              _kw = K - 2; // K = 5, 7
#if defined(WITH_VNNI)
              gemm_OVTxx(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
              gemm_OVTxx_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            }
            if (pad_r > 2 && K > 5) {
              _kw = K - 3; // K = 7
#if defined(WITH_VNNI)
              gemm_OVTxxx(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#else
              gemm_OVTxxx_opt(input, &md3(aweights, _kh, _kw, 0), V1, _kh, _kw, _I2);
#endif
            }
          }
        }
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
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
      unroll_for (_T, T) {
        op_restore_output<JO>(xc, output, routput, bias, mmout[_O][_T],
            src_scale, src_factor, weights_scale, weights_factor,
            _O1, _O0, _O, _T, attr);
      }}
    } else {
      unroll_for (_O, JO) {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[_O][_T], _O, _T);
      }}
    }
  }

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<(P == 2 || P == 4), void>::type
  op_conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
      uint8_t *input, int8_t *weights, BiasType *bias, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      int khs, int khe, int kws, int kwe, int pad_l, int pad_r, int attr, int _O1, int _O0)
  {
    // 3x3 conv
    constexpr int AKH = 3 / 2;
    constexpr int AKW = 3 / 2;

    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    MD3(int8_t, aweights, weights, xc.kh, xc.kw, xc.O1 * xc.I2 * V1 * O * V * Vx); // compact

    __i<V> mmout[JO][T], mmwei[JO][P];

    if (get_attr(attr, r_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else {
      // load accumulated s32 output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_load_output<JO>(xc, output, _O, _T);
      }
    }

    for (int _kh = khs; _kh < khe; ++_kh) {
      // mid
      for (int _kw = kws; _kw < kwe; ++_kw) {
        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
#if defined(WITH_VNNI)
            unroll_for(_P, P) {
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P, _O);

              unroll_for(_T, T) {
                auto mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][_P]);
              }
            }
#else
            unroll_for(_P, P / 2) {
              unroll_auto(_O, JO) {
                mmwei[_O][_P * 2] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2, _O);
                mmwei[_O][_P * 2 + 1] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2 + 1, _O);
              }

              unroll_for(_T, T) {
                auto mmbcst1 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2, _T);
                auto mmbcst2 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2 + 1, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst1, mmbcst2,
                                    mmwei[_O][_P * 2], mmwei[_O][_P * 2 + 1]);
              }
            }
#endif
          } // _V1
        }   // I2
        // Ir
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V1 = 0; _V1 < Ir; ++_V1) {
          unroll_auto(_O, JO) mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, 0, _O);

          unroll_for(_T, T) {
            auto mmbcst = op_load_input<1>(xc, input, _kh - AKH, _kw - AKW, _I2,
                                           _V1, 0, _T);
            unroll_for(_O, JO)
              mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
          }
        } // _V1
      }   // _kw loop, mid

      // left
      if (pad_l) {
        constexpr int _kw = 0;

        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
#if defined(WITH_VNNI)
            unroll_for(_P, P) {
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P, _O);

              unroll_from_to(_T, 1, T) {
                auto mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][_P]);
              }
            }
#else
            unroll_for(_P, P / 2) {
              unroll_auto(_O, JO) {
                mmwei[_O][_P * 2] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2, _O);
                mmwei[_O][_P * 2 + 1] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2 + 1, _O);
              }

              unroll_from_to(_T, 1, T) {
                auto mmbcst1 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2, _T);
                auto mmbcst2 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2 + 1, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst1, mmbcst2,
                                    mmwei[_O][_P * 2], mmwei[_O][_P * 2 + 1]);
              }
            }
#endif
          } // _V1
        }   // I2
        // Ir
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V1 = 0; _V1 < Ir; ++_V1) {
          unroll_auto(_O, JO) mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, 0, _O);
          unroll_from_to(_T, 1, T) {
            auto mmbcst = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _I2,
                                           _V1, 0, _T);
            unroll_for(_O, JO)
              mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
          }
        } // _V1
      }   // left

      // right
      if (pad_r) {
        constexpr int _kw = 2;

        for (int _I2 = 0; _I2 < I2; ++_I2) {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
#if defined(WITH_VNNI)
            unroll_for(_P, P) {
              unroll_auto(_O, JO)
                mmwei[_O][_P] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P, _O);

              unroll_for(_T, T - 1 + S - 1) {
                auto mmbcst = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][_P]);
              }
            }
#else
            unroll_for(_P, P / 2) {
              unroll_auto(_O, JO) {
                mmwei[_O][_P * 2] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2, _O);
                mmwei[_O][_P * 2 + 1] = op_load_weights<JO, P>(
                  xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, _P * 2 + 1, _O);
              }

              unroll_for(_T, T - 1 + S - 1) {
                auto mmbcst1 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2, _T);
                auto mmbcst2 = op_load_input<P>(
                    xc, input, _kh - AKH, _kw - AKW, _I2, _V1, _P * 2 + 1, _T);
                unroll_for(_O, JO)
                  mmout[_O][_T] = op_int8_fma_opt(mmout[_O][_T], mmbcst1, mmbcst2,
                                    mmwei[_O][_P * 2], mmwei[_O][_P * 2 + 1]);
              }
            }
#endif
          } // _V1
        }
        // Ir
        int _I2 = xc.I2 - 1;
#pragma nounroll
        for (int _V1 = 0; _V1 < Ir; ++_V1) {
          unroll_auto(_O, JO) mmwei[_O][0] = op_load_weights<JO, 1>(
              xc, &md3(aweights, _kh, _kw, 0), _I2, _V1, 0, _O);

          unroll_for(_T, T - 1 + S - 1) {
            auto mmbcst = op_load_input<P>(xc, input, _kh - AKH, _kw - AKW, _I2,
                                           _V1, 0, _T);
            unroll_for(_O, JO)
              mmout[_O][_T] = op_int8_fma(mmout[_O][_T], mmbcst, mmwei[_O][0]);
          }
        } // _V1
      }   // right
    }     // _kh loop

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
      unroll_for (_T, T) {
        op_restore_output<JO>(xc, output, routput, bias, mmout[_O][_T],
            src_scale, src_factor, weights_scale, weights_factor,
            _O1, _O0, _O, _T, attr);
      }}
    } else {
      unroll_for (_O, JO) {
      unroll_for (_T, T) {
        op_store_output<JO>(xc, output, mmout[_O][_T], _O, _T);
      }}
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 1) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
          InputType *input, WeightsType *weights, BiasType *bias,
          ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
          ScaleType *weights_factor, int khs, int khe, int kws, int kwe,
          int pad_l, int pad_r, int attr)
  {
    int V1r = F_traits<F>::is_compact_ir_weights ? xc.Ir : V1;
    MD5(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1, xc.I2 * V1r, O, V * Vx); // compact
    MD2(OutputType, aoutput_blocked, output, xc.O1, O * xc.oh * xc.ow * V);
    MD4(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O *V);
    MD2(RoutputType, aroutput_blocked, routput, xc.O1, O * xc.oh * xc.ow * V);
    MD4(RoutputType, aroutput_nhwc, routput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O *V);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output ? &md4(aoutput_nhwc, 0, 0, _O1, 0)
                                              : &md2(aoutput_blocked, _O1, 0);
      auto rout = F_traits<F>::is_nhwc_output ? &md4(aroutput_nhwc, 0, 0, _O1, 0)
                                              : &md2(aroutput_blocked, _O1, 0);
      op_conv<JO0, JP0, true>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, 0, 0), &md2(abias, _O1, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, 0);
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 2) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
          InputType *input, WeightsType *weights, BiasType *bias,
          ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
          ScaleType *weights_factor, int khs, int khe, int kws, int kwe,
          int pad_l, int pad_r, int attr)
  {
    int V1r = F_traits<F>::is_compact_ir_weights ? xc.Ir : V1;
    MD5(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1, xc.I2 * V1r, O, V * Vx); // compact
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(RoutputType, aroutput_blocked, routput, xc.O1, O, xc.oh * xc.ow * V);
    MD5(RoutputType, aroutput_nhwc, routput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
                                              : &md3(aoutput_blocked, _O1, 0, 0);
      auto rout = F_traits<F>::is_nhwc_output ? &md5(aroutput_nhwc, 0, 0, _O1, 0, 0)
                                              : &md3(aroutput_blocked, _O1, 0, 0);
      op_conv<JO0, JP0, false>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, 0);

      aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
                                         : &md3(aoutput_blocked, _O1, JO0, 0);
      rout = F_traits<F>::is_nhwc_output ? &md5(aroutput_nhwc, 0, 0, _O1, JO0, 0)
                                         : &md3(aroutput_blocked, _O1, JO0, 0);
      op_conv<JO1, JP1, false>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, JO0);
    }
  }

  template <int O = O, int T = T> static inline
      typename std::enable_if<(J_traits<O, T, K_CONV, WeightsType>::J == 3) &&
      (F_traits<F>::is_compact_weights || F_traits<F>::is_compact_ir_weights)>::type
      conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
          InputType *input, WeightsType *weights, BiasType *bias,
          ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
          ScaleType *weights_factor, int khs, int khe, int kws, int kwe,
          int pad_l, int pad_r, int attr)
  {
    int V1r = F_traits<F>::is_compact_ir_weights ? xc.Ir : V1;
    MD5(WeightsType, aweights, weights, xc.kh * xc.kw, xc.O1, xc.I2 * V1r, O, V * Vx); // compact
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(RoutputType, aroutput_blocked, routput, xc.O1, O, xc.oh * xc.ow * V);
    MD5(RoutputType, aroutput_nhwc, routput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
                                              : &md3(aoutput_blocked, _O1, 0, 0);
      auto rout = F_traits<F>::is_nhwc_output ? &md5(aroutput_nhwc, 0, 0, _O1, 0, 0)
                                              : &md3(aroutput_blocked, _O1, 0, 0);
      op_conv<JO0, JP0, false>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, 0, 0), &md3(abias, _O1, 0, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, 0);

      aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
                                         : &md3(aoutput_blocked, _O1, JO0, 0);
      rout = F_traits<F>::is_nhwc_output ? &md5(aroutput_nhwc, 0, 0, _O1, JO0, 0)
                                         : &md3(aroutput_blocked, _O1, JO0, 0);
      op_conv<JO1, JP1, false>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, JO0, 0), &md3(abias, _O1, JO0, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, JO0);

      aout = F_traits<F>::is_nhwc_output ? &md5(aoutput_nhwc, 0, 0, _O1, JO0 + JO1, 0)
                                         : &md3(aoutput_blocked, _O1, JO0 + JO1, 0);
      rout = F_traits<F>::is_nhwc_output ? &md5(aroutput_nhwc, 0, 0, _O1, JO0 + JO1, 0)
                                         : &md3(aroutput_blocked, _O1, JO0 + JO1, 0);
      op_conv<JO2, JP2, false>(xc, aout, rout, input,
          &md5(aweights, 0, _O1, 0, JO0 + JO1, 0), &md3(abias, _O1, JO0 + JO1, 0),
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr, _O1, JO0 + JO1);
    }
  }

};

} // namespace euler
