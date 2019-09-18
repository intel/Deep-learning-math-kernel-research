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

template <typename GarrayTypes, typename OoutputType, int V, int Vx, int I, typename KP>
struct u8s8_gemm_kernel_otj {
  static inline void gemm(
      elx_conv_params_t &, typename GarrayTypes::OutputType *,
      OoutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *, int,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *) {}
};

template <typename GarrayTypes, typename OoutputType, int V, int Vx, int ...Kp>
struct u8s8_gemm_kernel_otj<GarrayTypes, OoutputType, V, Vx, ISA_SKX_AVX512,
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

  // Jamming components
  constexpr static int J   = J_traits<O, T, K_GEMM, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, K_GEMM, WeightsType>::O0;
  constexpr static int JP0 = J_traits<O, T, K_GEMM, WeightsType>::P0;
  constexpr static int MP0 = J_traits<O, T, K_GEMM, WeightsType>::P0 == 1
      ? 2 : J_traits<O, T, K_GEMM, WeightsType>::P0;
  constexpr static int JO1 = J_traits<O, T, K_GEMM, WeightsType>::O1;
  constexpr static int JP1 = J_traits<O, T, K_GEMM, WeightsType>::P1;
  constexpr static int MP1 = J_traits<O, T, K_GEMM, WeightsType>::P1 == 1
      ? 2 : J_traits<O, T, K_GEMM, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, K_GEMM, WeightsType>::O2;
  constexpr static int JP2 = J_traits<O, T, K_GEMM, WeightsType>::P2;
  constexpr static int MP2 = J_traits<O, T, K_GEMM, WeightsType>::P2 == 1
      ? 2 : J_traits<O, T, K_GEMM, WeightsType>::P2;

  constexpr static int V1 = V / Vx;
  // INT8 gemm kernel
  //
  static inline void op_int8_fma(__i<V>& out, __i<V>& a, __i<V>& b) {
    // TODO: check ISA
#if defined(WITH_VNNI)
    out = _mm512_dpbusds_epi32(out, a, b);
#else
    __i<V> one = _mm<V>::set1_epi16(1);
    __i<V> t0 = _mm<V>::maddubs_epi16(a, b);
    t0 = _mm<V>::madd_epi16(t0, one);
    out = _mm<V>::add_epi32(t0, out);
#endif
  }

  static inline void op_int8_fma_opt(
      __i<V>& out, __i<V>& a1, __i<V>& a2, __i<V>& b1, __i<V>& b2) {
    __i<V> one = _mm<V>::set1_epi16(1);
    __i<V> t1 = _mm<V>::maddubs_epi16(a1, b1);
    __i<V> t2 = _mm<V>::maddubs_epi16(a2, b2);
    t1 = _mm512_adds_epi16(t1, t2);
    t1 = _mm<V>::madd_epi16(t1, one);
    out = _mm<V>::add_epi32(t1, out);
  }

  template <const int P>
  static inline __i<V> op_int8_load_input(elx_conv_params_t &xc, uint8_t *input,
      const int _I2, const int _V1, const int _P, const int _T) {
    if (F_traits<F>::is_compact_input) {
      MD6(uint8_t, ainput, input, xc.I2, T, S, V1 / P, P, Vx);
      return _mm<V>::set1_epi32(*(int32_t*)&md6(ainput, _I2, _T, 0, _V1, _P, 0));
    } else if (F_traits<F>::is_blocked_input) {
      MD3(uint8_t, ainput0, input, xc.I2, xc.ih * xc.iw, V);
      MD5(uint8_t, ainput1, &md3(ainput0, _I2, 0, 0), T, S, V1 / P, P, Vx);
      return _mm<V>::set1_epi32(*(int32_t*)&md5(ainput1, _T, 0, _V1, _P, 0));
    } else {
      assert(F_traits<F>::is_nhwc_input);
      MD4(uint8_t, ainput0, input, xc.wt, T, S, xc.ic);
      MD6(uint8_t, ainput1, &md4(ainput0, 0, _T, 0, 0), xc.ic4, xc.ic3, xc.I2, V1 / P, P, Vx);
      return _mm<V>::set1_epi32(*(int32_t*)&md6(ainput1, 0, 0, _I2, _V1, _P, 0));
    }
  }

  template <const int JO, const int P>
  static inline __i<V> op_int8_load_weights(elx_conv_params_t &xc,
      int8_t *weights, const int _I2, const int _V1, const int _P, const int _O)
  {
    __i<V> res;
    if (F_traits<F>::is_compact_weights) {
      MD5(int8_t, aweights5, weights, xc.I2, V1 / P, P, O, V * Vx);
      res = _mm<V>::load_epi32(&md5(aweights5, _I2, _V1, _P, _O, 0));
    } else {
      MD6(int8_t, aweights6, weights, JO, xc.ic34, xc.I2, V1 / P, P, V * Vx);
      res = _mm<V>::load_epi32(&md6(aweights6, _O, 0, _I2, _V1, _P, 0));
    }
    return res;
  }

  template <int JO>
  static inline __i<V> op_int8_load_output(elx_conv_params_t &xc,
    OutputType *output, const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);

    if (std::is_same<OutputType, float>::value) {
      return _mm<V>::load_epi32((__i<V> *)aout);
    } else {
      auto fp16v = _mm<V / 2>::load_si256((__i<V/2> *)aout);
      return _mm<V>::cvtepi16_epi32(fp16v);
    }
  }

  template <int JO>
  static inline void op_int8_store_output(elx_conv_params_t &xc,
      OutputType *output, __i<V> res, const int _O, const int _T)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);

    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);

    if (std::is_same<OutputType, float>::value) {
      _mm<V>::store_epi32((__i<V> *)aout, res);
    } else {
      _mm<V / 2>::store_si256((__i<V / 2> *)aout, _mm<V>::cvtepi32_epi16(res));
    }
  }

  template <const int JO, bool ip_sum>
  static inline void op_int8_restore_output(elx_conv_params_t &xc,
      OutputType *output, OoutputType *ooutput, BiasType *bias, __i<V> res,
      ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
      ScaleType *weights_factor, const int _O1, const int _O0, const int _O,
      const int _T, const int attr)
  {
    MD3(OutputType, aoutput_compact0, output, JO, T, V);
    MD3(OoutputType, aooutput_compact0, ooutput, JO, T, V);

    MD2(OutputType, aoutput_blocked0, output, JO, xc.oh * xc.ow * V);
    MD2(OutputType, aoutput_blocked1, &md2(aoutput_blocked0, _O, 0), T, V);
    MD2(OoutputType, aooutput_blocked0, ooutput, JO, xc.oh * xc.ow * V);
    MD2(OoutputType, aooutput_blocked1, &md2(aooutput_blocked0, _O, 0), T, V);

    MD2(OutputType, aoutput_nhwc0, output, T, xc.OC);
    MD3(OutputType, aoutput_nhwc1, &md2(aoutput_nhwc0, _T, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);
    MD2(OoutputType, aooutput_nhwc0, ooutput, T, xc.oc);
    MD3(OoutputType, aooutput_nhwc1, &md2(aooutput_nhwc0, _T, 0), xc.oc4 * xc.oc3 * xc.O1, xc.O, V);

    auto aout = F_traits<F>::is_compact_output ? &md3(aoutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aoutput_blocked1, _T, 0) : &md3(aoutput_nhwc1, 0, _O, 0);
    auto aoout = F_traits<F>::is_compact_output ? &md3(aooutput_compact0, _O, _T, 0)
              : F_traits<F>::is_blocked_output
              ? &md2(aooutput_blocked1, _T, 0) : &md3(aooutput_nhwc1, 0, _O, 0);

    MD3(float, aweights_scale3, weights_scale, xc.O1, O, V);
    MD2(float, aweights_scale, &md3(aweights_scale3, _O1, _O0, 0), JO, V);
    MD3(float, aweights_factor3, weights_factor, xc.O1, O, V);
    MD2(float, aweights_factor, &md3(aweights_factor3, _O1, _O0, 0), JO, V);

    __m<V> fout = _mm<V>::cvtepi32_ps(res);

    // requantization
    // global sampling for input/output
    // XXX: fixme
    if (xc.sampling_kind == CALIBRATED, 1) {
      __m<V> S = *(__m<V> *)&md2(aweights_scale, _O, 0);
      __m<V> z = *(__m<V> *)&md2(aweights_factor, _O, 0);
      fout = fout * S + z;
    } else {
      // XXX: TODO: enable  fine/coarse sampling without perf lose
      // Winograd with FINE/COARSE sampling
      auto z = _mm<V>::set1_ps(src_factor[_T]);
      auto acc = *(__m<V> *)&md2(aweights_factor, _O, 0);
      fout -= (z * acc);
      auto Sa = _mm<V>::set1_ps(src_scale[_T]);
      auto Sw = *(__m<V> *)&md2(aweights_scale, _O, 0);
      fout = Sa * Sw * fout;
    }

    // fuse sum
    if (ip_sum) {
      if (std::is_same<OoutputType, uint8_t>::value
          || std::is_same<OoutputType, int8_t>::value) {
        __m<V> sum_S = *(__m<V> *)(xc.sum_quant_S_vec);
        __m128i &mmoo = *(__m128i *)aoout;
        __i<V> mmoos32;
        if (std::is_same<OoutputType, int8_t>::value)
          mmoos32 = _mm<V>::cvtepi8_epi32(mmoo);
        else
          mmoos32 = _mm<V>::cvtepu8_epi32(mmoo);
        auto mmoof32 = _mm<V>::cvtepi32_ps(mmoos32);
        mmoof32 = mmoof32 * sum_S;
        fout += mmoof32;
      } else {
        // no implementation for FP32 output
      }
    }

    // fuse relu
    if (get_attr(attr, relu_idx)) {
      auto lower = *(__m<V> *)(xc.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(xc.relu_bound_upper_vec);
      fout = _mm<V>::max_ps(fout, lower);
      fout = _mm<V>::min_ps(fout, upper);
    }
    // store output
    if (std::is_same<OoutputType, uint8_t>::value
        || std::is_same<OoutputType, int8_t>::value) {
      __i<V> mmress32 = _mm<V>::cvt_roundps_epi32(
          fout, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
      __m128i mmresx8;
      if (std::is_same<OoutputType, int8_t>::value)
        mmresx8 = _mm<V>::cvtsepi32_epi8(mmress32);
      else
        mmresx8 = _mm<V>::cvtusepi32_epi8(mmress32);
      _mm_store_si128((__m128i *)aoout, mmresx8);
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

  // u8s8f32 fma
  template <int JO, int P, int MP, bool has_Or>
  static inline typename std::enable_if<(P == 1 || P == 2 || P == 4), void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, OoutputType *ooutput,
      uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T];
#if defined(WITH_VNNI)
    __i<V> mmwei[JO][P];
#else
    __i<V> mmwei[JO][MP];
#endif

    int I2 = xc.I2, Ir = 0;
    if (get_attr(attr, has_Ir_idx)) {
      I2 = xc.I2 - 1;
      Ir = xc.Ir;
    }

    if (get_attr(attr, r_output_idx)) {
      // clear output
      __i<V> tmp = _mm<V>::setzero_epi32();
      unroll_for (_O, JO)
      unroll_for (_T, T)
        mmout[_O][_T] = tmp;
    } else { // for 1x1 and direct path
      // load accumulated s32 output
      unroll_for (_O, JO) {
        unroll_for (_T, T)
          mmout[_O][_T] = op_int8_load_output<JO>(xc, output, _O, _T);
      }
    }

#if !defined(WITH_VNNI)
    if (get_attr(attr, fma_opt_idx)) {
      for (int _I2 = 0; _I2 < I2; ++_I2) {
        if (P == 1) {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1; _V1 += 2) {
            unroll_for(_O, JO) {
              mmwei[_O][0] =
                  op_int8_load_weights<JO, P>(xc, weights, _I2, _V1, 0, _O);
              mmwei[_O][1] =
                  op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 1, 0, _O);
            }

            unroll_for(_T, T) {
              __i<V> bcast1 =
                  op_int8_load_input<P>(xc, input, _I2, _V1, 0, _T);
              __i<V> bcast2 =
                  op_int8_load_input<P>(xc, input, _I2, _V1 + 1, 0, _T);
              unroll_for(_O, JO) op_int8_fma_opt(
                  mmout[_O][_T], bcast1, bcast2, mmwei[_O][0], mmwei[_O][1]);
            }
          }
        } else {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
            unroll_for(_P, P / 2) {
              unroll_for(_O, JO) {
                mmwei[_O][_P * 2] =
                    op_int8_load_weights<JO, P>(xc, weights, _I2, _V1, _P * 2, _O);
                mmwei[_O][_P * 2 + 1] =
                    op_int8_load_weights<JO, P>(xc, weights, _I2, _V1, _P * 2 + 1, _O);
              }

              unroll_for(_T, T) {
                __i<V> bcast1 =
                    op_int8_load_input<P>(xc, input, _I2, _V1, _P * 2, _T);
                __i<V> bcast2 =
                    op_int8_load_input<P>(xc, input, _I2, _V1, _P * 2 + 1, _T);
                unroll_for(_O, JO) op_int8_fma_opt(
                    mmout[_O][_T], bcast1, bcast2, mmwei[_O][_P * 2], mmwei[_O][_P * 2 + 1]);
              }
            }
          }
        }
      }
    } else
#endif
    {
      for (int _I2 = 0; _I2 < I2; ++_I2) {
//#pragma nounroll
        for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
          unroll_for(_P, P) {
            unroll_for(_O, JO) mmwei[_O][_P] =
                op_int8_load_weights<JO, P>(xc, weights, _I2, _V1, _P, _O);

            unroll_for(_T, T) {
              __i<V> bcast =
                  op_int8_load_input<P>(xc, input, _I2, _V1, _P, _T);
              unroll_for(_O, JO) op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][_P]);
            }
          }
        }
      }
    }

    if (Ir > 0) {
//#pragma nounroll
      for (int _V1 = 0; _V1 < Ir; ++_V1) {
        unroll_for(_O, JO) mmwei[_O][0] =
            op_int8_load_weights<JO, 1>(xc, weights, xc.I2 - 1, _V1, 0, _O);

        unroll_for(_T, T) {
          __i<V> bcast =
              op_int8_load_input<1>(xc, input, xc.I2 - 1, _V1, 0, _T);
          unroll_for(_O, JO) op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][0]);
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      if (get_attr(attr, ip_sum_idx)) {
        unroll_for (_O, JO) {
        unroll_for (_T, T) {
            op_int8_restore_output<JO, true>(xc, output, ooutput, bias,
                mmout[_O][_T], src_scale, src_factor, weights_scale,
                weights_factor, _O1, _O0, _O, _T, attr);
        }}
      } else {
        unroll_for (_O, JO) {
        unroll_for (_T, T) {
          op_int8_restore_output<JO, false>(xc, output, ooutput, bias,
              mmout[_O][_T], src_scale, src_factor, weights_scale,
              weights_factor, _O1, _O0, _O, _T, attr);
        }}
      }
    } else { // Store accumulated s32 output: direct/1x1/wino
      unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_int8_store_output<JO>(xc, output, mmout[_O][_T], _O, _T);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 1>::type
  gemm(elx_conv_params_t &xc, OutputType *output, OoutputType *ooutput,
      InputType *input, WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride = F_traits<F>::is_compact_weights
                         ? xc.I2 * V1 * O * V * Vx : O * xc.IC * V;

    MD2(OutputType, aoutput_compact, output, xc.O1, O * T * V);
    MD2(OutputType, aoutput_blocked, output, xc.O1, O * xc.oh * xc.ow * V);
    MD4(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O * V);

    MD2(OoutputType, aooutput_compact, ooutput, xc.O1, O * T * V);
    MD2(OoutputType, aooutput_blocked, ooutput, xc.O1, O * xc.oh * xc.ow * V);
    MD4(OoutputType, aooutput_nhwc, ooutput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O * V);

    MD2(WeightsType, aweights, weights, xc.O1, W_stride);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md4(aoutput_nhwc, 0, 0, _O1, 0)
          : F_traits<F>::is_compact_output ? &md2(aoutput_compact, _O1, 0)
                                           : &md2(aoutput_blocked, _O1, 0);
      auto aoout = F_traits<F>::is_nhwc_output
          ? &md4(aooutput_nhwc, 0, 0, _O1, 0)
          : F_traits<F>::is_compact_output ? &md2(aooutput_compact, _O1, 0)
                                           : &md2(aooutput_blocked, _O1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO0, JP0, MP0, true>(xc, aout, aoout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, src_scale, src_factor, weights_scale,
            weights_factor, _O1, 0);
      } else {
        op_gemm<JO0, JP0, MP0, false>(xc, aout, aoout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, src_scale, src_factor, weights_scale,
            weights_factor, _O1, 0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 2>::type
  gemm(elx_conv_params_t &xc, OutputType *output, OoutputType *ooutput,
      InputType *input, WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? xc.I2 * V1 : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V * Vx : xc.IC * V;

    MD3(OutputType, aoutput_compact, output, xc.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD3(OoutputType, aooutput_compact, ooutput, xc.O1, O, T * V);
    MD3(OoutputType, aooutput_blocked, ooutput, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OoutputType, aooutput_nhwc, ooutput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD4(WeightsType, aweights, weights, xc.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      auto aoout = F_traits<F>::is_nhwc_output
          ? &md5(aooutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aooutput_compact, _O1, 0, 0)
                                           : &md3(aooutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, MP0, false>(xc, aout, aoout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      aoout = F_traits<F>::is_nhwc_output
          ? &md5(aooutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aooutput_compact, _O1, JO0, 0)
                                           : &md3(aooutput_blocked, _O1, JO0, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO1, JP1, MP1, true>(xc, aout, aoout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0);
      } else {
        op_gemm<JO1, JP1, MP1, false>(xc, aout, aoout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, K_GEMM, WeightsType>::J == 3>::type
  gemm(elx_conv_params_t &xc, OutputType *output, OoutputType *ooutput,
      InputType *input, WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? xc.I2 * V1 : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V * Vx : xc.IC * V;

    MD3(OutputType, aoutput_compact, output, xc.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD3(OoutputType, aooutput_compact, ooutput, xc.O1, O, T * V);
    MD3(OoutputType, aooutput_blocked, ooutput, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OoutputType, aooutput_nhwc, ooutput, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD4(WeightsType, aweights, weights, xc.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      auto aoout = F_traits<F>::is_nhwc_output
          ? &md5(aooutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aooutput_compact, _O1, 0, 0)
                                           : &md3(aooutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, MP0, false>(xc, aout, aoout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      aoout = F_traits<F>::is_nhwc_output
          ? &md5(aooutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aooutput_compact, _O1, JO0, 0)
                                           : &md3(aooutput_blocked, _O1, JO0, 0);
      op_gemm<JO1, JP1, MP1, false>(xc, aout, aoout, input, &md4(aweights, _O1, 0, JO0, 0),
          &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, JO0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0 + JO1, 0)
          : F_traits<F>::is_compact_output
              ? &md3(aoutput_compact, _O1, JO0 + JO1, 0)
              : &md3(aoutput_blocked, _O1, JO0 + JO1, 0);
      aoout = F_traits<F>::is_nhwc_output
          ? &md5(aooutput_nhwc, 0, 0, _O1, JO0 + JO1, 0)
          : F_traits<F>::is_compact_output
              ? &md3(aooutput_compact, _O1, JO0 + JO1, 0)
              : &md3(aooutput_blocked, _O1, JO0 + JO1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO2, JP2, MP2, true>(xc, aout, aoout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0 + JO1);
      } else {
        op_gemm<JO2, JP2, MP2, false>(xc, aout, aoout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0 + JO1);
      }
    }
  }

};

} // namespace euler
