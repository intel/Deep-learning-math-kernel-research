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
  constexpr static int J = J_traits<O, T, WeightsType>::J;
  constexpr static int JO0 = J_traits<O, T, WeightsType>::O0;
  constexpr static int JP0 = J_traits<O, T, WeightsType>::P0;
  constexpr static int MP0 = J_traits<O, T, WeightsType>::P0 == 1
      ? 2 : J_traits<O, T, WeightsType>::P0;
  constexpr static int JO1 = J_traits<O, T, WeightsType>::O1;
  constexpr static int JP1 = J_traits<O, T, WeightsType>::P1;
  constexpr static int MP1 = J_traits<O, T, WeightsType>::P1 == 1
      ? 2 : J_traits<O, T, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, WeightsType>::O2;
  constexpr static int JP2 = J_traits<O, T, WeightsType>::P2;
  constexpr static int MP2 = J_traits<O, T, WeightsType>::P2 == 1
      ? 2 : J_traits<O, T, WeightsType>::P2;

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
  static inline __i<V> op_int8_load_input(
      uint8_t *input, const int _V1, const int _P, const int _T)
  {
    static_assert(F_traits<F>::is_compact_input || F_traits<F>::is_blocked_input,
                  "only compact and blocked format input enabled");

    MD5(uint8_t, ainput5, input, T, S, V1 / P, P, Vx);
    return _mm<V>::set1_epi32(*(int32_t*)&md5(ainput5, _T, 0, _V1, _P, 0));
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
      OutputType *output, OoutputType *ooutput, BiasType *bias, __i<V> res,
      ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
      ScaleType *weights_factor, const int _O1, const int _O0, const int _O,
      const int _T, const int attr)
  {
    MD2(OutputType, aoutput2, output, T, V);
    MD2(OoutputType, aooutput2, ooutput, T, V);
    MD3(float, aweights_scale3, weights_scale, xc.O1, O, V);
    MD2(float, aweights_scale, &md3(aweights_scale3, _O1, _O0, 0), JO, V);
    MD3(float, aweights_factor3, weights_factor, xc.O1, O, V);
    MD2(float, aweights_factor, &md3(aweights_factor3, _O1, _O0, 0), JO, V);

    __m<V> fout = _mm<V>::cvtepi32_ps(res);

    // add bias (direct conv 1x1)
    __m<V> fbias;
    if (get_attr(attr, bias_idx)) {
      MD2(BiasType, abias2, bias, JO, V);
      if (std::is_same<BiasType, float>::value) {
        fbias = _mm<V>::load_ps(&md2(abias2, _O, 0));
      } else {
        auto fp16v = _mm<V / 2>::load_si256((__m256i *)&md2(abias2, _O, 0));
        fbias = _mm<V>::cvtph_ps(fp16v);
      }
    } else {
      fbias = _mm<V>::set1_ps(0.0);
    }

    // requantization
    if (std::is_same<OoutputType, uint8_t>::value
        || std::is_same<OoutputType, int8_t>::value) {
      // global sampling for input/output (direct conv 1x1)
      __m<V> S = *(__m<V> *)&md2(aweights_scale, _O, 0);
      __m<V> z = *(__m<V> *)&md2(aweights_factor, _O, 0);
      fout = fout * S + z;
    } else {
      auto z = _mm<V>::set1_ps(src_factor[_T]);
      auto acc = *(__m<V> *)&md2(aweights_factor, _O, 0);
      fout -= (z * acc);
      auto Sa = _mm<V>::set1_ps(src_scale[_T]);
      auto Sw = *(__m<V> *)&md2(aweights_scale, _O, 0);
      fout = Sa * Sw * fout + fbias;
    }

    // fuse sum
    if (get_attr(attr, ip_sum_idx)) {
      if (std::is_same<OoutputType, uint8_t>::value
          || std::is_same<OoutputType, int8_t>::value) {
        __m<V> sum_S = _mm<V>::set1_ps(xc.sum_quant_S);
        __m128i &mmoo = *(__m128i *)&md2(aooutput2, _T, 0);
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
      fout = _mm<V>::max_ps(fout, _mm<V>::setzero_ps());
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
      _mm_store_si128((__m128i *)&md2(aooutput2, _T, 0), mmresx8);
    } else {
      if (std::is_same<OutputType, float>::value)
        _mm<V>::store_ps(&md2(aoutput2, _T, 0), fout);
      else {
        auto fp16v = _mm<V>::cvtps_ph(
            fout, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)&md2(aoutput2, _T, 0), fp16v);
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
    __i<V> mmout[JO][T], mmwei[JO][MP];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V1 * Vx: xc.ih * xc.iw * V1 * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(OoutputType, aooutput, ooutput, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    // preload weights
#if !defined(WITH_VNNI)
    if (get_attr(attr, fma_opt_idx)) {
      if (P == 1) {
        unroll_for(_O, JO) {
          mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);
          mmwei[_O][1] = op_int8_load_weights<JO, P>(xc, weights, 0, 1, 0, _O);
        }
      } else {
        unroll_for(_O, JO) {
          unroll_for(_P, P) {
            mmwei[_O][_P] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, _P, _O);
          }
        }
      }
    } else
#endif
    {
      unroll_for(_P, P) {
        unroll_for(_O, JO) {
          mmwei[_O][_P] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, _P, _O);
        }
      }
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
          mmout[_O][_T] = op_int8_load_output(&md2(aoutput, _O, 0), _T);
      }
    }

#if !defined(WITH_VNNI)
    if (get_attr(attr, fma_opt_idx)) {
      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
        if (P == 1) {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1; _V1 += 2) {
            unroll_for(_T, T) {
              __i<V> bcast1 =
                  op_int8_load_input<P>(&md2(ainput, _I2, 0), _V1, 0, _T);
              __i<V> bcast2 =
                  op_int8_load_input<P>(&md2(ainput, _I2, 0), _V1 + 1, 0, _T);
              unroll_for(_O, JO) op_int8_fma_opt(
                  mmout[_O][_T], bcast1, bcast2, mmwei[_O][0], mmwei[_O][1]);
            }
            unroll_for(_O, JO) {
              mmwei[_O][0] =
                  op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 2, 0, _O);
              mmwei[_O][1] =
                  op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 3, 0, _O);
            }
          }
        } else {
#pragma nounroll
          for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
            unroll_for(_P, P / 2) {
              unroll_for(_T, T) {
                __i<V> bcast1 =
                    op_int8_load_input<P>(&md2(ainput, _I2, 0), _V1, _P * 2, _T);
                __i<V> bcast2 =
                    op_int8_load_input<P>(&md2(ainput, _I2, 0), _V1, _P * 2 + 1, _T);
                unroll_for(_O, JO) op_int8_fma_opt(
                    mmout[_O][_T], bcast1, bcast2, mmwei[_O][_P * 2], mmwei[_O][_P * 2 + 1]);
              }
              unroll_for(_O, JO) {
                mmwei[_O][_P * 2] =
                    op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 1, _P * 2, _O);
                mmwei[_O][_P * 2 + 1] =
                    op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 1, _P * 2 + 1, _O);
              }
            }
          }
        }
      }
    } else
#endif
    {
      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
        for (int _V1 = 0; _V1 < V1 / P; ++_V1) {
          unroll_for(_P, P) {
            unroll_for(_T, T) {
              __i<V> bcast =
                  op_int8_load_input<P>(&md2(ainput, _I2, 0), _V1, _P, _T);
              unroll_for(_O, JO) op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][_P]);
            }
            unroll_for(_O, JO) mmwei[_O][_P] =
                op_int8_load_weights<JO, P>(xc, weights, _I2, _V1 + 1, _P, _O);
          }
        }
      }
    }

    // store output
    if (get_attr(attr, c_output_idx)) {
      unroll_for (_O, JO) {
      unroll_for (_T, T) {
        op_int8_restore_output<JO>(xc, &md2(aoutput, _O, 0),
            &md2(aooutput, _O, 0), bias, mmout[_O][_T],
            src_scale, src_factor, weights_scale,
            weights_factor, _O1, _O0, _O, _T, attr);
      }}
    } else { // For 1x1/direct. Store accumulated s32 output
      unroll_for (_O, JO) {
      unroll_for (_T, T)
        op_int8_store_output(&md2(aoutput, _O, 0), mmout[_O][_T], _T);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 1>::type
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
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 2>::type
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
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 3>::type
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
