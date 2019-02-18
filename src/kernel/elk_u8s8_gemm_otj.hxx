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
struct u8s8_gemm_kernel_otj {
  static inline void gemm(
      elx_conv_params_t &, typename GarrayTypes::OutputType *,
      typename GarrayTypes::InputType *,
      typename GarrayTypes::WeightsType *,
      typename GarrayTypes::BiasType *, int,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *,
      typename GarrayTypes::ScaleType *) {}
};

template <typename GarrayTypes, int V, int Vx, int ...Kp>
struct u8s8_gemm_kernel_otj<GarrayTypes, V, Vx, ISA_SKX_AVX512,
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
  constexpr static int JO1 = J_traits<O, T, WeightsType>::O1;
  constexpr static int JP1 = J_traits<O, T, WeightsType>::P1;
  constexpr static int JO2 = J_traits<O, T, WeightsType>::O2;
  constexpr static int JP2 = J_traits<O, T, WeightsType>::P2;

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

    __m<V> fout = _mm<V>::cvtepi32_ps(res);
    auto z = _mm<V>::set1_ps(src_factor[_T]);
    auto acc = *(__m<V> *)&md2(aweights_factor, _O, 0);
    fout -= (z * acc);
    auto Sa = _mm<V>::set1_ps(src_scale[_T]);
    auto Sw = *(__m<V> *)&md2(aweights_scale, _O, 0);
    fout = Sa * Sw * fout;

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
  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 1, void>::type
  op_gemm(elx_conv_params_t &xc, OutputType *output, uint8_t *input,
      int8_t *weights, BiasType *bias, int attr, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale,
      ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx : xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    if (get_attr(attr, has_Ir_idx)) {
      el_error("Unimplement non-64x IC for int8 gemm");
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
        unroll_for (_O, JO)
          mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][0]);
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

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 2, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx: xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    if (get_attr(attr, has_Ir_idx)) {
      el_error("Unimplement non-64x IC for int8 gemm");
    }

    // preload weights
    unroll_for (_O, JO)
      mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);

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
          mmwei[_O][1] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 1, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][0]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][1]);
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

  template <int JO, int P, bool has_Or>
  static inline typename std::enable_if<P == 4, void>::type
  op_gemm(elx_conv_params_t &xc,
      OutputType *output, uint8_t *input, int8_t *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor, int _O1, int _O0)
  {
    __i<V> mmout[JO][T], mmwei[JO][P];
    const int I2_stride
        = F_traits<F>::is_compact_input ? T * V * Vx: xc.ih * xc.iw * V * Vx;
    const int O_stride
        = F_traits<F>::is_compact_output ? T * V : xc.oh * xc.ow * V;

    MD2(OutputType, aoutput, output, JO, O_stride);
    MD2(uint8_t, ainput, input, xc.I2, I2_stride);

    if (get_attr(attr, has_Ir_idx)) {
      el_error("Unimplement non-64x IC for int8 gemm");
    }

    // preload weights
    unroll_for (_O, JO) {
      mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 0, _O);
      mmwei[_O][1] = op_int8_load_weights<JO, P>(xc, weights, 0, 0, 1, _O);
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
          mmwei[_O][2] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 2, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 0, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][0]);
        }
        // _P = 1
        unroll_for (_O, JO)
          mmwei[_O][3] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V, 3, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 1, _T);
          unroll_for (_O, JO) {
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][1]);
          }
        }
        // _P = 2
        unroll_for (_O, JO)
          mmwei[_O][0] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 0, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 2, _T);
          unroll_for (_O, JO)
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][2]);
        }
        // _P = 3
        unroll_for (_O, JO)
          mmwei[_O][1] = op_int8_load_weights<JO, P>(xc, weights, _I2, _V + 1, 1, _O);
        unroll_for (_T, T) {
          __i<V> bcast = op_int8_load_input<P>(&md2(ainput, _I2, 0), _V, 3, _T);
          unroll_for (_O, JO) {
            mmout[_O][_T] = op_int8_fma(mmout[_O][_T], bcast, mmwei[_O][3]);
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
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 1>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride = F_traits<F>::is_compact_weights
                         ? xc.I2 * V * O * V * Vx : O * xc.IC * V;

    MD2(OutputType, aoutput_compact, output, xc.O1, O * T * V);
    MD2(OutputType, aoutput_blocked, output, xc.O1, O * xc.oh * xc.ow * V);
    MD4(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O * V);

    MD2(WeightsType, aweights, weights, xc.O1, W_stride);
    MD2(BiasType, abias, bias, xc.O1, O * V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md4(aoutput_nhwc, 0, 0, _O1, 0)
          : F_traits<F>::is_compact_output ? &md2(aoutput_compact, _O1, 0)
                                           : &md2(aoutput_blocked, _O1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO0, JP0, true>(xc, aout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, src_scale, src_factor, weights_scale,
            weights_factor, _O1, 0);
      } else {
        op_gemm<JO0, JP0, false>(xc, aout, input, &md2(aweights, _O1, 0),
            &md2(abias, _O1, 0), attr, src_scale, src_factor, weights_scale,
            weights_factor, _O1, 0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 2>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? xc.I2 * V : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V * Vx : xc.IC * V;

    MD3(OutputType, aoutput_compact, output, xc.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD4(WeightsType, aweights, weights, xc.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, false>(xc, aout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO1, JP1, true>(xc, aout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0);
      } else {
        op_gemm<JO1, JP1, false>(xc, aout, input, &md4(aweights, _O1, 0, JO0, 0),
            &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0);
      }
    }
  }

  template <int O = O, int T = T>
  static inline typename std::enable_if<J_traits<O, T, WeightsType>::J == 3>::type
  gemm(elx_conv_params_t &xc, OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias, int attr,
      ScaleType *src_scale, ScaleType *src_factor,
      ScaleType *weights_scale, ScaleType *weights_factor)
  {
    const int W_stride0
        = F_traits<F>::is_compact_weights ? xc.I2 * V : 1;
    const int W_stride1
        = F_traits<F>::is_compact_weights ? V * Vx : xc.IC * V;

    MD3(OutputType, aoutput_compact, output, xc.O1, O, T * V);
    MD3(OutputType, aoutput_blocked, output, xc.O1, O, xc.oh * xc.ow * V);
    MD5(OutputType, aoutput_nhwc, output, xc.oh * xc.ow, xc.oc4 * xc.oc3, xc.O1, O, V);

    MD4(WeightsType, aweights, weights, xc.O1, W_stride0, O, W_stride1);
    MD3(BiasType, abias, bias, xc.O1, O, V);

    for (int _O1 = 0; _O1 < xc.O1; ++_O1) {
      auto aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, 0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, 0, 0)
                                           : &md3(aoutput_blocked, _O1, 0, 0);
      op_gemm<JO0, JP0, false>(xc, aout, input, &md4(aweights, _O1, 0, 0, 0),
          &md3(abias, _O1, 0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, 0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0, 0)
          : F_traits<F>::is_compact_output ? &md3(aoutput_compact, _O1, JO0, 0)
                                           : &md3(aoutput_blocked, _O1, JO0, 0);
      op_gemm<JO1, JP1, false>(xc, aout, input, &md4(aweights, _O1, 0, JO0, 0),
          &md3(abias, _O1, JO0, 0), attr, src_scale, src_factor, weights_scale,
          weights_factor, _O1, JO0);
      aout = F_traits<F>::is_nhwc_output
          ? &md5(aoutput_nhwc, 0, 0, _O1, JO0 + JO1, 0)
          : F_traits<F>::is_compact_output
              ? &md3(aoutput_compact, _O1, JO0 + JO1, 0)
              : &md3(aoutput_blocked, _O1, JO0 + JO1, 0);
      if (F_traits<F>::is_nhwc_output && get_attr(attr, has_Or_idx)
          && _O1 == xc.O1 - 1) {
        op_gemm<JO2, JP2, true>(xc, aout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0 + JO1);
      } else {
        op_gemm<JO2, JP2, false>(xc, aout, input,
            &md4(aweights, _O1, 0, JO0 + JO1, 0),
            &md3(abias, _O1, JO0 + JO1, 0), attr, src_scale, src_factor,
            weights_scale, weights_factor, _O1, JO0 + JO1);
      }
    }
  }

};

} // namespace euler
