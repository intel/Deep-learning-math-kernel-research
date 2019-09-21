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

#if defined(WITH_VBMI)
#define PERM_INDEX_MID   \
        63, 47, 31, 15,  \
        62, 46, 30, 14,  \
        61, 45, 29, 13,  \
        60, 44, 28, 12,  \
        59, 43, 27, 11,  \
        58, 42, 26, 10,  \
        57, 41, 25, 9,   \
        56, 40, 24, 8,   \
        55, 39, 23, 7,   \
        54, 38, 22, 6,   \
        53, 37, 21, 5,   \
        52, 36, 20, 4,   \
        51, 35, 19, 3,   \
        50, 34, 18, 2,   \
        49, 33, 17, 1,   \
        48, 32, 16, 0
#define PERM_INDEX_LEFT  \
        0, 31, 15, 0,    \
        0, 30, 14, 0,    \
        0, 29, 13, 0,    \
        0, 28, 12, 0,    \
        0, 27, 11, 0,    \
        0, 26, 10, 0,    \
        0, 25, 9,  0,    \
        0, 24, 8,  0,    \
        0, 23, 7,  0,    \
        0, 22, 6,  0,    \
        0, 21, 5,  0,    \
        0, 20, 4,  0,    \
        0, 19, 3,  0,    \
        0, 18, 2,  0,    \
        0, 17, 1,  0,    \
        0, 16, 0,  0
#define PERM_INDEX_RIGHT \
        0,  0, 31, 15,   \
        0,  0, 30, 14,   \
        0,  0, 29, 13,   \
        0,  0, 28, 12,   \
        0,  0, 27, 11,   \
        0,  0, 26, 10,   \
        0,  0, 25, 9,    \
        0,  0, 24, 8,    \
        0,  0, 23, 7,    \
        0,  0, 22, 6,    \
        0,  0, 21, 5,    \
        0,  0, 20, 4,    \
        0,  0, 19, 3,    \
        0,  0, 18, 2,    \
        0,  0, 17, 1,    \
        0,  0, 16, 0
#else
#define PERM_INDEX_MID   \
         15, 11, 7, 3,   \
         14, 10, 6, 2,   \
         13, 9,  5, 1,   \
         12, 8,  4, 0,   \
         15, 11, 7, 3,   \
         14, 10, 6, 2,   \
         13, 9,  5, 1,   \
         12, 8,  4, 0,   \
         15, 11, 7, 3,   \
         14, 10, 6, 2,   \
         13, 9,  5, 1,   \
         12, 8,  4, 0,   \
         15, 11, 7, 3,   \
         14, 10, 6, 2,   \
         13, 9,  5, 1,   \
         12, 8,  4, 0

#define PERM_INDEX_LEFT  \
         0, 7,  3,  0,   \
         0, 6,  2,  0,   \
         0, 5,  1,  0,   \
         0, 4,  0,  0,   \
         0, 7,  3,  0,   \
         0, 6,  2,  0,   \
         0, 5,  1,  0,   \
         0, 4,  0,  0,   \
         0, 7,  3,  0,   \
         0, 6,  2,  0,   \
         0, 5,  1,  0,   \
         0, 4,  0,  0,   \
         0, 7,  3,  0,   \
         0, 6,  2,  0,   \
         0, 5,  1,  0,   \
         0, 4,  0,  0

#define PERM_INDEX_RIGHT \
         0, 0,  7,  3,   \
         0, 0,  6,  2,   \
         0, 0,  5,  1,   \
         0, 0,  4,  0,   \
         0, 0,  7,  3,   \
         0, 0,  6,  2,   \
         0, 0,  5,  1,   \
         0, 0,  4,  0,   \
         0, 0,  7,  3,   \
         0, 0,  6,  2,   \
         0, 0,  5,  1,   \
         0, 0,  4,  0,   \
         0, 0,  7,  3,   \
         0, 0,  6,  2,   \
         0, 0,  5,  1,   \
         0, 0,  4,  0 
#endif

namespace euler {

template <typename GarrayTypes, typename RoutputType, int V, int Vx, int I, typename KP>
struct u8s8_depthwise_conv_kernel_otj {
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
struct u8s8_depthwise_conv_kernel_otj<GarrayTypes, RoutputType, V, Vx,
  ISA_SKX_AVX512, estl::integer_sequence<Kp...>> {
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

  // INT8 gemm kernel
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

  static inline __i<V> op_load_input(elx_conv_params_t &xc, InputType *input,
      const int _ih, const int _iw, const int _g2, const int _T, __mmask64 k,
      __m512i index_8)
  {
    __i<V> res;
#if defined(WITH_VNNI)
    if (F_traits<F>::is_blocked_input) {
      MD4(InputType, ainput0, input, xc.I2, xc.ih, xc.iw, V);
      MD3(InputType, ainput1, &md4(ainput0, _g2, _ih, _iw, 0), T, S, V);
      __i<V> in = _mm<V>::load_epi32(&md3(ainput1, _T, 0, 0));
#if defined(WITH_VBMI)
      res = _mm512_maskz_permutexvar_epi8(k, index, in);
#else
      __i<V> index_32 = _mm<V>::set_epi32(
          15, 11, 7, 3,
          14, 10, 6, 2,
          13, 9,  5, 1,
          12, 8,  4, 0);
      auto in_32 = _mm512_permutexvar_epi32(index_32, in);
      res = _mm512_maskz_shuffle_epi8(k, in_32, index_8);
#endif
    } else {
      el_error("elk: not supported input format");
    }
#else
    el_error("u8s8_depthwise: only support VNNI");
#endif
    return res;
  }

  static inline __i<V> op_load_weights(elx_conv_params_t &xc,
      WeightsType *weights, const int _g2, const int _kh)
  {
    __i<V> res;
#if defined(WITH_VNNI)
    MD3(int8_t, aweights, weights, xc.ic2, xc.kh, V * Vx);
    if (F_traits<F>::is_compact_weights) {
      res = _mm<V>::load_epi32(&md3(aweights, _g2, _kh, 0));
    } else {
      el_error("elk: not supported weights format");
    }
#else
    el_error("u8s8_depthwise: only support VNNI");
#endif
    return res;
  }

  static inline void op_restore_output(elx_conv_params_t &xc, OutputType *output,
      RoutputType *routput, BiasType *bias, __i<V> res, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      const int _T, const int attr)
  {
    MD2(RoutputType, aroutput_blocked, routput, T, V);
    MD1(float, aweights_scale, weights_scale, V);
    MD1(float, aweights_factor, weights_factor, V);

    auto rout = &md2(aroutput_blocked, _T, 0);
    __m<V> fout = _mm<V>::cvtepi32_ps(res);

    // restore and requantization
    auto scale = *(__m<V> *)&md1(aweights_scale, 0);
    auto factor = *(__m<V> *)&md1(aweights_factor, 0);
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
    }
  }

  static inline __i<V> op_restore_output_u32(elx_conv_params_t &xc, OutputType *output,
      RoutputType *routput, BiasType *bias, __i<V> res, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      const int _T, const int attr)
  {
    //static_assert(std::is_same<RoutputType, uint8_t>::value, "Expect U8 output only");

    MD1(float, aweights_scale, weights_scale, V);
    MD1(float, aweights_factor, weights_factor, V);

    auto scale = *(__m<V> *)weights_scale;
    auto factor = *(__m<V> *)weights_factor;

    // restore and requantization
    __m<V> fout = _mm<V>::cvtepi32_ps(res);
    fout = fout * scale + factor;

    // fuse relu
    if (get_attr(attr, relu_idx)) {
      auto lower = *(__m<V> *)(xc.relu_bound_lower_vec);
      auto upper = *(__m<V> *)(xc.relu_bound_upper_vec);
      fout = _mm<V>::max_ps(fout, lower);
      fout = _mm<V>::min_ps(fout, upper);
    }

    // return output
    __i<V> u32 = _mm<V>::cvt_roundps_epi32(
        fout, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    return u32;
  }

  static void printv(__i<V> vec, const char *tag) {
    printf("%s: ", tag);
    for (int i = 0; i < V; ++i)
      printf("0x%08x ", ((unsigned int*)&vec)[i]);
    printf("\n");
  }

  static inline void
  op_conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
      uint8_t *input, int8_t *weights, BiasType *bias, ScaleType *src_scale,
      ScaleType *src_factor, ScaleType *weights_scale, ScaleType *weights_factor,
      int khs, int khe, int kws, int kwe, int pad_l, int pad_r, int attr)
  {
    const int AKH = xc.kh / 2;
    constexpr int AKW = K / 2;

    __mmask64 k_mid = _cvtu64_mask64(0xffffffffffffffffUL);
    __mmask64 k_left = _cvtu64_mask64(0x6666666666666666UL);
    __mmask64 k_right = _cvtu64_mask64(0x3333333333333333UL);
    __i<V> index_mid = _mm512_set_epi8(PERM_INDEX_MID);
    __i<V> index_left = _mm512_set_epi8(PERM_INDEX_LEFT);
    __i<V> index_right = _mm512_set_epi8(PERM_INDEX_RIGHT);
    __i<V> mmout[T], mminp;

    MD2(BiasType, abias, bias, xc.ic2, V);
    MD2(ScaleType, aweights_scale, weights_scale, xc.ic2, V);
    MD2(ScaleType, aweights_factor, weights_factor, xc.ic2, V);
    MD2(RoutputType, aroutput, routput, xc.ic2, xc.oh * xc.ow * V);

    for (int _g2 = 0; _g2 < xc.g2; ++_g2) {
      // clear output
      unroll_for (_T, T) {
        mmout[_T] = _mm<V>::setzero_epi32();;
      }

      if (S == 1) {
        __i<V> mmwei_x210, mmwei_210x;
        if (!pad_l && !pad_r) { // mid
          for (int _kh = khs; _kh < khe; ++_kh) {
            mmwei_x210 = op_load_weights(xc, weights, _g2, _kh);
            mmwei_210x = _mm512_slli_epi32(mmwei_x210, 8);

            unroll_for(_T, (T + 1) / 2) {
              mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, 2 * _T, k_mid, index_mid);
              mmout[2 * _T] = op_int8_fma(mmout[2 * _T], mminp, mmwei_x210);
              if ((2 * _T + 1) < T)
                mmout[2 * _T + 1] = op_int8_fma(mmout[2 * _T + 1], mminp, mmwei_210x);
            }
          }
        } else if (pad_l && !pad_r) { // left-mid
          for (int _kh = khs; _kh < khe; ++_kh) {
            mmwei_x210 = op_load_weights(xc, weights, _g2, _kh);
            mmwei_210x = _mm512_slli_epi32(mmwei_x210, 8);

            mminp = op_load_input(xc, input, _kh - AKH, 0, _g2, 0, k_left, index_left);
            mmout[0] = op_int8_fma(mmout[0], mminp, mmwei_x210);

            unroll_for(_T, T / 2) {
              mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, 2 * _T + 1, k_mid, index_mid);
              mmout[2 * _T + 1] = op_int8_fma(mmout[2 * _T + 1], mminp, mmwei_x210);
              if ((2 * _T + 2) < T)
                mmout[2 * _T + 2] = op_int8_fma(mmout[2 * _T + 2], mminp, mmwei_210x);
            }
          }
        } else if (!pad_l && pad_r) { // mid-right
          for (int _kh = khs; _kh < khe; ++_kh) {
            mmwei_x210 = op_load_weights(xc, weights, _g2, _kh);
            mmwei_210x = _mm512_slli_epi32(mmwei_x210, 8);

            unroll_for(_T, T / 2) {
              mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, 2 * _T, k_mid, index_mid);
              mmout[2 * _T] = op_int8_fma(mmout[2 * _T], mminp, mmwei_x210);
              if ((2 * _T + 1) < (T - 1))
                mmout[2 * _T + 1] = op_int8_fma(mmout[2 * _T + 1], mminp, mmwei_210x);
            }
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, T - 1, k_right, index_right);
            mmout[T - 1] = op_int8_fma(mmout[T - 1], mminp, mmwei_x210);
          }
        } else { // left-mid-right
          for (int _kh = khs; _kh < khe; ++_kh) {
            mmwei_x210 = op_load_weights(xc, weights, _g2, _kh);
            mmwei_210x = _mm512_slli_epi32(mmwei_x210, 8);

            mminp = op_load_input(xc, input, _kh - AKH, 0, _g2, 0, k_left, index_left);
            mmout[0] = op_int8_fma(mmout[0], mminp, mmwei_x210);

            unroll_for(_T, (T - 1) / 2) {
              mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, 2 * _T + 1, k_mid, index_mid);
              mmout[2 * _T + 1] = op_int8_fma(mmout[2 * _T + 1], mminp, mmwei_x210);
              if ((2 * _T + 2) < (T - 1))
                mmout[2 * _T + 2] = op_int8_fma(mmout[2 * _T + 2], mminp, mmwei_210x);
            }
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, T - 1, k_right, index_right);
            mmout[T - 1] = op_int8_fma(mmout[T - 1], mminp, mmwei_x210);
          }
        }
      } else {
        __i<V> mmwei;
        for (int _kh = khs; _kh < khe; ++_kh) {
          mmwei = op_load_weights(xc, weights, _g2, _kh);
          // left
          if (pad_l) {
            mminp = op_load_input(xc, input, _kh - AKH, 0, _g2, 0, k_left, index_left);
            mmout[0] = op_int8_fma(mmout[0], mminp, mmwei);
          } else {
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, 0, k_mid, index_mid);
            mmout[0] = op_int8_fma(mmout[0], mminp, mmwei);
          }
          // mid
          unroll_from_to(_T, 1, T - 1) {
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, _T, k_mid, index_mid);
            mmout[_T] = op_int8_fma(mmout[_T], mminp, mmwei);
          }
          // right
          if (pad_r) {
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, T - 1, k_right, index_right);
            mmout[T-1] = op_int8_fma(mmout[T - 1], mminp, mmwei);
          } else {
            mminp = op_load_input(xc, input, _kh - AKH, -1, _g2, T - 1, k_mid, index_mid);
            mmout[T-1] = op_int8_fma(mmout[T - 1], mminp, mmwei);
          }
        }
      }

      // store output
      if (std::is_same<RoutputType, uint8_t>::value) {
        unroll_for(_T, T / 4) {
          __i<V> out_u32, out_u32_tmp;
          // restore
          out_u32 = op_restore_output_u32(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[4 * _T],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), _T, attr);
          out_u32_tmp = op_restore_output_u32(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[4 * _T + 1],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), _T, attr);
          out_u32_tmp = _mm512_slli_epi32(out_u32_tmp, 8);
          out_u32 += out_u32_tmp;
          out_u32_tmp = op_restore_output_u32(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[4 * _T + 2],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), _T, attr);
          out_u32_tmp = _mm512_slli_epi32(out_u32_tmp, 16);
          out_u32 += out_u32_tmp;
          out_u32_tmp = op_restore_output_u32(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[4 * _T + 3],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), _T, attr);
          out_u32_tmp = _mm512_slli_epi32(out_u32_tmp, 24);
          out_u32 += out_u32_tmp;

          __i<V> index_32 = _mm<V>::set_epi32(
              15, 11, 7, 3,
              14, 10, 6, 2,
              13, 9,  5, 1,
              12, 8,  4, 0);

          // transpose
          out_u32 = _mm512_shuffle_epi8(out_u32, index_mid);
          out_u32 = _mm512_permutexvar_epi32(index_32, out_u32);

          // store
          MD2(RoutputType, arout, &md2(aroutput, _g2, 0), T, V);
          _mm<V>::store_epi32(&md2(arout, 4 * _T, 0), out_u32);
        }
        unroll_for (_T, T % 4) {
          op_restore_output(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[4 * (T / 4) + _T],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), 4 * (T / 4) + _T, attr);
        }
      } else {
        unroll_for (_T, T) {
          op_restore_output(xc, output,
                            &md2(aroutput, _g2, 0),
                            &md2(abias, _g2, 0), mmout[_T],
                            src_scale, src_factor,
                            &md2(aweights_scale, _g2, 0),
                            &md2(aweights_factor, _g2, 0), _T, attr);
        }
      }
    }
  }

  template <int O = O, int T = T> static inline void
      conv(elx_conv_params_t &xc, OutputType *output, RoutputType *routput,
          InputType *input, WeightsType *weights, BiasType *bias,
          ScaleType *src_scale, ScaleType *src_factor, ScaleType *weights_scale,
          ScaleType *weights_factor, int khs, int khe, int kws, int kwe,
          int pad_l, int pad_r, int attr)
  {
      op_conv(xc, output, routput, input,
          weights, bias,
          src_scale, src_factor, weights_scale, weights_factor,
          khs, khe, kws, kwe, pad_l, pad_r, attr);
  }

};

} // namespace euler
