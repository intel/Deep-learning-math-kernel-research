#ifndef __ELK_CONV_HPP__
#define __ELK_CONV_HPP__

#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

#define MAX_FMA_PRL 36

namespace euler {

// Type: data type
// TarrayType: tranform data type
// T: tile blocking unit
// A: tile size
// K: kernel size
// V: vector size
// I: ISA
// with_bias: has bias
template <bool ...conditions> struct cd_traits {
  enum {border_ind = 0, bias_ind, relu_ind, ip_sum_ind, op_sum_ind};
  constexpr static int c_[] {conditions...};
  constexpr static bool is_border = c_[border_ind];
  constexpr static bool with_bias = c_[bias_ind];
  constexpr static bool with_relu = c_[relu_ind];
  constexpr static bool with_ip_sum = c_[ip_sum_ind];
  static_assert(sizeof...(conditions) == 4,
      "Template argument error! Please specify if border, bias, relu, sum...");
};

template <int ...configs> struct gemm_traits {
  constexpr static int c_[] {configs...};
  enum { instr_set , pack_size, register_group };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int T = c_[register_group];
  static_assert(sizeof...(configs) == 3,
      "Template argument error! Please specify I, V, T...");
};

template <int ...configs> struct winograd_traits {
  constexpr static int c_[] {configs...};
  enum { instr_set = 0, pack_size, tile_size, kernel_size };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int A = c_[tile_size];
  constexpr static int K = c_[kernel_size];
  static_assert(sizeof...(configs) == 4,
      "Template argument error! Please specify I, V, A, K...");
};

template <typename InputType, typename WeightsType, typename OutputType,
     typename BiasType, typename TarrayType, int ...configs>
class convolution_winograd_kernel_base {
protected:
  constexpr static int c_[] {configs...};
  enum { instr_set = 0, pack_size, tile_size, kernel_size };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int A = c_[tile_size];
  constexpr static int K = c_[kernel_size];
  static_assert(sizeof...(configs) == 4,
      "Template argument error! Please specify I, V, A, K...");

  template <bool is_border>
  static inline void __trans_input(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      TarrayType atinput[A][A][V], InputType *input,
      int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool is_border>
  static inline void __trans_inputa(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      TarrayType atinput[A][A][V], InputType *input,
      int wA, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool ...conditions>
  static inline void __trans_output(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      OutputType *output, TarrayType atoutput[A][A][V],
      BiasType *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      TarrayType *toutputa, TarrayType *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      OutputType *output, TarrayType aoutputa[A][A - K + 1][V],
      BiasType *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(TarrayType atweights[A][A][V][V],
      WeightsType aweights[K][K][V][V]);
};

/*
template <typename Type, int ...configs>
class gemm_kernel_base {
  constexpr static int c_[] {configs...};
  enum { instr_set , pack_size, register_group };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int T = c_[register_group];
  static_assert(sizeof...(configs) == 3,
      "Template argument error! Please specify I, V, T...");

public:
  static inline void __gemm(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc, Type *toutput, Type *tinput, Type *tweights,
      bool zero_out);

  static inline void __gemm_tail(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc, Type *toutput, Type *tinput, Type *tweights,
      bool zero_out);
};
*/

template <typename InputType, typename WeightsType, typename OutputType,
     typename BiasType, typename TarrayType, int ...configs>
class convolution_winograd_kernel :
  public convolution_winograd_kernel_base<
    InputType, WeightsType, OutputType, BiasType, TarrayType, configs...> {
  using super = convolution_winograd_kernel_base<
      InputType, WeightsType, OutputType, BiasType, TarrayType, configs...>;
  constexpr static int A = super::A;
  constexpr static int V = super::V;
  constexpr static int K = super::K;
public:
  // Interfaces
  template <bool is_border>
  static void trans_input(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      TarrayType atinput[A][A][V], InputType *input,
      int hA_start, int hA_end, int wA_start, int wA_end) {
    super::template __trans_input<is_border>(
        xc, atinput, input, hA_start, hA_end, wA_start, wA_end);
  }

  template <bool is_border>
  static void trans_inputa(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      TarrayType atinput[A][A][V], InputType *input,
      int wA, int hA_start, int hA_end, int wA_start, int wA_end) {
    super::template __trans_inputa<is_border>(
        xc, atinput, input, wA, hA_start, hA_end, wA_start, wA_end);
  }

  template <bool ...conditions>
  static void trans_output(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType>& xc,
      OutputType* output, TarrayType atoutput[A][A][V],
      BiasType *bias, int hOA_end, int wOA_end) {
    super::template __trans_output<conditions...>(
        xc, output, atoutput, bias, hOA_end, wOA_end);
  }

  template <bool ...conditions>
  static void trans_outputa_th(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType>& xc,
      TarrayType *toutputa, TarrayType *toutput, int Tz, bool stream_out) {
    super::template __trans_outputa_th<conditions...>(
        xc, toutputa, toutput, Tz, stream_out);
  }

  template <bool ...conditions>
  static void trans_outputa_bh(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc,
      OutputType *output, TarrayType atoutputa[A][A - K + 1][V],
      BiasType *bias, int hOA_end, int wOA_end) {
    super::template __trans_outputa_bh<conditions...>(
        xc, output, atoutputa, bias, hOA_end, wOA_end);
  }

  static void trans_weights(
      TarrayType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]) {
    super::__trans_weights(atweights, aweights);
  }
};

/*
template <typename Type, int ...configs>
class gemm_kernel :
  public gemm_kernel_base<Type, configs...> {

  using super = gemm_kernel_base<Type, configs...>;

public:
  static void gemm(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc, Type *toutput, Type *tinput, Type *tweights,
      bool zero_out) {
    super::__gemm(xc, toutput, tinput, tweights, zero_out);
  }

  static void gemm_tail(
      elx_conv_t<InputType, WeightsType, OutputType, BiasType> &xc, Type *toutput, Type *tinput, Type *tweights,
      bool zero_out) {
    super::__gemm_tail(xc, toutput, tinput, tweights, zero_out);
  }
};
*/
} // namespace euler

#endif // __ELK_CONV_HPP__
