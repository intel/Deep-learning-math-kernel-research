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
// T: tile blocking unit
// A: tile size
// K: kernel size
// V: vector size
// I: ISA
// with_bias: has bias
template <bool ...conditions> struct cd_traits {
  enum {border_ind = 0, bias_ind, relu_ind, sum_ind};
  constexpr static int c_[] {conditions...};
  constexpr static bool is_border = c_[border_ind];
  constexpr static bool with_bias = c_[bias_ind];
  constexpr static bool with_relu = c_[relu_ind];
  constexpr static bool with_sum = c_[sum_ind];
};

template <typename Type, int ...configs>
class convolution_winograd_kernel_base {
protected:
  constexpr static int c_[] {configs...};
  enum { instr_set = 0, pack_size, tile_size, kernel_size };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int A = c_[tile_size];
  constexpr static int K = c_[kernel_size];

  template <bool is_border>
  static inline void __trans_input(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool is_border>
  static inline void __trans_inputa(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int wA, int hT_start, int hT_end, int wT_start, int wT_end);

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<Type> &xc, Type *output,
      Type *atoutput[A][A][V], Type *bias, int hOA_end, int wOA_end);

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<Type> &xc, Type *toutputa,
      Type *toutput, int Tz, bool stream_out);

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<Type> &xc, Type *output,
      Type *aoutputa[A][A - K + 1][V], Type *bias, int hOA_end, int wOA_end);

  static inline void __trans_weights(Type atweights[A][A][V][V],
      Type aweights[K][K][V][V]);
};

template <typename Type, int ...configs>
class convolution_gemm_base {
  constexpr static int c_[] {configs...};
  enum { instr_set , pack_size, register_group };

  constexpr static int I = c_[instr_set];
  constexpr static int V = c_[pack_size];
  constexpr static int T = c_[register_group];

public:
  static inline void __gemm(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out);

  static inline void __gemm_tail(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out);
};

template <typename Type, int ...configs>
class convolution_winograd_kernel :
  public convolution_winograd_kernel_base<Type, configs...> {
  using super = convolution_winograd_kernel_base<Type, configs...>;
  constexpr static int A = super::A;
  constexpr static int V = super::V;
  constexpr static int K = super::K;
public:
  // Interfaces
  template <bool is_border>
  static void trans_input(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int hA_start, int hA_end, int wA_start, int wA_end) {
    __trans_input(xc, atinput, input, hA_start, hA_end,
        wA_start, wA_end);
  }

  template <bool is_border>
  static void trans_inputa(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int wA, int hA_start, int hA_end, int wA_start,
      int wA_end) {
    __trans_inputa(xc, atinput, input, wA, hA_start,
        hA_end, wA_start, wA_end);
  }

  template <bool ...conditions>
  static void trans_output(elx_conv_t<Type>& xc, Type* output,
      Type atoutput[A][A][V], Type *bias, int hOA_end, int wOA_end) {
    __trans_output(xc, output, atoutput, bias, hOA_end,
        wOA_end);
  }

  template <bool ...conditions>
  static void trans_outputa_th(elx_conv_t<Type>& xc,
      Type *toutputa, Type *toutput, int Tz, bool stream_out) {
    __trans_outputa_th(xc, toutputa, toutput, Tz, stream_out);
  }

  template <bool ...conditions>
  static void trans_outputa_bh(elx_conv_t<Type> &xc, Type *output,
      Type atoutputa[A][A - K + 1][V], Type *bias, int hOA_end, int wOA_end) {
    __trans_outputa_bh(xc, output, atoutputa, bias, hOA_end, wOA_end);
  }

  static void trans_weights(
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]) {
    __trans_weights(atweights, aweights);
  }
};

template <typename Type, int ...configs>
class convolution_gemm :
  public convolution_gemm_base<Type, configs...> {

  using convolution_gemm_base<Type, configs...>::__gemm;
  using convolution_gemm_base<Type, configs...>::__gemm_tail;

public:
  static void gemm(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    __gemm(xc, toutput, tinput, tweights, zero_out);
  }

  static void gemm_tail(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    __gemm_tail(xc, toutput, tinput, tweights, zero_out);
  }
};

} // namespace euler

#endif // __ELK_CONV_HPP__
