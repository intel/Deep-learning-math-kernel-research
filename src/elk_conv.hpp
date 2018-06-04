#ifndef __ELK_CONV_HPP__
#define __ELK_CONV_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

#define IMM_BCAST16(x) x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x
#define MAX_FMA_PRL 33

namespace euler {

// Type: data type
// T: tile blocking unit
// A: tile size
// K: kernel size
// V: vector size
// I: ISA
// with_bias: has bias
template <typename Type, const int T, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias>
struct winograd_template_parameter_t {
};

#define BIAS(x) x
#define BORDER(x) x

#define D_INPUT(d_type, d_a, d_k, d_v, d_i, d_border)                          \
  d_type, const int T, d_a, d_k, d_v, d_i, d_border, const bool with_bias
#define R_INPUT(type, a, k, v, i, border) type, T, a, k, v, i, border, with_bias
#define S_INPUT(type, a, k, v, i, border)                                      \
  type, 0, a, k, v, i, border, BIAS(false)

#define D_OUTPUT(d_type, d_a, d_k, d_v, d_i, d_border, d_bias)                 \
  d_type, const int T, d_a, d_k, d_v, d_i, d_border, d_bias
#define R_OUTPUT(type, a, k, v, i, border, bias)                               \
  type, T, a, k, v, i, border, bias
#define S_OUTPUT(type, a, k, v, i, border, bias)                               \
  type, 0, a, k, v, i, border, bias

#define D_GEMM(d_type, d_t, d_v, d_i)                                          \
  d_type, d_t, const int A, const int K, d_v, d_i, const bool is_border,       \
      const bool with_bias
#define R_GEMM(type, t, v, i) type, t, A, K, v, i, is_border, with_bias
#define S_GEMM(type, t, v, i) type, t, 0, 0, v, i, BORDER(false), BIAS(false)

#define D_WEIGHTS(d_type, d_a, d_k, d_v, d_i)                                  \
  d_type, const int T, d_a, d_k, d_v, d_i, const bool is_border,               \
      const bool with_bias
#define R_WEIGHTS(type, a, k, v, i) type, T, a, k, v, i, is_border, with_bias
#define S_WEIGHTS(type, a, k, v, i)                                            \
  type, 0, a, k, v, i, BORDER(false), BIAS(false)

template <typename Type, const int T, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias>
struct convolution_winograd_kernel {
  // Interfaces
  static void trans_input(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int _hA_start, int _hA_end, int _wA_start, int _wA_end);

  static void trans_output(elx_conv_t<Type>& xc, Type* output,
      Type atoutput[A][A][V], Type* bias, int _hOA_end, int _wOA_end);

  static void trans_weights(
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]);

  static void gemm(elx_conv_t<Type>& xc, Type* toutput, Type* tinput,
      Type* tweights, bool zero_out);

  // C
  template <const bool is_border_>
  static inline void __trans_input(winograd_template_parameter_t<S_INPUT(float,
                                       5, 3, 16, ISA_GENERIC, is_border_)>,
      elx_conv_t<float> &xc, float atinput[5][5][16], float *input,
      int _hT_start, int _hT_end, int _wT_start, int _wT_end);

  template <const bool is_border_, const bool with_bias_>
  static inline void __trans_output(
      winograd_template_parameter_t<S_OUTPUT(
          float, 5, 3, 16, ISA_GENERIC, is_border_, with_bias_)>,
      elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
      float *bias, int _hOA_end, int _wOA_end);

  static inline void __trans_weights(
      winograd_template_parameter_t<S_WEIGHTS(float, 5, 3, 16, ISA_GENERIC)>,
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]);

  template <const int T_>
  static inline void __gemm(
      winograd_template_parameter_t<S_GEMM(float, T_, 16, ISA_GENERIC)>,
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out);

  // AVX512
  template <const bool is_border_>
  static inline void __trans_input(winograd_template_parameter_t<S_INPUT(float,
                                       5, 3, 16, ISA_SKX_AVX512, is_border_)>,
      elx_conv_t<float> &xc, float atinput[5][5][16], float *input,
      int _hT_start, int _hT_end, int _wT_start, int _wT_end);

  template <const bool is_border_, const bool with_bias_>
  static inline void __trans_output(
      winograd_template_parameter_t<S_OUTPUT(
          float, 5, 3, 16, ISA_SKX_AVX512, is_border_, with_bias_)>,
      elx_conv_t<float> &xc, float *output, float atoutput[A][A][V],
      float *bias, int _hOA_end, int _wOA_end);

  static inline void __trans_weights(
      winograd_template_parameter_t<S_WEIGHTS(float, 5, 3, 16, ISA_SKX_AVX512)>,
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]);

#define DEF_gemm(z, n, nil)                                                    \
  static inline void __gemm(                                                   \
      winograd_template_parameter_t<S_GEMM(float, n, 16, ISA_SKX_AVX512)>,     \
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,   \
      bool zero_out);
  BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, DEF_gemm, nil);

  // Generic
  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_, const bool with_bias_>
  static inline void __trans_input(winograd_template_parameter_t<Type_, T_, A_,
                                       K_, V_, I_, is_border_, with_bias_>,
      elx_conv_t<Type_> &, Type_[A_][A_][V_], Type_ *, int, int, int, int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_, const bool with_bias_>
  static inline void __trans_output(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_>,
      elx_conv_t<Type_> &, Type_ *, Type_[A_][A_][V_], Type_ *, int, int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_, const bool with_bias_>
  static inline void __trans_weights(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_>,
      Type[A_][A_][V_][V_], Type[K_][K_][V_][V_]);

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_, const bool with_bias_>
  static inline void __gemm(winograd_template_parameter_t<Type_, T_, A_, K_, V_,
                                I_, is_border_, with_bias_>,
      elx_conv_t<Type> &, Type *, Type *, Type *, bool)
  {
  }
};

} // namespace euler

#endif // __ELK_CONV_HPP__
