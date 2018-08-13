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
template <typename Type, const int T, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias,
    const bool with_relu, const bool with_sum>
struct winograd_template_parameter_t {
};

#define BIAS(x) x
#define BORDER(x) x
#define RELU(x) x
#define SUM(x) x

#define D_INPUT(d_type, d_a, d_k, d_v, d_i, d_border)                          \
  d_type, const int T, d_a, d_k, d_v, d_i, d_border,                           \
      const bool with_bias, const bool with_relu, const bool with_sum
#define R_INPUT(type, a, k, v, i, border) type, T, a, k, v, i, border,         \
  with_bias, with_relu, with_sum
#define S_INPUT(type, a, k, v, i, border)                                      \
  type, 0, a, k, v, i, border, BIAS(false), RELU(false), SUM(false)

#define D_OUTPUT(d_type, d_a, d_k, d_v, d_i, d_border, d_bias, d_relu, d_sum)  \
  d_type, const int T, d_a, d_k, d_v, d_i, d_border, d_bias, d_relu, d_sum
#define R_OUTPUT(type, a, k, v, i, border, bias, relu, sum)                    \
  type, T, a, k, v, i, border, bias, relu, sum
#define S_OUTPUT(type, a, k, v, i, border, bias, relu, sum)                    \
  type, 0, a, k, v, i, border, bias, relu, sum

#define D_GEMM(d_type, d_t, d_v, d_i)                                          \
  d_type, d_t, const int A, const int K, d_v, d_i, const bool is_border,       \
      const bool with_bias, const bool with_relu, const bool with_sum
#define R_GEMM(type, t, v, i) type, t, A, K, v, i, is_border,                  \
  with_bias, with_relu, with_sum
#define S_GEMM(type, t, v, i) type, t, 0, 0, v, i, BORDER(false),              \
  BIAS(false), RELU(false), SUM(false)

#define D_WEIGHTS(d_type, d_a, d_k, d_v, d_i)                                  \
  d_type, const int T, d_a, d_k, d_v, d_i, const bool is_border,               \
      const bool with_bias, const bool with_relu, const bool with_sum
#define R_WEIGHTS(type, a, k, v, i) type, T, a, k, v, i, is_border,            \
  with_bias, with_relu, with_sum
#define S_WEIGHTS(type, a, k, v, i)                                            \
  type, 0, a, k, v, i, BORDER(false), BIAS(false), RELU(false), SUM(false)

template <typename Type, const int T, const int A, const int K, const int V,
    const int I, const bool is_border, const bool with_bias,
    const bool with_relu, const bool with_sum>
struct convolution_winograd_kernel {
  // Interfaces
  static void trans_input(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int _hA_start, int _hA_end, int _wA_start, int _wA_end);

  static void trans_inputa(elx_conv_t<Type> &xc, Type atinput[A][A][V],
      Type *input, int _wA, int _hA_start, int _hA_end, int _wA_start,
      int _wA_end);

  static void trans_output(elx_conv_t<Type>& xc, Type* output,
      Type atoutput[A][A][V], Type *bias, int _hOA_end, int _wOA_end);

  static void trans_outputa_th(elx_conv_t<Type>& xc,
      Type *toutputa, Type *toutput, int Tz, bool stream_out);

  static void trans_outputa_bh(elx_conv_t<Type> &xc, Type *output,
      Type atoutputa[A][A - K + 1][V], Type *bias, int _hOA_end, int _wOA_end);

  static void trans_weights(
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]);

  static void gemm(elx_conv_t<Type> &xc, Type *toutput, Type *tinput,
      Type *tweights, bool zero_out);

  static void gemm_tail(elx_conv_t<Type> &xc, Type *toutput, Type *tinput,
      Type *tweights, bool zero_out);

#define TRANS_KERNEL(Type_, _A, _K, _V, _I)                                  \
  template <const bool is_border_>                                           \
  static inline void __trans_input(winograd_template_parameter_t<S_INPUT(    \
                                       Type_, _A, _K, _V, _I, is_border_)>,  \
      elx_conv_t<Type_> &xc, Type_ atinput[A][A][V], Type_ *input,           \
      int _hT_start, int _hT_end, int _wT_start, int _wT_end);               \
                                                                             \
  template <const bool is_border_>                                           \
  static inline void __trans_inputa(winograd_template_parameter_t<S_INPUT(   \
                                        Type_, _A, _K, _V, _I, is_border_)>, \
      elx_conv_t<Type_> &xc, Type_ atinput[A][A][V], Type_ *input, int _wA,  \
      int _hT_start, int _hT_end, int _wT_start, int _wT_end);               \
                                                                             \
  template <const bool is_border_, const bool with_bias_,                    \
      const bool with_relu_, const bool with_sum_>                           \
  static inline void __trans_output(                                         \
      winograd_template_parameter_t<S_OUTPUT(Type_, _A, _K, _V, _I,          \
          is_border_, with_bias_, with_relu_, with_sum_)>,                   \
      elx_conv_t<Type_> &xc, Type_ *output, Type_ atoutput[A][A][V],         \
      Type_ *bias, int _hOA_end, int _wOA_end);                              \
                                                                             \
  static inline void __trans_outputa_th(                                     \
      winograd_template_parameter_t<S_OUTPUT(                                \
          Type_, _A, _K, _V, _I, false, false, false, false)>,               \
      elx_conv_t<Type_> &xc, Type_ *toutputa, Type_ *toutput, int Tz,        \
      bool stream_out);                                                      \
                                                                             \
  template <const bool is_border_, const bool with_bias_,                    \
      const bool with_relu_, const bool with_sum_>                           \
  static inline void __trans_outputa_bh(                                     \
      winograd_template_parameter_t<S_OUTPUT(Type_, _A, _K, _V, _I,          \
          is_border_, with_bias_, with_relu_, with_sum_)>,                    \
      elx_conv_t<Type_> &xc, Type_ *output,                                  \
      Type_ atoutputa[A][A - K + 1][V], Type_ *bias, int _hOA_end,           \
      int _wOA_end);                                                         \
                                                                             \
  static inline void __trans_weights(                                        \
      winograd_template_parameter_t<S_WEIGHTS(Type_, _A, _K, _V, _I)>,       \
      Type atweights[A][A][V][V], Type aweights[K][K][V][V]);

  TRANS_KERNEL(float, 4, 3, 16, ISA_GENERIC);
  TRANS_KERNEL(float, 4, 3, 16, ISA_SKX_AVX512);

  TRANS_KERNEL(float, 5, 3, 16, ISA_GENERIC);
  TRANS_KERNEL(float, 5, 3, 16, ISA_SKX_AVX512);

  TRANS_KERNEL(float, 6, 3, 16, ISA_GENERIC);
  TRANS_KERNEL(float, 6, 3, 16, ISA_SKX_AVX512);

  TRANS_KERNEL(float, 7, 3, 16, ISA_GENERIC);
  TRANS_KERNEL(float, 7, 3, 16, ISA_SKX_AVX512);

  template <const int T_>
  static inline void __gemm(
      winograd_template_parameter_t<S_GEMM(float, T_, 16, ISA_GENERIC)>,
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out);

  template <const int T_>
  static inline void __gemm_tail(
      winograd_template_parameter_t<S_GEMM(float, T_, 16, ISA_GENERIC)>,
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out);

#define DEF_gemm(z, n, nil)                                                    \
  static inline void __gemm(                                                   \
      winograd_template_parameter_t<S_GEMM(float, n, 16, ISA_SKX_AVX512)>,     \
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,   \
      bool zero_out);
#define DEF_gemm_tail(z, n, nil)                                               \
  static inline void __gemm_tail(                                              \
      winograd_template_parameter_t<S_GEMM(float, n, 16, ISA_SKX_AVX512)>,     \
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,   \
      bool zero_out);

  BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, DEF_gemm, nil);
  BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, DEF_gemm_tail, nil);

  // Generic
  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_, const bool with_bias_,
      const bool with_relu_, const bool with_sum_>
  static inline void __trans_input(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type_> &, Type_[A_][A_][V_], Type_ *, int, int, int, int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __trans_inputa(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type_> &, Type_[A_][A_][V_], Type_ *, int, int, int,
      int, int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __trans_output(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type_> &, Type_ *, Type_[A_][A_][V_], Type_ *, int, int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __trans_outputa_th(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type_> &, Type_ *, Type_ *, int, bool)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __trans_outputa_bh(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type_> &, Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int,
      int)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __trans_weights(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      Type[A_][A_][V_][V_], Type[K_][K_][V_][V_]);

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __gemm(
      winograd_template_parameter_t<Type_, T_, A_, K_, V_, I_, is_border_,
          with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type> &, Type *, Type *, Type *, bool)
  {
  }

  template <typename Type_, const int T_, const int A_, const int K_,
      const int V_, const int I_, const bool is_border_,
      const bool with_bias_, const bool with_relu_, const bool with_sum_>
  static inline void __gemm_tail(
      winograd_template_parameter_t<Type_, T_, A_,
          K_, V_, I_, is_border_, with_bias_, with_relu_, with_sum_>,
      elx_conv_t<Type> &, Type *, Type *, Type *, bool)
  {
  }
};

} // namespace euler

#define __TRANS_WEIGHTS(Type_, A_, K_, V_, I_)                               \
  template <D_WEIGHTS(                                                       \
      typename Type, const int A, const int K, const int V, const int I)>    \
  void                                                                       \
  convolution_winograd_kernel<R_WEIGHTS(Type, A, K, V, I)>::__trans_weights( \
      winograd_template_parameter_t<S_WEIGHTS(Type_, A_, K_, V_, I_)>,       \
      Type atweights[A][A][V][V], Type aweights[K][K][V][V])

#define TRANS_WEIGHTS(Type_, A_, K_, V_, I_)                                 \
  template void convolution_winograd_kernel<S_WEIGHTS(Type_, A_, K_, V_,     \
      I_)>::trans_weights(Type_[A_][A_][V_][V_], Type_[K_][K_][V_][V_]);

#define __TRANS_INPUT(Type_, A_, K_, V_, I_)                                 \
  template <D_INPUT(typename Type, const int A, const int K, const int V,    \
      const int I, const bool is_border)>                                    \
  template <const bool is_border_>                                           \
  void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I, is_border)>::   \
      __trans_input(winograd_template_parameter_t<S_INPUT(                   \
                        Type_, A_, K_, V_, I_, is_border_)>,                 \
          elx_conv_t<Type_> &xc, Type_ atinput[A][A][V], Type_ *input,       \
          int _hT_start, int _hT_end, int _wT_start, int _wT_end)

#define TRANS_INPUT(Type_, A_, K_, V_, I_)                                   \
  template void convolution_winograd_kernel<S_INPUT(Type_, A_, K_, V_, I_,   \
      BORDER(false))>::trans_input(elx_conv_t<Type_> &, Type_[A_][A_][V_],   \
      Type_ *, int, int, int, int);                                          \
                                                                             \
  template void convolution_winograd_kernel<S_INPUT(Type_, A_, K_, V_, I_,   \
      BORDER(true))>::trans_input(elx_conv_t<Type_> &, Type_[A_][A_][V_],    \
      Type_ *, int, int, int, int);

#define __TRANS_INPUTA(Type_, A_, K_, V_, I_)                                \
  template <D_INPUT(typename Type, const int A, const int K, const int V,    \
      const int I, const bool is_border)>                                    \
  template <const bool is_border_>                                           \
  void convolution_winograd_kernel<R_INPUT(Type, A, K, V, I, is_border)>::   \
      __trans_inputa(winograd_template_parameter_t<S_INPUT(                  \
                         Type_, A_, K_, V_, I_, is_border_)>,                \
          elx_conv_t<Type_> &xc, Type_ atinput[A][A][V], Type_ *input,       \
          int _wA, int _hT_start, int _hT_end, int _wT_start, int _wT_end)

#define TRANS_INPUTA(Type_, A_, K_, V_, I_)                                  \
  template void convolution_winograd_kernel<S_INPUT(Type_, A_, K_, V_, I_,   \
      BORDER(false))>::trans_inputa(elx_conv_t<Type_> &, Type_[A_][A_][V_],  \
      Type_ *, int, int, int, int, int);                                     \
                                                                             \
  template void convolution_winograd_kernel<S_INPUT(Type_, A_, K_, V_, I_,   \
      BORDER(true))>::trans_inputa(elx_conv_t<Type_> &, Type_[A_][A_][V_],   \
      Type_ *, int, int, int, int, int);

#define __TRANS_OUTPUT(Type_, A_, K_, V_, I_)                                \
  template <D_OUTPUT(typename Type, const int A, const int K, const int V,   \
      const int I, const bool is_border, const bool with_bias,               \
      const bool with_relu, const bool with_sum)>                            \
  template <const bool is_border_, const bool with_bias_,                    \
      const bool with_relu_, const bool with_sum_>                           \
  void convolution_winograd_kernel<R_OUTPUT(                                 \
      Type, A, K, V, I, is_border, with_bias, with_relu, with_sum)>::        \
      __trans_output(                                                        \
          winograd_template_parameter_t<S_OUTPUT(                            \
              Type_, A_, K_, V_, I_, is_border_, with_bias_,                 \
              with_relu_, with_sum_)>,                                       \
          elx_conv_t<Type_> &xc, Type_ *output, Type_ atoutput[A][A][V],     \
          Type_ *bias, int _hOA_end, int _wOA_end)

#define TRANS_OUPUT(Type_, A_, K_, V_, I_)                                   \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(false), SUM(false))>::                 \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(false), SUM(false))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(false), SUM(false))>::                \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(false), SUM(false))>::                 \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(true), SUM(false))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(true), SUM(false))>::                   \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(true), SUM(false))>::                 \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(true), SUM(false))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(false), SUM(true))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(false), SUM(true))>::                   \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(false), SUM(true))>::                 \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(false), SUM(true))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(true), SUM(true))>::                   \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(true), SUM(true))>::                    \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(true), SUM(true))>::                  \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);                        \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(true), SUM(true))>::                   \
      trans_output(elx_conv_t<Type_> &,                                      \
      Type_ *, Type_[A_][A_][V_], Type_ *, int, int);

#define __TRANS_OUTPUTA_TH(Type_, A_, K_, V_, I_)                            \
  template <D_OUTPUT(typename Type, const int A, const int K, const int V,   \
      const int I, const bool is_border,                                     \
      const bool with_bias, const bool with_relu, const bool with_sum)>      \
  void convolution_winograd_kernel<R_OUTPUT(                                 \
      Type, A, K, V, I, is_border, with_bias, with_relu, with_sum)>::        \
      __trans_outputa_th(                                                    \
          winograd_template_parameter_t<S_OUTPUT(                            \
              Type_, A_, K_, V_, I_, false, false, false, false)>,           \
          elx_conv_t<Type_> &xc, Type_ *toutputa, Type_ *toutput, int Tz,    \
          bool stream_out)

#define TRANS_OUTPUTA_TH(Type_, A_, K_, V_, I_)                              \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(false), SUM(false))>::                \
      trans_outputa_th(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_ *, int, bool);

#define __TRANS_OUTPUTA_BH(Type_, A_, K_, V_, I_)                            \
  template <D_OUTPUT(typename Type, const int A, const int K, const int V,   \
      const int I, const bool is_border,                                     \
      const bool with_bias, const bool with_relu, const bool with_sum)>      \
  template <const bool is_border_,                                           \
      const bool with_bias_, const bool with_relu_, const bool with_sum_>    \
  void convolution_winograd_kernel<R_OUTPUT(                                 \
      Type, A, K, V, I, is_border, with_bias, with_relu, with_sum)>::        \
      __trans_outputa_bh(                                                    \
          winograd_template_parameter_t<S_OUTPUT(Type_, A_,                  \
              K_, V_, I_, is_border_, with_bias_, with_relu_, with_sum_)>,   \
          elx_conv_t<Type_> &xc, Type_ *output,                              \
          Type_ atoutput[A][A - K + 1][V], Type_ *bias, int _hOA_end,        \
          int _wOA_end)

#define TRANS_OUTPUTA_BH(Type_, A_, K_, V_, I_)                              \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(false), SUM(false))>::                 \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(false), SUM(false))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(false), SUM(false))>::                \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(false), SUM(false))>::                 \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(true), SUM(false))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(true), SUM(false))>::                   \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(true), SUM(false))>::                 \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(true), SUM(false))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(false), SUM(true))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(false), SUM(true))>::                   \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(false), SUM(true))>::                 \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(false), SUM(true))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(true), RELU(true), SUM(true))>::                   \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(true), RELU(true), SUM(true))>::                    \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(false), BIAS(false), RELU(true), SUM(true))>::                  \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);               \
  template void convolution_winograd_kernel<S_OUTPUT(Type_, A_, K_, V_, I_,  \
      BORDER(true), BIAS(false), RELU(true), SUM(true))>::                   \
      trans_outputa_bh(elx_conv_t<Type_> &,                                  \
      Type_ *, Type_[A_][A_ - K_ + 1][V_], Type_ *, int, int);

#endif // __ELK_CONV_HPP__
