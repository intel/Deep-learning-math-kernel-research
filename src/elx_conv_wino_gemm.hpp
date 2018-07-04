#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "elk_conv.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
class elx_conv_wino_gemm_t : public elx_conv_t<Type> {
  public:
  elx_conv_wino_gemm_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_wino_gemm_t();

  virtual void execute(Type *output, Type *input, Type *weights, Type *bias);

  private:

  void __execute_a000(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a040(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a061(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a0e1(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a0e0(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a073(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a201(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a241(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a448(Type *output, Type *input, Type *weights, Type *bias);

  inline void __trans_input(Type *tinput, Type *input, int _t2, int Tz);
  void trans_input(Type *tinput, Type *input, int _t2, int Tz);
  void trans_input(Type *tinput, Type *input);

  inline void __trans_output(
      Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  void trans_output(Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  void trans_output(Type *res, Type *output, Type *toutput, Type *bias,
      int _t2, int Tz, int ic4, int oc4, bool inline_reduce);
  void trans_output(Type *output, Type *toutput, Type *bias);
  void trans_weights(Type *tweights, Type *weights, int oc4 = 1);

  void gemm(
      Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz);
  void gemm(Type *toutput, Type *tinput, Type *tweights);

  // Fusion along a (wA)
  void trans_inputa(Type *tinput, Type *input, int _t2, int _wA, int Tz);
  void trans_weightsa(Type *tweights, Type *weights);
  void gemma(Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz);
  void trans_outputa_th(Type *toutputa, Type *toutput, int Tz);
  void trans_outputa_bh(Type *output, Type *toutputa, Type *bias);

  int prepare_execute_opt();
  void bind_execute_functions();

  // Helpers for input/outout offset computing
  inline void t2spati(int _t2, int _T, int &_n, int &_ih, int &_iw,
      int &_hA_start, int &_hA_end, int &_wA_start, int &_wA_end);
  inline void t2spato(int _t2, int _T, int &_n, int &_oh, int &_ow,
      int &_hOA_end, int &_wOA_end);

  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm_;
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm0_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(false))>::trans_input) *ker_trans_input_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(true))>::trans_input) *ker_trans_input0_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(false))>::trans_inputa) *ker_trans_inputa_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(true))>::trans_inputa) *ker_trans_inputa0_;
  decltype(
      convolution_winograd_kernel<S_WEIGHTS(Type, A, K, V, I)>::trans_weights)
      *ker_trans_weights_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(false), BIAS(false))>::trans_output) *ker_trans_output_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(true), BIAS(false))>::trans_output) *ker_trans_output0_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(false), BIAS(false))>::trans_output) *ker_trans_output_nobias_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(true), BIAS(false))>::trans_output) *ker_trans_output0_nobias_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(false), BIAS(false))>::trans_outputa_th) *ker_trans_outputa_th_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(false), BIAS(false))>::trans_outputa_bh) *ker_trans_outputa_bh_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(true), BIAS(false))>::trans_outputa_bh) *ker_trans_outputa0_bh_;

  void (elx_conv_wino_gemm_t::*execute_opt_)(Type *, Type *, Type *, Type *);

  unsigned int xopt_;
  bool is_first_run_;
  bool inference_acc_;
  bool stream_in_;
  bool stream_out_;
  bool stream_wei_;
  size_t mthr_;
  Type *tweights_;
  Type *tinput_;
  Type *toutput_;
  Type *routput_; // reduce output
  Type *toutputa_;
  unsigned char *routput_cntr_;

#define MAX_THREAD_TEAMS (8)
  // tasks allocation per thread team
  struct { int start; int end; } ttm_[MAX_THREAD_TEAMS];
};

template class elx_conv_wino_gemm_t<float, 4, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 4, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
