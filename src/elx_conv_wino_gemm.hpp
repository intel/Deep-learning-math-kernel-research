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
  inline void __execute_ttm(
      Type *output, Type *input, Type *weights, Type *bias);
  inline void __execute(Type *output, Type *input, Type *weights, Type *bias);
  inline void __execute_flat(Type *output, Type *input, Type *weights, Type *bias);

  void trans_weights(Type *tweights, Type *weights);
  void trans_input(Type *tinput, Type *input, int _t2, int Tz);
  void trans_output(Type *output, Type *toutput, Type *bias, int _t2, int Tz);

  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm_;
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm0_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(false))>::trans_input) *ker_trans_input_;
  decltype(convolution_winograd_kernel<S_INPUT(
          Type, A, K, V, I, BORDER(true))>::trans_input) *ker_trans_input0_;
  decltype(
      convolution_winograd_kernel<S_WEIGHTS(Type, A, K, V, I)>::trans_weights)
      *ker_trans_weights_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(false), BIAS(false))>::trans_output) *ker_trans_output_;
  decltype(convolution_winograd_kernel<S_OUTPUT(Type, A, K, V, I,
          BORDER(true), BIAS(false))>::trans_output) *ker_trans_output0_;

  bool is_first_run_;
  bool inference_acc_;
  size_t mthr_;
  Type *tweights_;
  Type *tinput_;
  Type *toutput_;

#define MAX_THREAD_TEAMS (8)
  // tasks allocation per thread team
  struct { int start; int end; } ttm_[MAX_THREAD_TEAMS];
};

template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
