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

  virtual void execute(Type *input, Type *weights, Type *output, Type *bias);

  private:
  void trans_weights(Type *tweights, Type *weights);
  void trans_input(Type *tinput, Type *input, int _t2);
  void trans_output(Type *output, Type *toutput, Type *bias, int _t2);
  void gemm(Type *toutput, Type *tinput, Type *tweights);

  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::gemm)
      *ker_gemm_;
  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::trans_input)
      *ker_trans_input_;
  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::trans_input0)
      *ker_trans_input0_;
  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::trans_weights)
      *ker_trans_weights_;
  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::trans_output)
      *ker_trans_output_;
  decltype(
      convolution_winograd_kernel<Type, 0, A, K, V, I, false>::trans_output0)
      *ker_trans_output0_;

  size_t mthr_;
  Type *tweights_;
  Type *tinput_;
  Type *toutput_;
};

template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
