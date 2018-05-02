#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
class elx_conv_wino_gemm_t : public elx_conv_t<Type> {
 public:
  elx_conv_wino_gemm_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_wino_gemm_t();

  virtual void direct(Type *input, Type *weights, Type *output, Type *bias) {
    elx_error("Unimplemented");
  }
  virtual void winograd(Type *input, Type *weights, Type *output, Type *bias);

 private:
  void trans_weights(Type *tweights, Type *weights);
  void trans_input(Type *tinput, Type *input, int _t2);
  void trans_output(Type *output, Type *toutput, int _t2);
  void gemm(Type *toutput, Type *tinput, Type *tweights);

  using func_gemm_t = void(elx_conv_t<Type> &xc, Type *toutput, Type *tinput,
                           Type *tweights, bool zero_out);
  using func_trans_input_t = void(elx_conv_t<Type> &xc, Type atinput[A][A][V],
                                  Type *input);
  using func_trans_inputX_t = void(elx_conv_t<Type> &xc, Type atinput[A][A][V],
                                   Type *input, int _hA_start, int _hA_end,
                                   int _wA_start, int _wA_end);
  using func_trans_weights_t = void(Type atweights[A][A][V][V],
                                    Type aweights[K][K][V][V]);
  using func_trans_output_t = void(elx_conv_t<Type> &xc, Type *output,
                                   Type atoutput[A][A][V]);
  using func_trans_outputX_t = void(elx_conv_t<Type> &xc, Type *output,
                                    Type atoutput[A][A][V], int _hOA_end,
                                    int _wOA_end);

  func_gemm_t *ker_gemm_;
  func_trans_input_t *ker_trans_input_;
  func_trans_inputX_t *ker_trans_inputX_;
  func_trans_weights_t *ker_trans_weights_;
  func_trans_output_t *ker_trans_output_;
  func_trans_outputX_t *ker_trans_outputX_;
};

template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
