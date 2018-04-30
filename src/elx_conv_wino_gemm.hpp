#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int T, const int V,
          const int I>
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
};

template class elx_conv_wino_gemm_t<float, 5, 3, 25, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 25, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_WINO_GEMM_HPP__
