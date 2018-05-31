#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"

#ifndef __ELX_CONV_WINO_PROD_HPP__
#define __ELX_CONV_WINO_PROD_HPP__

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
class elx_conv_wino_prod_t : public elx_conv_t<Type> {
  public:
  elx_conv_wino_prod_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_wino_prod_t();

  virtual void execute(Type *output, Type *input, Type *weights, Type *bias);

  private:
  void trans_weights(Type *tweights, Type *weights);
  void trans_input(Type *tinput, Type *input);
  void product_trans_output(
      Type *output, Type *tinput, Type *tweights, Type *bias);

  Type *tinput_;
  Type *toutput_;
  Type *tweights_;
};

template class elx_conv_wino_prod_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_prod_t<float, 5, 3, 16, ISA_SKX_AVX512>;

} // namespace euler
#endif // __ELX_CONV_WINO_PROD_HPP__
