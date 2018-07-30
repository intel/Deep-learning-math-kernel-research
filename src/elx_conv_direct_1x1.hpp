#ifndef __ELX_CONV_DIRECT_1x1_HPP__
#define __ELX_CONV_DIRECT_1x1_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename Type, const int V, const int I>
class elx_conv_direct_1x1_t : public elx_conv_t<Type> {
  public:
  elx_conv_direct_1x1_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_direct_1x1_t();

  virtual void execute(Type *output, Type *input, Type *weights, Type *bias);

  private:
  bool is_first_run_;
  bool inference_acc_;

  bool stream_in_;
  bool stream_out_;
  bool stream_wei_;

  bool is_bfmt_;
  bool input_is_bfmt_;
  bool weights_is_bfmt_;
  bool output_is_bfmt_;
  bool input_as_bfmt_;
  bool weights_as_bfmt_;
  bool output_as_bfmt_;

  int mthr_;
};

template class elx_conv_direct_1x1_t<float, 16, ISA_GENERIC>;
template class elx_conv_direct_1x1_t<float, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1x1_HPP__
