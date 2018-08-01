#ifndef __ELX_CONV_DIRECT_1X1_HPP__
#define __ELX_CONV_DIRECT_1X1_HPP__

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
  void __execute_a000(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_a061(Type *output, Type *input, Type *weights, Type *bias);

  inline void __trans_input_plain(Type *tinput, Type *input);
  inline void __trans_input_blocked(Type *tinput, Type *input);
  void trans_input(Type *tinput, Type *input);

  inline void __trans_input_plain(Type *tinput, Type *input, int _t2, int Tz);
  inline void __trans_input_blocked(Type *tinput, Type *input, int _t2, int Tz);
  void trans_input(Type *tinput, Type *input, int _t2, int Tz);

  inline void __trans_output_plain(Type *output, Type *toutput, Type *bias);
  inline void __trans_output_blocked(Type *output, Type *toutput, Type *bias);
  void trans_output(Type *output, Type *toutput, Type *bias);

  inline void __trans_output_plain(Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  inline void __trans_output_blocked(Type *output, Type *toutput, Type *bias, int _t2, int Tz);
  void trans_output(Type *output, Type *toutput, Type *bias, int _t2, int Tz);

  inline void __trans_weights_plain(Type *tweights, Type *weights);
  inline void __trans_weights_blocked(Type *tweights, Type *weights);
  void trans_weights(Type *tweights, Type *weights);

  void gemm(Type *toutput, Type *tinput, Type *tweights);
  void gemm(Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz);

  int prepare_execute_opt();
  void bind_execute_functions();

  // TODO
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm_;
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm0_;
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm_tail_;
  decltype(
      convolution_winograd_kernel<S_GEMM(Type, 1, V, I)>::gemm) *ker_gemm0_tail_;

  void (elx_conv_direct_1x1_t::*execute_opt_)(Type *, Type *, Type *, Type *);

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

  Type *tweights_;
  Type *tinput_;
  Type *toutput_;

  unsigned int xopt_;
  int mthr_;
};

template class elx_conv_direct_1x1_t<float, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
