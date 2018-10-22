#ifndef __ELX_CONV_DIRECT_1X1_HPP__
#define __ELX_CONV_DIRECT_1X1_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "elk_conv_wino.hpp"
#include "kernel/elk_gemm_otj.hxx"

namespace euler {

template <typename Type, const int V, const int I>
class elx_conv_direct_1x1_t : public elx_conv_t<Type> {
  public:
  elx_conv_direct_1x1_t(eld_conv_t<Type> &dc);
  virtual ~elx_conv_direct_1x1_t();

  virtual void execute(Type *output, Type *input, Type *weights, Type *bias);

  private:
  void __execute_a061(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_f061(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_b061(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_e060(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_c060(Type *output, Type *input, Type *weights, Type *bias);
  void __execute_d060(Type *output, Type *input, Type *weights, Type *bias);

  inline void __trans_input_plain(Type *tinput, Type *input, int _ht, int _wt);
  inline void __trans_input_blocked(Type *tinput, Type *input, int _ht, int _wt);
  void trans_input(Type *tinput, Type *input, int _ht, int _wt);

  inline void __trans_pad_input_plain(Type *tinput, Type *input, int _ht, int _wt);
  inline void __trans_pad_input_blocked(Type *tinput, Type *input, int _ht, int _wt);

  inline void __trans_input_plain2(Type *tinput, Type *input, int _t2, int Tz);
  inline void __trans_input_blocked2(Type *tinput, Type *input, int _t2, int Tz);
  void trans_input2(Type *tinput, Type *input, int _t2, int Tz);

  inline void __trans_output_plain(Type *output, Type *toutput, int _oc4, int _ht, int _wt);
  inline void __trans_output_blocked(Type *output, Type *toutput, int _oc4, int _ht, int _wt);
  void trans_output(Type *output, Type *toutput, int _oc4, int _ht, int _wt);

  inline void __trans_output_plain2(Type *output, Type *toutput, int _oc4, int _t2, int Tz);
  inline void __trans_output_blocked2(Type *output, Type *toutput, int _oc4, int _t2, int Tz);
  void trans_output2(Type *output, Type *toutput, int _oc4, int _t2, int Tz);

  inline void __trans_weights_plain(Type *tweights, Type *weights);
  inline void __trans_weights_blocked(Type *tweights, Type *weights);
  void trans_weights(Type *tweights, Type *weights);

  void gemm_a061(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _ic4);
  void gemm_f061(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _t2, int Tz);
  void gemm_b061(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _ic4);
  void gemm_e060(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _ic4);
  void gemm_c060(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _ic4, int _oc4, int _t2);
  void gemm_d060(Type *toutput, Type *tinput, Type *tweights, Type *bias, int _ic4, int _oc4, int _ht, int _wt);

  void trans_input_2_blocked(Type *tinput, Type *input);
  void trans_weights_2_blocked(Type *tweghts, Type *weights);
  void trans_output_2_plain(Type *output, Type *toutput);

  void set_trans_buffers();
  int prepare_execute_opt();
  void bind_execute_functions();

  gemm_kernel_binder::ker *ker_gemm_I_O_T_;
  gemm_kernel_binder::ker *ker_gemm_I_O_Tr_;
  gemm_kernel_binder::ker *ker_gemm_I_OrT_;
  gemm_kernel_binder::ker *ker_gemm_I_OrTr_;
  gemm_kernel_binder::ker *ker_gemm_IrO_T_;
  gemm_kernel_binder::ker *ker_gemm_IrO_Tr_;
  gemm_kernel_binder::ker *ker_gemm_IrOrT_;
  gemm_kernel_binder::ker *ker_gemm_IrOrTr_;

  void (elx_conv_direct_1x1_t::*execute_opt_)(Type *, Type *, Type *, Type *);

  bool no_pad_;
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
  unsigned char *tinput_msk_;
  Type *binput_; // blocked input
  Type *bweights_;
  Type *boutput_;

  unsigned int xopt_;
  int attr_;
  int mthr_;
  size_t tweights_size_;
  size_t tinput_size_;
  size_t toutput_size_;
  size_t binput_size_;
  size_t bweights_size_;
  size_t boutput_size_;
  Type *scratch_;
};

template class elx_conv_direct_1x1_t<float, 16, ISA_SKX_AVX512>;
template class elx_conv_direct_1x1_t<float, 8, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
