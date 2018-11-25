#ifndef __ELX_CONV_DIRECT_1X1_HPP__
#define __ELX_CONV_DIRECT_1X1_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "elk_conv_wino.hpp"
#include "kernel/elk_gemm_otj.hxx"

namespace euler {

#define Template_elx_conv_direct_1x1_t                                         \
  template <typename UserTypes, typename TarrayType, const int V, const int I>

#define Instance_elx_conv_direct_1x1_t                                         \
    elx_conv_direct_1x1_t<UserTypes, TarrayType, V, I>

Template_elx_conv_direct_1x1_t
class elx_conv_direct_1x1_t : public elx_conv_t<UserTypes> {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  public:
  elx_conv_direct_1x1_t(eld_conv_t<UserTypes> &dc);
  virtual ~elx_conv_direct_1x1_t();

  virtual void execute(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);

  private:
  void __execute_a061(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
  void __execute_f061(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
  void __execute_b061(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
  void __execute_e060(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
  void __execute_c060(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);
  void __execute_d060(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias);

  inline void __trans_input_plain(InputType *tinput, InputType *input, int _ht, int _wt);
  inline void __trans_input_blocked(InputType *tinput, InputType *input, int _ht, int _wt);
  void trans_input(InputType *tinput, InputType *input, int _ht, int _wt);

  inline void __trans_pad_input_plain(InputType *tinput, InputType *input, int _ht, int _wt);
  inline void __trans_pad_input_blocked(InputType *tinput, InputType *input, int _ht, int _wt);

  inline void __trans_input_plain2(InputType *tinput, InputType *input, int _t2, int Tz);
  inline void __trans_input_blocked2(InputType *tinput, InputType *input, int _t2, int Tz);
  void trans_input2(InputType *tinput, InputType *input, int _t2, int Tz);

  inline void __trans_output_plain(OutputType *output, OutputType *toutput, int _oc4, int _ht, int _wt);
  inline void __trans_output_blocked(OutputType *output, OutputType *toutput, int _oc4, int _ht, int _wt);
  void trans_output(OutputType *output, OutputType *toutput, int _oc4, int _ht, int _wt);

  inline void __trans_output_plain2(OutputType *output, OutputType *toutput, int _oc4, int _t2, int Tz);
  inline void __trans_output_blocked2(OutputType *output, OutputType *toutput, int _oc4, int _t2, int Tz);
  void trans_output2(OutputType *output, OutputType *toutput, int _oc4, int _t2, int Tz);

  inline void __trans_weights_plain(WeightsType *tweights, WeightsType *weights);
  inline void __trans_weights_blocked(WeightsType *tweights, WeightsType *weights);
  void trans_weights(WeightsType *tweights, WeightsType *weights);

  void gemm_a061(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _ic4);
  void gemm_f061(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _t2, int Tz);
  void gemm_b061(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _ic4);
  void gemm_e060(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _ic4);
  void gemm_c060(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _ic4, int _oc4, int _t2);
  void gemm_d060(OutputType *toutput, InputType *tinput, WeightsType *tweights, BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);

  void trans_input_2_blocked(InputType *tinput, InputType *input);
  void trans_weights_2_blocked(WeightsType *tweghts, WeightsType *weights);
  void trans_output_2_plain(OutputType *output, OutputType *toutput);

  void set_trans_buffers();
  int prepare_execute_opt();
  void bind_execute_functions();

  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_I_O_T_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_I_O_Tr_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_I_OrT_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_I_OrTr_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_IrO_T_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_IrO_Tr_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_IrOrT_;
  gemm_kernel_binder::ker<conv_impl::FP32> *ker_gemm_IrOrTr_;

  void (elx_conv_direct_1x1_t::*execute_opt_)(OutputType *, InputType *, WeightsType *, BiasType *);

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

  WeightsType *tweights_;
  InputType *tinput_;
  OutputType *toutput_;
  unsigned char *tinput_msk_;
  InputType *binput_; // blocked input
  WeightsType *bweights_;
  OutputType *boutput_;

  unsigned int xopt_;
  int attr_;
  int mthr_;
  size_t tweights_size_;
  size_t tinput_size_;
  size_t toutput_size_;
  size_t binput_size_;
  size_t bweights_size_;
  size_t boutput_size_;
  TarrayType *scratch_;
};

/*
template<>
class elx_conv_direct_1x1_t<conv::FP16, 16, ISA_SKX_AVX512> : public elx_conv_t<conv::FP16> {
  public:
  elx_conv_direct_1x1_t(eld_conv_t<conv::FP16> &dc) : elx_conv_t<short>(dc) {}
  virtual void execute(short *output, short *input, short *weights, short *bias) {}
};*/

template class elx_conv_direct_1x1_t<conv::FP32, float, 16, ISA_SKX_AVX512>;
template class elx_conv_direct_1x1_t<conv::FP32, float, 8, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
