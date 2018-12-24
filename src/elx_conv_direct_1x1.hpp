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
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_1x1_t                                         \
    elx_conv_direct_1x1_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_1x1_t
class elx_conv_direct_1x1_t : public elx_conv_t<UserTypes> {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;

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

  inline void __trans_input_plain(TinputType *tinput, InputType *input, int _ht, int _wt);
  inline void __trans_input_blocked(TinputType *tinput, InputType *input, int _ht, int _wt);
  void trans_input(TinputType *tinput, InputType *input, int _ht, int _wt);

  inline void __trans_pad_input_plain(TinputType *tinput, InputType *input, int _ht, int _wt);
  inline void __trans_pad_input_blocked(TinputType *tinput, InputType *input, int _ht, int _wt);

  inline void __trans_input_plain2(TinputType *tinput, InputType *input, int _t2, int Tz);
  inline void __trans_input_blocked2(TinputType *tinput, InputType *input, int _t2, int Tz);
  void trans_input2(TinputType *tinput, InputType *input, int _t2, int Tz);

  inline void __trans_output_plain(OutputType *output, ToutputType *toutput, int _oc4, int _ht, int _wt);
  inline void __trans_output_blocked(OutputType *output, ToutputType *toutput, int _oc4, int _ht, int _wt);
  void trans_output(OutputType *output, ToutputType *toutput, int _oc4, int _ht, int _wt);

  inline void __trans_output_plain2(OutputType *output, ToutputType *toutput, int _oc4, int _t2, int Tz);
  inline void __trans_output_blocked2(OutputType *output, ToutputType *toutput, int _oc4, int _t2, int Tz);
  void trans_output2(OutputType *output, ToutputType *toutput, int _oc4, int _t2, int Tz);

  inline void __trans_weights_plain(TweightsType *tweights, WeightsType *weights);
  inline void __trans_weights_blocked(TweightsType *tweights, WeightsType *weights);
  void trans_weights(TweightsType *tweights, WeightsType *weights);

  void gemm_a061(ToutputType *toutput, TinputType *tinput, TweightsType *tweights, BiasType *bias, int _ic4);
  void gemm_f061(ToutputType *toutput, TinputType *tinput, TweightsType *tweights, BiasType *bias, int _t2, int Tz);
  void gemm_b061(OutputType *output, TinputType *tinput, TweightsType *tweights, BiasType *bias, int _ic4);
  void gemm_e060(OutputType *output, InputType *input, TweightsType *tweights, BiasType *bias, int _ic4);
  void gemm_c060(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias, int _ic4, int _oc4, int _t2);
  void gemm_d060(OutputType *output, InputType *input, WeightsType *weights, BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);

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

  TweightsType *tweights_;
  TinputType *tinput_;
  ToutputType *toutput_;
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
  void *scratch_;
};

template class elx_conv_direct_1x1_t<conv::FP32, conv_impl::FP32, 16, ISA_SKX_AVX512>;
template class elx_conv_direct_1x1_t<conv::FP32, conv_impl::FP32, 8, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
