#ifndef __ELX_CONV_DIRECT_HPP__
#define __ELX_CONV_DIRECT_HPP__

#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"
#include "kernel/elk_gemm_otj_binder.hxx"

namespace euler {

#define Template_elx_conv_direct_t                                             \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_t                                             \
  elx_conv_direct_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_t class elx_conv_direct_t
    : public elx_conv_t<UserTypes> {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;

  public:
  elx_conv_direct_t(eld_conv_t<UserTypes> &dc);
  virtual ~elx_conv_direct_t();

  virtual void execute(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  private:
  void __execute_a060(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_d060(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void trans_weights_blocked_to_compact(
      TweightsType *tweights, WeightsType *weights);

  void conv_a060(OutputType *output, InputType *input, TweightsType *weights,
      BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);
  void gemm_d060(OutputType *toutput, InputType *tinput, TweightsType *tweights,
      BiasType *bias, int _ic4, int _oc4, int _ht, int _wt);

  void set_trans_buffers();
  int prepare_execute_opt();
  void bind_execute_functions();

  // TODO: optimize it
  gemm_kernel_binder::kgemm<TarrayTypes> *ker_gemm_[64][8];
  gemm_kernel_binder::kgemm<TarrayTypes> *ker_gemmr_[64][8];
  gemm_kernel_binder::kconv<TarrayTypes> *ker_conv_;
  gemm_kernel_binder::kconv<TarrayTypes> *ker_conv_Tr_;

  void (elx_conv_direct_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

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

// fp32-f32f32f32
template class elx_conv_direct_t<conv::FP32, conv_impl::FP32, 16, ISA_SKX_AVX512>;

// fp16o-f32f32f16
template class elx_conv_direct_t<conv::FP16O, conv_impl::FP32_F16o, 16, ISA_SKX_AVX512>;

} // namespace euler
#endif // __ELX_CONV_DIRECT_HPP__
