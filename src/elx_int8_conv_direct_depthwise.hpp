#ifndef __ELX_CONV_DIRECT_DEPTHWISE_LP_HPP__
#define __ELX_CONV_DIRECT_DEPTHWISE_LP_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_u8s8_depthwise_conv_binder.hxx"

namespace euler {

#define Template_elx_int8_conv_direct_depthwise_t                                \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_int8_conv_direct_depthwise_t                                \
  elx_int8_conv_direct_depthwise_t<UserTypes, TarrayTypes, V, I>

Template_elx_int8_conv_direct_depthwise_t class elx_int8_conv_direct_depthwise_t : public elx_conv_t {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;
  static constexpr int KW = 4;

  public:
  elx_int8_conv_direct_depthwise_t(eld_conv_t &dc);
  virtual ~elx_int8_conv_direct_depthwise_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

  private:
  void __execute_c160(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void trans_weights_3x3(float *weights_scale, float * weights_shift,
      int8_t *weights_s8, WeightsType *weights, BiasType *bias);

  void conv_c160(OutputType *output, ToutputType *toutput, InputType *input,
      int8_t *tweights, BiasType *bias, float *src_scale,
      float *weights_scale, float *weights_shift, int _ht, int _wt);

  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  int prepare_execute_opt();
  void bind_execute_functions();
  void prepare_quant_calibration(eld_conv_t &);

  // TODO: optimize it
  u8s8_depthwise_conv_kernel_binder::kconv<TarrayTypes, OutputType> *ker_conv_;
  u8s8_depthwise_conv_kernel_binder::kconv<TarrayTypes, OutputType> *ker_conv_Tr_;

  void (elx_int8_conv_direct_depthwise_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  bool is_first_run_;
  bool inference_acc_;

  size_t tweights_size_;
  TweightsType *tweights_;
  size_t tweights_s8_size_;
  size_t toutput_size_;
  size_t input_scale_size_;
  size_t weights_scale_size_;
  size_t weights_shift_size_;
  ToutputType *toutput_;
  float *input_scale_;
  float *weights_scale_;
  float *weights_shift_;
  int8_t *tweights_s8_;
  unsigned int xopt_;
  int attr_;
  int mthr_;
};

} // namespace euler
#endif // __ELX_CONV_DIRECT_DEPTHWISE_LP_HPP__
