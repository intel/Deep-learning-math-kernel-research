#ifndef __ELX_CONV_DIRECT_LP_HPP__
#define __ELX_CONV_DIRECT_LP_HPP__

#include <vector>
#include <tuple>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_u8s8_gemm_otj_binder.hxx"
#include "kernel/elk_u8s8_conv_otj_binder.hxx"

namespace euler {

#define Template_elx_conv_direct_lp_t                                          \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_lp_t                                          \
  elx_conv_direct_lp_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_lp_t class elx_conv_direct_lp_t : public elx_conv_t {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TweightsType = float;
  using ToutputType = typename TarrayTypes::OutputType;
  using TscaleType = typename TarrayTypes::ScaleType;

  public:
  elx_conv_direct_lp_t(eld_conv_t &dc);
  virtual ~elx_conv_direct_lp_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

  private:
  void __execute_a160(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_d160(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void __trans_weights_acc(TscaleType *weights_scale, TscaleType * weights_factor,
      int8_t *weights_s8, BiasType *bias);
  void trans_weights(TscaleType *weights_scale, TscaleType * weights_factor,
      int8_t *weights_s8, WeightsType *weights, BiasType *bias);

  void conv_a160(OutputType *output, ToutputType *toutput, InputType *input,
      int8_t *tweights, BiasType *bias, TscaleType *src_scale,
      TscaleType *weights_scale, TscaleType *weights_factor,
      int _ic4, int _oc4, int _ht, int _wt);
  void gemm_d160(OutputType *output, ToutputType *toutput, InputType *input,
      int8_t *tweights, BiasType *bias, TscaleType *src_scale,
      TscaleType *weights_scale, TscaleType *weights_factor,
      int _ic4, int _oc4, int _ht, int _wt);

  int prepare_execute_opt();
  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  void bind_execute_functions();
  void prepare_weights_acc();
  void prepare_quant_calibration(eld_conv_t &);

  // TODO: optimize it
  u8s8_gemm_kernel_binder::kgemm<TarrayTypes, float> *ker_gemm_[128][8];
  u8s8_conv_kernel_binder::kconv<TarrayTypes, OutputType> *ker_conv_;
  u8s8_conv_kernel_binder::kconv<TarrayTypes, OutputType> *ker_conv_Tr_;

  void (elx_conv_direct_lp_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  bool is_first_run_;
  bool inference_acc_;
  bool compact_ir_weights_;

  // asymmetric quantization support
  std::vector<std::tuple<int, int>> wacc_h_ranges_, wacc_w_ranges_;
  int wacc_h_, wacc_w_; // size of wacc
  int _wacc_hf_, _wacc_wf_, _wacc_hfr_, _wacc_wfr_; // index/reverse-index of acc-full
  int _wacc_ohfs_, _wacc_ohfe_, _wacc_owfs_, _wacc_owfe_; // _oh range of acc-full
  int wacc_wT_, wacc_wt_; // wT: T: a160
                          //     1: d160
                          // wt: number of T
                          //     3: left/middle/right, for input-z != 0
                          //     1: input-z = 0
  size_t tweights_s8_size_;
  size_t toutput_size_;
  size_t input_scale_size_;
  size_t weights_scale_size_;
  size_t weights_factor_size_;
  
  ToutputType *toutput_;
  TscaleType *input_scale_;
  TscaleType *weights_scale_;
  TscaleType *weights_factor_;
  int8_t *tweights_s8_;
  unsigned int xopt_;
  int attr_;
  int mthr_;
};

// int8-u8f32u8f32
template class elx_conv_direct_lp_t<conv::U8F32U8F32, conv_impl::INT8_F32, 16, ISA_SKX_AVX512>;
// int8-u8f32s8f32
template class elx_conv_direct_lp_t<conv::U8F32S8F32, conv_impl::INT8_F32, 16, ISA_SKX_AVX512>;
// int8-u8f32f32f32
template class elx_conv_direct_lp_t<conv::U8F32F32F32, conv_impl::INT8_F32, 16, ISA_SKX_AVX512>;

} // namespace euler
#endif // __ELX_CONV_DIRECT_LP_HPP__
