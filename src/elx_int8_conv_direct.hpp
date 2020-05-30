#ifndef __ELX_CONV_DIRECT_LP_HPP__
#define __ELX_CONV_DIRECT_LP_HPP__

#include <vector>
#include <tuple>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_u8s8_gemm_binder.hxx"
#include "kernel/elk_u8s8_conv_binder.hxx"

namespace euler {

#define Template_elx_int8_conv_direct_t                                          \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_int8_conv_direct_t                                          \
  elx_int8_conv_direct_t<UserTypes, TarrayTypes, V, I>

Template_elx_int8_conv_direct_t class elx_int8_conv_direct_t : public elx_conv_t {
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TweightsType = float;
  using ToutputType = typename TarrayTypes::OutputType;

  public:
  elx_int8_conv_direct_t(eld_conv_t &dc);
  virtual ~elx_int8_conv_direct_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

  private:
  void __execute_c160(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a160(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void __trans_weights_acc(float *weights_scale, float * weights_shift,
      int8_t *weights_s8, BiasType *bias);
  void trans_weights(float *weights_scale, float * weights_shift,
      int8_t *weights_s8, WeightsType *weights, BiasType *bias);

  void conv_c160(OutputType *output, ToutputType *toutput, InputType *input,
      int8_t *tweights, BiasType *bias, float *src_scale,
      float *weights_scale, float *weights_shift,
      int _I4, int _O4, int _ht, int _wt);
  void gemm_a160(OutputType *output, ToutputType *toutput, InputType *input,
      int8_t *tweights, BiasType *bias, float *src_scale,
      float *weights_scale, float *weights_shift,
      int _I4, int _O4, int _ht, int _wt);

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

  void (elx_int8_conv_direct_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  bool is_first_run_;
  bool inference_acc_;
  bool compact_ir_weights_;

  // asymmetric quantization support
  std::vector<std::tuple<int, int>> wacc_h_ranges_, wacc_w_ranges_;
  int wacc_h_, wacc_w_; // size of wacc
  int _wacc_hf_, _wacc_wf_, _wacc_hfr_, _wacc_wfr_; // index/reverse-index of acc-full
  int _wacc_ohfs_, _wacc_ohfe_, _wacc_owfs_, _wacc_owfe_; // _oh range of acc-full
  int wacc_wT_, wacc_wt_; // wT: T: c160
                          //     1: a160
                          // wt: number of T
                          //     3: left/middle/right, for input-z != 0
                          //     1: input-z = 0
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
#endif // __ELX_CONV_DIRECT_LP_HPP__
