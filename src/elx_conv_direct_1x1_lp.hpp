#ifndef __ELX_CONV_DIRECT_1X1_LP_HPP__
#define __ELX_CONV_DIRECT_1X1_LP_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_u8s8_gemm_otj_binder.hxx"

namespace euler {

#define Template_elx_conv_direct_1x1_lp_t                                      \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_conv_direct_1x1_lp_t                                      \
    elx_conv_direct_1x1_lp_t<UserTypes, TarrayTypes, V, I>

Template_elx_conv_direct_1x1_lp_t
class elx_conv_direct_1x1_lp_t : public elx_conv_t {
public:
  // Configurable parameters
  using elx_conv_t::IC;
  using elx_conv_t::OC;
  using elx_conv_t::T;
  using elx_conv_t::I2;
  using elx_conv_t::O2;
  using elx_conv_t::oc4;
  using elx_conv_t::ic3;
  using elx_conv_t::oc3;
  using elx_conv_t::V1;
  using elx_conv_t::Vx;
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;
  using TscaleType = typename TarrayTypes::ScaleType;

  public:
  elx_conv_direct_1x1_lp_t(eld_conv_t &dc);
  virtual ~elx_conv_direct_1x1_lp_t();

  virtual void execute(void *, void *, void *, void *);

  private:
  void __execute_b161(OutputType *, InputType *, WeightsType *, BiasType *);
  void __execute_c160(OutputType *, InputType *, WeightsType *, BiasType *);

  inline void trans_weights_s8_blocked_oc(TscaleType *, int8_t *, WeightsType *, BiasType *);

  void requant_output(OutputType *, ToutputType *);


  void gemm_b161(ToutputType *, OutputType *, uint8_t *, int8_t *,
      TscaleType *, TscaleType *, BiasType *, int);
  void gemm_c160(ToutputType *, OutputType *, uint8_t *, int8_t *,
      TscaleType *, TscaleType *, BiasType *, int, int, int);

  void prepare_quant_calibration(eld_conv_t &);
  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  int prepare_execute_opt();
  void bind_execute_functions();

  u8s8_gemm_kernel_binder::kgemm<TarrayTypes, OutputType> *ker_u8s8_gemm_I_O_T_;
  u8s8_gemm_kernel_binder::kgemm<TarrayTypes, OutputType> *ker_u8s8_gemm_I_O_Tr_;

  void (elx_conv_direct_1x1_lp_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  bool no_pad_;
  bool is_first_run_;
  bool inference_acc_;
  bool toutput_opt_;

  bool stream_in_;
  bool stream_out_;

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
  InputType *binput_; // blocked input
  WeightsType *bweights_;
  OutputType *boutput_;
  TscaleType *input_scale_;
  int8_t *tweights_s8_;
  TscaleType *weights_scale_;

  unsigned int xopt_;
  int attr_;
  int mthr_;
  size_t tweights_size_;
  size_t tinput_size_;
  size_t toutput_size_;
  size_t binput_size_;
  size_t bweights_size_;
  size_t boutput_size_;
  size_t input_scale_size_;
  size_t tweights_s8_size_;
  size_t weights_scale_size_;
};

//u8f32u8f32-u8s8f32
template class elx_conv_direct_1x1_lp_t<conv::U8F32U8F32, conv_impl::INT8_F32, 16, ISA_SKX_AVX512>;
//u8f32s8f32-u8s8f32
template class elx_conv_direct_1x1_lp_t<conv::U8F32S8F32, conv_impl::INT8_F32, 16, ISA_SKX_AVX512>;

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
