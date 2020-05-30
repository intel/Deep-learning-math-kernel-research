#ifndef __ELX_CONV_DIRECT_1X1_LP_HPP__
#define __ELX_CONV_DIRECT_1X1_LP_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "kernel/elk_u8s8_gemm_binder.hxx"

namespace euler {

#define Template_elx_int8_conv_direct_1x1_t                                      \
  template <typename UserTypes, typename TarrayTypes, const int V, const int I>

#define Instance_elx_int8_conv_direct_1x1_t                                      \
    elx_int8_conv_direct_1x1_t<UserTypes, TarrayTypes, V, I>

Template_elx_int8_conv_direct_1x1_t
class elx_int8_conv_direct_1x1_t : public elx_conv_t {
public:
  // Configurable parameters
  using InputType = typename UserTypes::InputType;
  using WeightsType = typename UserTypes::WeightsType;
  using OutputType = typename UserTypes::OutputType;
  using BiasType = typename UserTypes::BiasType;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;

  public:
  elx_int8_conv_direct_1x1_t(eld_conv_t &dc);
  virtual ~elx_int8_conv_direct_1x1_t();

  virtual void execute(void *, void *, void *, void *);

  private:
  void __execute_a160(OutputType *, InputType *, WeightsType *, BiasType *);
  void __execute_a160_s1(OutputType *, InputType *, WeightsType *, BiasType *);
  void __execute_a160_s2(OutputType *, InputType *, WeightsType *, BiasType *);

  inline void trans_weights_s8_blocked_oc(float *, int8_t *, WeightsType *, BiasType *);

  void gemm_a160_s1(ToutputType *, OutputType *, uint8_t *, int8_t *,
      float *, float *, BiasType *, int, int, int);
  void gemm_a160_s2(ToutputType *, OutputType *, uint8_t *, int8_t *,
      float *, float *, BiasType *, int);

  void prepare_quant_calibration(eld_conv_t &);
  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  int prepare_execute_opt();
  void bind_execute_functions();

  u8s8_gemm_kernel_binder::kgemm<TarrayTypes, OutputType> *ker_u8s8_gemm_I_O_T_;
  u8s8_gemm_kernel_binder::kgemm<TarrayTypes, OutputType> *ker_u8s8_gemm_I_O_Tr_;

  void (elx_int8_conv_direct_1x1_t::*execute_opt_)(
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
  float *input_scale_;
  int8_t *tweights_s8_;
  float *weights_scale_;

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

}  // namespace euler
#endif  // __ELX_CONV_DIRECT_1X1_HPP__
