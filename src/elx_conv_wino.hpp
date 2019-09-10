#ifndef __ELX_CONV_WINO_HPP__
#define __ELX_CONV_WINO_HPP__

#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_allocator.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino_gemm.hpp"
#include "elx_conv_wino_trans_input.hpp"
#include "elx_conv_wino_trans_output.hpp"
#include "elx_conv_wino_trans_weights.hpp"

/*
  Winograd data types: (input,weights,output)
  +--------------+------+-----+-------------+--------------+-----------+
  |Name          |XOPT  |F16C |UserTypes    |TarrayTypes   |GemmOpTypes|
  +--------------+------+-----+-------------+--------------+-----------+
  |bf16          |TBD   |false|bf16         |bf16          |bf16       |
  |fp16          |A061… |true |fp16         |fp32,fp16,fp16|fp32       |
  |fp32-f16c     |A061… |true |fp32         |fp32,fp16,fp16|fp32       |
  |fp32          |A061… |false|fp32         |fp32          |fp32       |
  +--------------+------+-----+-------------+--------------+-----------+

  * Non-INT8 mode, GarrayTypes equals to TarrayTypes.
*/

namespace euler {

#define Template_elx_conv_wino_t                                               \
  template <typename UserTypes, typename TarrayTypes,                          \
      const int A, const int K, const int V, const int I>

#define Instance_elx_conv_wino_t                                               \
  elx_conv_wino_t<UserTypes, TarrayTypes, A, K, V, I>

Template_elx_conv_wino_t
class elx_conv_wino_t : public elx_conv_t {
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

  using TrOpType = float;

  // t-buffer type
  using TinputType = typename TarrayTypes::InputType;
  using TweightsType = typename TarrayTypes::WeightsType;
  using ToutputType = typename TarrayTypes::OutputType;
  using TscaleType = typename TarrayTypes::ScaleType;

  constexpr static bool is_border = true;
  constexpr static bool has_bias = true;
  constexpr static bool has_relu = true;
  constexpr static bool has_sum = true;
  constexpr static bool no = false;

public:
  elx_conv_wino_t(eld_conv_t &dc);
  virtual ~elx_conv_wino_t();

  virtual void execute(void *output, void *input, void *weights, void *bias);

private:
  void __execute_a000(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a033(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a061(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a071(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a073(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a079(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);
  void __execute_a07b(OutputType *output, InputType *input,
      WeightsType *weights, BiasType *bias);

  void set_scratch_buffers(void *base);
  void set_workspace_buffers(void *base);
  int prepare_execute_opt();
  void bind_execute_functions();

  void (elx_conv_wino_t::*execute_opt_)(
      OutputType *, InputType *, WeightsType *, BiasType *);

  elx_conv_wino_trans_input_t<TinputType, InputType, I, A, K, V>
    trans_input;

  elx_conv_wino_trans_weights_t<TweightsType, WeightsType, I, A, K, V>
    trans_weights;

  elx_conv_wino_gemm_t<TarrayTypes, A, V, I> gemm;

  elx_conv_wino_trans_output_t<OutputType, BiasType, ToutputType, I, A, K, V>
    trans_output;

  unsigned int xopt_;
  bool is_first_run_;
  bool inference_acc_;
  bool is_bfmt_;
  bool input_is_bfmt_;
  bool weights_is_bfmt_;
  bool output_is_bfmt_;
  bool input_as_bfmt_;
  bool weights_as_bfmt_;
  bool output_as_bfmt_;
  int mthr_;
  size_t tweights_size_;
  size_t tinput_size_;
  size_t toutput_size_;
  size_t binput_size_;
  size_t bweights_size_;
  size_t boutput_size_;

  TweightsType *tweights_;
  TinputType *tinput_;
  ToutputType *toutput_;
  InputType *binput_; // blocked input
  WeightsType *bweights_;
  OutputType *boutput_;
};

// fp32-f32f32f32
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 4, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 5, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 6, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 7, 3, 16, ISA_SKX_AVX512>;

// fp32-f16f16f16
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 4, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 5, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 6, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 7, 3, 16, ISA_SKX_AVX512>;

#ifdef ENABLE_USER_FP16
// fp16-f32f16f16
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 4, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 5, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 6, 3, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 7, 3, 16, ISA_SKX_AVX512>;
#endif

}  // namespace euler
#endif  // __ELX_CONV_WINO_HPP__
