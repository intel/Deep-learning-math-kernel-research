#pragma once

#include "kernel/elk_gemm_binder.hxx"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
class elx_conv_wino_gemm_t {
public:
  using TinputType   = typename GarrayTypes::InputType;
  using TweightsType = typename GarrayTypes::WeightsType;
  using ToutputType  = typename GarrayTypes::OutputType;
  using TscaleType   = typename GarrayTypes::ScaleType;

  elx_conv_wino_gemm_t() {};
  virtual ~elx_conv_wino_gemm_t() {};
  void setup(elx_conv_params_t *xc);

  // FP32 GEMM
  void execute(ToutputType *toutput, TinputType *tinput,
      TweightsType *tweights, int _t2, int Tz, int _I4 = 0);

  void execute_na(ToutputType *toutput, TinputType *tinput,
      TweightsType *tweights, int _t2, int Tz, int _I4);

  void execute(ToutputType *toutput, TinputType *tinput,
      TweightsType *tweights, int _I4 = 0);

  void execute_na(ToutputType *toutput, TinputType *tinput,
      TweightsType *tweights, int _I4 = 0);

private:
  void bind_kernel_functions();

  using ker_type = typename gemm_kernel_binder::kgemm<GarrayTypes>;
  ker_type *ker_gemm_;
  ker_type *ker_gemm0_;

  int attr_;
  int mthr_;
  elx_conv_params_t *xc = nullptr;
};

// f32f32f32f32
template class elx_conv_wino_gemm_t<conv_impl::FP32, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32, 7, 16, ISA_AVX512>;

// f16f16f16f32
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16iwo, 7, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// f32f16f16f16
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 4, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 5, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 6, 16, ISA_AVX512>;
template class elx_conv_wino_gemm_t<conv_impl::FP32_F16wob, 7, 16, ISA_AVX512>;
#endif
}  // namespace euler
