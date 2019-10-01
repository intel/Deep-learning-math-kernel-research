#pragma once

#include "kernel/elk_u8s8_gemm_binder.hxx"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
class elx_conv_wino_u8s8_gemm_t {
public:
  using TinputType   = typename GarrayTypes::InputType;
  using TweightsType = typename GarrayTypes::WeightsType;
  using ToutputType  = typename GarrayTypes::OutputType;
  using TscaleType   = typename GarrayTypes::ScaleType;

  elx_conv_wino_u8s8_gemm_t() {};
  virtual ~elx_conv_wino_u8s8_gemm_t() {};
  void setup(elx_conv_params_t *xc);

  // INT8 GEMM
  void execute(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      TscaleType *src_scale, TscaleType *weights_scale, TscaleType *weights_factor,
      int _t2, int Tz, int _I4 = 0);

  void execute_na(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      TscaleType *src_scale, TscaleType *weights_scale, TscaleType *weights_factor,
      int _t2, int Tz, int _I4 = 0);

  void execute_na(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      TscaleType *src_scale, TscaleType *src_factor, TscaleType *weights_scale,
      TscaleType *weights_factor, int _I4 = 0);

private:
  void bind_kernel_functions();

  using i8_ker_type = typename u8s8_gemm_kernel_binder::kgemm<GarrayTypes, float>;

  i8_ker_type *ker_u8s8_gemm_;
  i8_ker_type *ker_u8s8_gemm0_;

  int attr_;
  int mthr_;
  elx_conv_params_t *xc = nullptr;
};

// GarrayTypes IN8_FP32
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F32, 4, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F32, 5, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F32, 6, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F32, 7, 16, ISA_SKX_AVX512>;

// GarrayTypes IN8_F16o
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16o, 4, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16o, 5, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16o, 6, 16, ISA_SKX_AVX512>;

#ifdef ENABLE_USER_FP16
// GarrayTypes IN8_F16b
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16b, 4, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16b, 5, 16, ISA_SKX_AVX512>;
template class elx_conv_wino_u8s8_gemm_t<conv_impl::INT8_F16b, 6, 16, ISA_SKX_AVX512>;
#endif
}  // namespace euler
