#pragma once

#include "kernel/elk_u8s8_gemm_binder.hxx"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
class elx_int8_conv_wino_gemm_t {
public:
  using TinputType   = typename GarrayTypes::InputType;
  using TweightsType = typename GarrayTypes::WeightsType;
  using ToutputType  = typename GarrayTypes::OutputType;

  elx_int8_conv_wino_gemm_t() {};
  virtual ~elx_int8_conv_wino_gemm_t() {};
  void setup(elx_param_t *ep);

  // INT8 GEMM
  void execute(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      float *src_scale, float *weights_scale, float *weights_shift,
      int _t2, int Tz, int _I4 = 0);

  void execute_na(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      float *src_scale, float *weights_scale, float *weights_shift,
      int _t2, int Tz, int _I4 = 0);

  void execute_na(ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
      float *src_scale, float *src_shift, float *weights_scale,
      float *weights_shift, int _I4 = 0);

private:
  void bind_kernel_functions();

  using i8_ker_type = typename u8s8_gemm_kernel_binder::kgemm<GarrayTypes, float>;

  i8_ker_type *ker_u8s8_gemm_;
  i8_ker_type *ker_u8s8_gemm0_;

  int attr_;
  int mthr_;
  elx_param_t *ep = nullptr;
};

}  // namespace euler
