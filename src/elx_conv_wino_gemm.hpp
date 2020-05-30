#pragma once

#include "kernel/elk_gemm_binder.hxx"

namespace euler {

template <typename GarrayTypes, const int A, const int V, const int I>
class elx_conv_wino_gemm_t {
public:
  using TinputType   = typename GarrayTypes::InputType;
  using TweightsType = typename GarrayTypes::WeightsType;
  using ToutputType  = typename GarrayTypes::OutputType;

  elx_conv_wino_gemm_t() {};
  virtual ~elx_conv_wino_gemm_t() {};
  void setup(elx_param_t *ep);

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
  elx_param_t *ep = nullptr;
};

}  // namespace euler
