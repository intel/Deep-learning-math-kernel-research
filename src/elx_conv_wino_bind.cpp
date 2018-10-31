#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "euler.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::bind_execute_functions()
{
  ker_trans_input_
    = convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_input<no>;
  ker_trans_input0_
    = convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_input<is_border>;
  ker_trans_inputa_
    = convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_inputa<no>;
  ker_trans_inputa0_
    = convolution_winograd_kernel<
        Type, I, V, A, K>::template trans_inputa<is_border>;
  ker_trans_weights_
    = convolution_winograd_kernel<Type, I, V, A, K>::trans_weights;

  // TODO: ker_trans_output_nobias_norelu_nosum (no fusion)
  // Fusion operation is done in related ker_trans_output_
  ker_trans_output_nobias_ = convolution_winograd_kernel<Type, I, V, A, K>::
      template trans_output<no, no, no, no>;
  ker_trans_output0_nobias_ = convolution_winograd_kernel<Type, I, V, A, K>::
      template trans_output<is_border, no, no, no>;

  using kernel_set = convolution_winograd_kernel<Type, I, V, A, K>;
  static const struct {
    decltype (ker_trans_output_) f1_;
    decltype (ker_trans_output0_) f2_;
    decltype (ker_trans_outputa_bh_) f3_;
    decltype (ker_trans_outputa0_bh_) f4_;
  } pointer_table[2][2][2] = {
    {{{kernel_set::template trans_output<0, 0, 0, 0>
    , kernel_set::template trans_output<1, 0, 0, 0>
    , kernel_set::template trans_outputa_bh<0, 0, 0, 0>
    , kernel_set::template trans_outputa_bh<1, 0, 0, 0>}

    ,{kernel_set::template trans_output<0, 0, 0, 1>
    , kernel_set::template trans_output<1, 0, 0, 1>
    , kernel_set::template trans_outputa_bh<0, 0, 0, 1>
    , kernel_set::template trans_outputa_bh<1, 0, 0, 1>}}

    ,{{kernel_set::template trans_output<0, 0, 1, 0>
    , kernel_set::template trans_output<1, 0, 1, 0>
    , kernel_set::template trans_outputa_bh<0, 0, 1, 0>
    , kernel_set::template trans_outputa_bh<1, 0, 1, 0>}

    ,{kernel_set::template trans_output<0, 0, 1, 1>
    , kernel_set::template trans_output<1, 0, 1, 1>
    , kernel_set::template trans_outputa_bh<0, 0, 1, 1>
    , kernel_set::template trans_outputa_bh<1, 0, 1, 1>}}}

    ,{{{kernel_set::template trans_output<0, 1, 0, 0>
    , kernel_set::template trans_output<1, 1, 0, 0>
    , kernel_set::template trans_outputa_bh<0, 1, 0, 0>
    , kernel_set::template trans_outputa_bh<1, 1, 0, 0>}

    ,{kernel_set::template trans_output<0, 1, 0, 1>
    , kernel_set::template trans_output<1, 1, 0, 1>
    , kernel_set::template trans_outputa_bh<0, 1, 0, 1>
    , kernel_set::template trans_outputa_bh<1, 1, 0, 1>}}

    ,{{kernel_set::template trans_output<0, 1, 1, 0>
    , kernel_set::template trans_output<1, 1, 1, 0>
    , kernel_set::template trans_outputa_bh<0, 1, 1, 0>
    , kernel_set::template trans_outputa_bh<1, 1, 1, 0>}

    ,{kernel_set::template trans_output<0, 1, 1, 1>
    , kernel_set::template trans_output<1, 1, 1, 1>
    , kernel_set::template trans_outputa_bh<0, 1, 1, 1>
    , kernel_set::template trans_outputa_bh<1, 1, 1, 1>}}}
  };

  auto slot = pointer_table[this->with_bias][this->with_relu][this->with_ip_sum];
  ker_trans_output_ = slot.f1_;
  ker_trans_output0_ = slot.f2_;
  ker_trans_outputa_bh_ = slot.f3_;
  ker_trans_outputa0_bh_ = slot.f4_;

  static const struct {
    decltype (ker_trans_output_) f1_;
    decltype (ker_trans_output0_) f2_;
  } pointer_table2[2][2] = {
    {{kernel_set::template trans_output<0, 0, 0, 1>
    , kernel_set::template trans_output<1, 0, 0, 1>}

    ,{kernel_set::template trans_output<0, 0, 1, 1>
    , kernel_set::template trans_output<1, 0, 1, 1>}}

    ,{{kernel_set::template trans_output<0, 1, 0, 1>
    , kernel_set::template trans_output<1, 1, 0, 1>}

    ,{kernel_set::template trans_output<0, 1, 1, 1>
    , kernel_set::template trans_output<1, 1, 1, 1>}}
  };

  auto slot2 = pointer_table2[this->with_bias][this->with_relu];
  ker_trans_output_acc_ = slot2.f1_;
  ker_trans_output0_acc_ = slot2.f2_;

  ker_trans_outputa_th_ = convolution_winograd_kernel<Type, I, V, A, K>::
      template trans_outputa_th<no, no, no, no>;

  auto bind_gemm_kernel =
    [&](int O, int T, bool has_Ir, gemm_kernel_binder::ker **func) {
    if (this->Ir != V && has_Ir) {
      gemm_kernel_binder::bind<Type, V, I, 1, GKF_CCC, true>(O, T, func);
    } else {
      gemm_kernel_binder::bind<Type, V, I, 1, GKF_CCC, false>(O, T, func);
    }
  };

  bind_gemm_kernel(this->O, this->T, false, &ker_gemm_);
  bind_gemm_kernel(this->O, this->Tr, false, &ker_gemm0_);
  bind_gemm_kernel(this->O, this->T, true, &ker_gemm_tail_);
  bind_gemm_kernel(this->O, this->Tr, true, &ker_gemm0_tail_);

#define EXECUTE_CASE(n)                                                      \
  case 0x##n:                                                                \
    printf("execute_opt=" #n "\n");                                          \
    execute_opt_ = &elx_conv_wino_t<Type, A, K, V, I>::__execute_##n;        \
    break

  switch (xopt_) {
  EXECUTE_CASE(a000);
  EXECUTE_CASE(a030);
  EXECUTE_CASE(a061);
  EXECUTE_CASE(a071);
  EXECUTE_CASE(a073);
  EXECUTE_CASE(a079);
  EXECUTE_CASE(a07b);
  EXECUTE_CASE(a0e0);
  EXECUTE_CASE(a0e1);
  default:
    el_error("Unimplemented");
    break;
  }
}

} // namespace euler
