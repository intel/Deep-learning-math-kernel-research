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

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::bind_execute_functions()
{
  ker_trans_input_
    = Instance_convolution_winograd_kernel::template trans_input<no>;
  ker_trans_input0_
    = Instance_convolution_winograd_kernel::template trans_input<is_border>;
  ker_trans_inputa_
    = Instance_convolution_winograd_kernel::template trans_inputa<no>;
  ker_trans_inputa0_
    = Instance_convolution_winograd_kernel::template trans_inputa<is_border>;
  ker_trans_weights_
    = Instance_convolution_winograd_kernel::trans_weights;

  // TODO: ker_trans_output_nobias_norelu_nosum (no fusion)
  // Fusion operation is done in related ker_trans_output_
  ker_trans_output_nobias_ = Instance_convolution_winograd_kernel::
      template trans_output<no, no, no, no>;
  ker_trans_output0_nobias_ = Instance_convolution_winograd_kernel::
      template trans_output<is_border, no, no, no>;

  using kernel_set = Instance_convolution_winograd_kernel;
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

  ker_trans_outputa_th_ = Instance_convolution_winograd_kernel::
      template trans_outputa_th<no, no, no, no>;

  auto bind_gemm_kernel =
      [&](int O, int T, bool has_Ir,
      gemm_kernel_binder::ker<Instance_elx_conv_t, TarrayType, TarrayType> **func1,
      gemm_kernel_binder::ker<Instance_elx_conv_t, uint8_t, int8_t> **func2) {
    if (this->Ir != V * this->Vx && has_Ir) {
      gemm_kernel_binder::bind<Instance_elx_conv_t, TarrayType, TarrayType,
          V, 1, I, 1, GKF_CCC, true>(O, T, func1);
      gemm_kernel_binder::bind<Instance_elx_conv_t, uint8_t, int8_t,
          V, 4, I, 1, GKF_CCC, true>(O, T, func2);
    } else {
      gemm_kernel_binder::bind<Instance_elx_conv_t, TarrayType, TarrayType,
          V, 1, I, 1, GKF_CCC, false>(O, T, func1);
      gemm_kernel_binder::bind<Instance_elx_conv_t, uint8_t, int8_t,
          V, 4, I, 1, GKF_CCC, false>(O, T, func2);
    }
  };

  bind_gemm_kernel(this->O, this->T, false, &ker_gemm_, &ker_i8_gemm_);
  bind_gemm_kernel(this->O, this->Tr, false, &ker_gemm0_, &ker_i8_gemm0_);
  bind_gemm_kernel(this->O, this->T, true, &ker_gemm_tail_, &ker_i8_gemm_tail_);
  bind_gemm_kernel(this->O, this->Tr, true, &ker_gemm0_tail_, &ker_i8_gemm0_tail_);

#define EXECUTE_CASE(n)                                                      \
  case 0x##n:                                                                \
    printf("execute_opt=" #n "\n");                                          \
    execute_opt_ = &Instance_elx_conv_wino_t::__execute_##n;                 \
    break

  switch (xopt_) {
  EXECUTE_CASE(a000);
  EXECUTE_CASE(a033);
  EXECUTE_CASE(a061);
  EXECUTE_CASE(a071);
  EXECUTE_CASE(a073);
  EXECUTE_CASE(a079);
  EXECUTE_CASE(a07b);
  EXECUTE_CASE(a0e0);
  EXECUTE_CASE(a0e1);
  EXECUTE_CASE(a161);
  default:
    el_error("Unimplemented");
    break;
  }
}

} // namespace euler
