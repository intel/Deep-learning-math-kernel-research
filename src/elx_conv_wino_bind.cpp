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

Template_elx_conv_wino_t void
Instance_elx_conv_wino_t::bind_execute_functions() {
  if (input_is_bfmt_ || input_as_bfmt_) {
    ker_trans_input_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_BLOCKED,
                                                                   no>;
    ker_trans_input0_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_BLOCKED,
                                                                   is_border>;
  } else if (this->input_fmt == nhwc) {
    ker_trans_input_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_NHWC,
                                                                   no>;
    ker_trans_input0_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_NHWC,
                                                                   is_border>;
  } else {  // nchw
    ker_trans_input_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_COMPACT,
                                                                   no>;
    ker_trans_input0_ =
        Instance_convolution_winograd_kernel::template trans_input<TKF_COMPACT,
                                                                   is_border>;
  }
  ker_trans_weights_ = Instance_convolution_winograd_kernel::trans_weights;

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
  } pointer_table[2][2][2] = {
    {{{kernel_set::template trans_output<0, 0, 0, 0>
    , kernel_set::template trans_output<1, 0, 0, 0>}

    ,{kernel_set::template trans_output<0, 0, 0, 1>
    , kernel_set::template trans_output<1, 0, 0, 1>}}

    ,{{kernel_set::template trans_output<0, 0, 1, 0>
    , kernel_set::template trans_output<1, 0, 1, 0>}

    ,{kernel_set::template trans_output<0, 0, 1, 1>
    , kernel_set::template trans_output<1, 0, 1, 1>}}}

    ,{{{kernel_set::template trans_output<0, 1, 0, 0>
    , kernel_set::template trans_output<1, 1, 0, 0>}

    ,{kernel_set::template trans_output<0, 1, 0, 1>
    , kernel_set::template trans_output<1, 1, 0, 1>}}

    ,{{kernel_set::template trans_output<0, 1, 1, 0>
    , kernel_set::template trans_output<1, 1, 1, 0>}

    ,{kernel_set::template trans_output<0, 1, 1, 1>
    , kernel_set::template trans_output<1, 1, 1, 1>}}}
  };

  auto slot = pointer_table[this->with_bias][this->with_relu][this->with_ip_sum];
  ker_trans_output_ = slot.f1_;
  ker_trans_output0_ = slot.f2_;

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

  auto bind_gemm_kernel =
      [&](int O, int T, ker_type **func1, i8_ker_type **func2) {
    gemm_kernel_binder::bind<TarrayTypes,
        V, 1, I, 1, GKF_CCC>(O, T, func1);

    if (this->fp_mode) {
      gemm_kernel_binder::bind<conv_impl::INT8_F16b,
          V, 4, I, 1, GKF_CCC>(O, T, func2);
    } if (this->f16c_opt) {
      gemm_kernel_binder::bind<conv_impl::INT8_F16o,
          V, 4, I, 1, GKF_CCC>(O, T, func2);
    } else {
      gemm_kernel_binder::bind<conv_impl::INT8_F32,
          V, 4, I, 1, GKF_CCC>(O, T, func2);
    }
  };

  bind_gemm_kernel(this->O, this->T, &ker_gemm_, &ker_i8_gemm_);
  bind_gemm_kernel(this->O, this->Tr, &ker_gemm0_, &ker_i8_gemm0_);

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
  EXECUTE_CASE(a133);
  EXECUTE_CASE(a161);
  EXECUTE_CASE(a173);
  default:
    el_error("Unimplemented");
    break;
  }
}

} // namespace euler
