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

  ker_trans_weights_ = Instance_convolution_winograd_kernel::trans_weights;

  if (input_is_bfmt_ || input_as_bfmt_) {
    ker_trans_input_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_BLOCKED, no>;
    ker_trans_input0_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_BLOCKED, is_border>;
  } else if (this->input_fmt == nhwc) {
    ker_trans_input_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_NHWC, no>;
    ker_trans_input0_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_NHWC, is_border>;
  } else {  // nchw
    ker_trans_input_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_COMPACT, no>;
    ker_trans_input0_ = Instance_convolution_winograd_kernel
      ::template trans_input<TKF_COMPACT, is_border>;
  }

#undef E
#define E(format, border, bias, relu, sum)                                    \
  Instance_convolution_winograd_kernel::template trans_output<format, border, \
                                                              bias, relu, sum>
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } C_pointer_table[2][2][2] = {
      {{{E(TKF_COMPACT, 0, 0, 0, 0), E(TKF_COMPACT, 1, 0, 0, 0)},
        {E(TKF_COMPACT, 0, 0, 0, 1), E(TKF_COMPACT, 1, 0, 0, 1)}},
       {{E(TKF_COMPACT, 0, 0, 1, 0), E(TKF_COMPACT, 1, 0, 1, 0)},
        {E(TKF_COMPACT, 0, 0, 1, 1), E(TKF_COMPACT, 1, 0, 1, 1)}}},
      {{{E(TKF_COMPACT, 0, 1, 0, 0), E(TKF_COMPACT, 1, 1, 0, 0)},
        {E(TKF_COMPACT, 0, 1, 0, 1), E(TKF_COMPACT, 1, 1, 0, 1)}},
       {{E(TKF_COMPACT, 0, 1, 1, 0), E(TKF_COMPACT, 1, 1, 1, 0)},
        {E(TKF_COMPACT, 0, 1, 1, 1), E(TKF_COMPACT, 1, 1, 1, 1)}}}};
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } D_pointer_table[2][2][2] = {
      {{{E(TKF_BLOCKED, 0, 0, 0, 0), E(TKF_BLOCKED, 1, 0, 0, 0)},
        {E(TKF_BLOCKED, 0, 0, 0, 1), E(TKF_BLOCKED, 1, 0, 0, 1)}},
       {{E(TKF_BLOCKED, 0, 0, 1, 0), E(TKF_BLOCKED, 1, 0, 1, 0)},
        {E(TKF_BLOCKED, 0, 0, 1, 1), E(TKF_BLOCKED, 1, 0, 1, 1)}}},
      {{{E(TKF_BLOCKED, 0, 1, 0, 0), E(TKF_BLOCKED, 1, 1, 0, 0)},
        {E(TKF_BLOCKED, 0, 1, 0, 1), E(TKF_BLOCKED, 1, 1, 0, 1)}},
       {{E(TKF_BLOCKED, 0, 1, 1, 0), E(TKF_BLOCKED, 1, 1, 1, 0)},
        {E(TKF_BLOCKED, 0, 1, 1, 1), E(TKF_BLOCKED, 1, 1, 1, 1)}}}};
  static const struct {
    decltype(ker_trans_output_) f1_;
    decltype(ker_trans_output0_) f2_;
  } F_pointer_table[2][2][2] = {
      {{{E(TKF_NHWC, 0, 0, 0, 0), E(TKF_NHWC, 1, 0, 0, 0)},
        {E(TKF_NHWC, 0, 0, 0, 1), E(TKF_NHWC, 1, 0, 0, 1)}},
       {{E(TKF_NHWC, 0, 0, 1, 0), E(TKF_NHWC, 1, 0, 1, 0)},
        {E(TKF_NHWC, 0, 0, 1, 1), E(TKF_NHWC, 1, 0, 1, 1)}}},
      {{{E(TKF_NHWC, 0, 1, 0, 0), E(TKF_NHWC, 1, 1, 0, 0)},
        {E(TKF_NHWC, 0, 1, 0, 1), E(TKF_NHWC, 1, 1, 0, 1)}},
       {{E(TKF_NHWC, 0, 1, 1, 0), E(TKF_NHWC, 1, 1, 1, 0)},
        {E(TKF_NHWC, 0, 1, 1, 1), E(TKF_NHWC, 1, 1, 1, 1)}}}};

  if (output_is_bfmt_ || output_as_bfmt_) {
    ker_trans_output_ =
      D_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f1_;
    ker_trans_output0_ =
      D_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f2_;
    ker_trans_output_acc_ = D_pointer_table[this->with_bias][this->with_relu][1].f1_;
    ker_trans_output0_acc_ = D_pointer_table[this->with_bias][this->with_relu][1].f2_;
  } else if (this->output_fmt == nhwc) {
    ker_trans_output_ =
      F_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f1_;
    ker_trans_output0_ =
      F_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f2_;
    ker_trans_output_acc_ = F_pointer_table[this->with_bias][this->with_relu][1].f1_;
    ker_trans_output0_acc_ = F_pointer_table[this->with_bias][this->with_relu][1].f2_;
  } else {  // nchw
    ker_trans_output_ =
      C_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f1_;
    ker_trans_output0_ =
      C_pointer_table[this->with_bias][this->with_relu][this->with_ip_sum].f2_;
    ker_trans_output_acc_ = C_pointer_table[this->with_bias][this->with_relu][1].f1_;
    ker_trans_output0_acc_ = C_pointer_table[this->with_bias][this->with_relu][1].f2_;
  }

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
