#pragma once

#include "elx_int8_conv_direct_1x1.hpp"

namespace euler {

Template_elx_int8_conv_direct_1x1_t void
Instance_elx_int8_conv_direct_1x1_t::bind_execute_functions() {
#define BIND_KERNEL(S, F) u8s8_gemm_kernel_binder::bind<S, F>(O, T, func);

  auto bind_kernel = [&](int O, int T,
                         u8s8_gemm_kernel_binder::kgemm<TarrayTypes,
                         OutputType> **func) {
    if (ep.ws == 1) {
      if (ep.input_fmt == nChw16c && ep.output_fmt == nChw16c)
        BIND_KERNEL(1, GKF_DCD)
      else if (ep.input_fmt == nhwc && ep.output_fmt == nChw16c)
        BIND_KERNEL(1, GKF_FCD)
      else if (ep.input_fmt == nChw16c && ep.output_fmt == nhwc)
        BIND_KERNEL(1, GKF_DCF)
      else
        BIND_KERNEL(1, GKF_FCF)
    } else if (ep.ws == 2) {
      if (ep.input_fmt == nChw16c && ep.output_fmt == nChw16c) {
        BIND_KERNEL(2, GKF_DCD)
      } else if (ep.input_fmt == nhwc && ep.output_fmt == nChw16c) {
        BIND_KERNEL(2, GKF_FCD)
      } else if (ep.input_fmt == nChw16c && ep.output_fmt == nhwc) {
        BIND_KERNEL(2, GKF_DCF)
      } else { // nhwc -> nhwc
        BIND_KERNEL(2, GKF_FCF)
      }
    } else {
      el_error("ws > 2 not enabled");
    }
  };

  bind_kernel(ep.O, ep.T, &ker_u8s8_gemm_I_O_T_);
  bind_kernel(ep.O, ep.Tr, &ker_u8s8_gemm_I_O_Tr_);

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    execute_opt_ = &Instance_elx_int8_conv_direct_1x1_t::__execute_##n;        \
    break

  switch (xopt_) {
    EXECUTE_CASE(a160);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
