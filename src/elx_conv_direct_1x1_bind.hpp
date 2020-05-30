#pragma once

#include "elx_conv_direct_1x1.hpp"

namespace euler {

Template_elx_conv_direct_1x1_t void
Instance_elx_conv_direct_1x1_t::bind_execute_functions()
{
#define BIND_KERNEL(S, F)                                                    \
  gemm_kernel_binder::bind<S, F>(O, T, func);

  auto bind_kernel = [&](int O, int T,
      gemm_kernel_binder::kgemm<TarrayTypes> **func) {
    switch (xopt_) {
    case (a061p2):
      if (ep.input_fmt == nhwc) {
        if (ep.ws == 1)
          BIND_KERNEL(1, GKF_FCF)
        else if (ep.ws == 2)
          BIND_KERNEL(2, GKF_FCF)
      } else {
        BIND_KERNEL(1, GKF_CCC)
      }
      break;
    case (a061p1):
      if (ep.input_fmt == nhwc)
        BIND_KERNEL(1, GKF_FCF)
      else
        BIND_KERNEL(1, GKF_CCC)
      break;
    case (a061):
      BIND_KERNEL(1, GKF_CCD)
      break;
    case (a060):
      BIND_KERNEL(1, GKF_DCD)
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  bind_kernel(ep.O, ep.T, &ker_gemm_I_O_T_);
  bind_kernel(ep.O, ep.Tr, &ker_gemm_I_O_Tr_);

  switch (xopt_) {
    case a060:
      execute_opt_ = &Instance_elx_conv_direct_1x1_t::__execute_a060;
      break;
    case a061:
      execute_opt_ = &Instance_elx_conv_direct_1x1_t::__execute_a061;
      break;
    case a061p1:
      execute_opt_ = &Instance_elx_conv_direct_1x1_t::__execute_a061p1;
      break;
    case a061p2:
      execute_opt_ = &Instance_elx_conv_direct_1x1_t::__execute_a061p2;
      break;
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
