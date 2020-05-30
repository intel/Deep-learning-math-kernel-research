#pragma once

#include "elx_int8_conv_direct_depthwise.hpp"

namespace euler {

Template_elx_int8_conv_direct_depthwise_t void
Instance_elx_int8_conv_direct_depthwise_t::bind_execute_functions()
{
#define BIND_CONV_KERNEL(S, F, K)                                              \
  if (K == 3) {                                                                \
    u8s8_depthwise_conv_kernel_binder::bind<S, F, 3>(O, T, func);              \
  }

  auto bind_conv_kernel = [&](int O, int T,
      u8s8_depthwise_conv_kernel_binder::kconv<TarrayTypes, OutputType> **func,
      int K) {
    switch (xopt_) {
    case (0xc160):
      if (ep.input_fmt == nChw16c && ep.output_fmt == nChw16c) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_DCD, K);
        } else if (ep.ws == 2) {
          BIND_CONV_KERNEL(2, GKF_DCD, K);
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else {
        el_error("direct_depthwise: int8: kernel fmt not supported");
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  if (xopt_ == 0xc160) {
    bind_conv_kernel(ep.O, ep.T, &ker_conv_, ep.kw);
    bind_conv_kernel(ep.O, ep.Tr, &ker_conv_Tr_, ep.kw);
  }

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    execute_opt_ = &Instance_elx_int8_conv_direct_depthwise_t::__execute_##n;  \
    break

  switch (xopt_) {
    EXECUTE_CASE(c160);
  default:
    el_error("direct_depthwise: int8: Unimplemented xopt");
    break;
  }
}

} // namespace euler
