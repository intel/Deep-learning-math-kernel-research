#pragma once

#include "elx_conv_direct_vmg.hpp"

namespace euler {

Template_elx_conv_direct_vmg_t void
Instance_elx_conv_direct_vmg_t::bind_execute_functions() {
#define BIND_CONV_KERNEL(S, F, K, G)                         \
  if (K == 3) {                                              \
    if (G == 1) {                                            \
      vmg_conv_kernel_binder::bind<S, F, 3, 1>(O, T, func);  \
    } else if (G == 2) {                                     \
      vmg_conv_kernel_binder::bind<S, F, 3, 2>(O, T, func);  \
    } else if (G == 4) {                                     \
      vmg_conv_kernel_binder::bind<S, F, 3, 4>(O, T, func);  \
    } else if (G == 8) {                                     \
      vmg_conv_kernel_binder::bind<S, F, 3, 8>(O, T, func);  \
    } else if (G == 16) {                                    \
      vmg_conv_kernel_binder::bind<S, F, 3, 16>(O, T, func); \
    }                                                        \
  } else {                                                   \
    el_error("Unimplemented: VMG conv for K != 3");          \
  }

  auto bind_conv_kernel = [&](int O, int T,
                              vmg_conv_kernel_binder::kconv<TarrayTypes> **func,
                              int K, int G) {
    switch (xopt_) {
    case (0xc060):
      if (ep.input_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_FCF, K, G);
        } else {
          el_error("Stride > 1 not yet bounded");
        }
      } else {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_DCD, K, G);
        } else {
          el_error("Stride > 1 not yet bounded");
        }
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  bind_conv_kernel(ep.O, ep.T, &ker_conv_, ep.kw, ep.G);
  bind_conv_kernel(ep.O, ep.Tr, &ker_conv_Tr_, ep.kw, ep.G);
#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    execute_opt_ = &Instance_elx_conv_direct_vmg_t::__execute_##n;             \
    break

  switch (xopt_) {
    EXECUTE_CASE(c060);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
