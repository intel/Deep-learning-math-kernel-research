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
    case (0xa061):
      if (ep.input_fmt == nhwc) {
        if (ep.ws == 1)
          BIND_KERNEL(1, GKF_FCF)
        else if (ep.ws == 2)
          BIND_KERNEL(2, GKF_FCF)
      } else {
        BIND_KERNEL(1, GKF_CCC)
      }
      break;
    case (0xf061):
      if (ep.input_fmt == nhwc)
        BIND_KERNEL(1, GKF_FCF)
      else
        BIND_KERNEL(1, GKF_CCC)
      break;
    case (0xb061):
      BIND_KERNEL(1, GKF_CCD)
      break;
    case (0xc060):
      BIND_KERNEL(1, GKF_DCD)
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  bind_kernel(ep.O, ep.T, &ker_gemm_I_O_T_);
  bind_kernel(ep.O, ep.Tr, &ker_gemm_I_O_Tr_);

#define EXECUTE_CASE(n)                                                     \
  case 0x##n:                                                               \
    execute_opt_ = &Instance_elx_conv_direct_1x1_t::__execute_##n;          \
    break

  switch (xopt_) {
    EXECUTE_CASE(a061);
    EXECUTE_CASE(f061);
    EXECUTE_CASE(b061);
    EXECUTE_CASE(c060);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
