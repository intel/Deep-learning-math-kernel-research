#include "elx_conv_direct_1x1_lp.hpp"

namespace euler {

Template_elx_conv_direct_1x1_lp_t void
Instance_elx_conv_direct_1x1_lp_t::bind_execute_functions()
{
#define BIND_KERNEL(S, F)                                               \
  u8s8_gemm_kernel_binder::bind<S, F>(O, T, func);

  auto bind_kernel = [&](int O, int T,                                         \
      u8s8_gemm_kernel_binder::kgemm<TarrayTypes, OutputType> **func) {        \
    switch (xopt_) {                                                           \
    case (0xc160):                                                             \
      BIND_KERNEL(1, GKF_DCD)                                                  \
      break;                                                                   \
    case (0xb161):                                                             \
      BIND_KERNEL(2, GKF_DCD)                                                  \
      break;                                                                   \
    default:                                                                   \
      el_error("Unknown xopt");                                                \
      break;                                                                   \
    }                                                                          \
  };

  bind_kernel(this->O, this->T, &ker_u8s8_gemm_I_O_T_);
  bind_kernel(this->O, this->Tr, &ker_u8s8_gemm_I_O_Tr_);

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    printf("execute_opt=" #n "\n");                                            \
    execute_opt_ = &Instance_elx_conv_direct_1x1_lp_t::__execute_##n;          \
    break

  switch (xopt_) {
    EXECUTE_CASE(c160);
    EXECUTE_CASE(b161);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
