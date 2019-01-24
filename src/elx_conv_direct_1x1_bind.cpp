#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elx_conv_direct_1x1.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

Template_elx_conv_direct_1x1_t void
Instance_elx_conv_direct_1x1_t::bind_execute_functions()
{
#define BIND_KERNEL(S, F)                                                    \
  gemm_kernel_binder::bind<TarrayTypes, V, 1, I, S, F>(O, T, func);

  auto bind_kernel = [&](int O, int T,
      gemm_kernel_binder::kgemm<TarrayTypes> **func) {
    switch (xopt_) {
    case (0xa061):
      if (this->input_fmt == nhwc) {
        if (this->ws == 1)
          BIND_KERNEL(1, GKF_FCF)
        else if (this->ws == 2)
          BIND_KERNEL(2, GKF_FCF)
      } else {
        BIND_KERNEL(1, GKF_CCC)
      }
      break;
    case (0xf061):
      if (this->input_fmt == nhwc)
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

  bind_kernel(this->O, this->T, &ker_gemm_I_O_T_);
  bind_kernel(this->O, this->Tr, &ker_gemm_I_O_Tr_);

#define EXECUTE_CASE(n)                                                     \
  case 0x##n:                                                               \
    printf("execute_opt=" #n "\n");                                         \
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
