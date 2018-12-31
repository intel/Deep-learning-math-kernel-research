#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elx_conv_direct.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::bind_execute_functions()
{
#define BIND_KERNEL(S, F)                                                      \
  if (has_Ir) {                                                                \
    gemm_kernel_binder::bind<conv_impl::FP32, V, 1, I, S,                      \
        F, true>(O, T, func);                                                  \
  } else {                                                                     \
    gemm_kernel_binder::bind<conv_impl::FP32, V, 1, I, S,                      \
        F, false>(O, T, func);                                                 \
  }

  auto bind_gemm_kernel = [&](int O, int T,
      gemm_kernel_binder::kgemm<conv_impl::FP32> **func, bool has_Ir) {
    switch (xopt_) {
    case (0xd060):
      if (this->input_fmt == nchw) {
        BIND_KERNEL(1, GKF_ECD)
      } else {
        BIND_KERNEL(1, GKF_DCD)
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  auto bind_conv_kernel = [&](int O, int T,
      gemm_kernel_binder::kconv<conv_impl::FP32> **func, bool has_Ir) {
    switch (xopt_) {
    case (0xa060):
      if (this->input_fmt == nchw) {
        BIND_KERNEL(1, GKF_ECD);
      } else {
        BIND_KERNEL(1, GKF_DCD);
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  if (xopt_ == 0xa060) {
    bind_conv_kernel(this->O, this->T, &ker_conv_, false);
    bind_conv_kernel(this->O, this->Tr, &ker_conv_Tr_, false);
  } else if (xopt_ == 0xd060) {
    bind_gemm_kernel(this->O, this->T, &ker_gemm_I_O_T_, false);
    bind_gemm_kernel(this->O, this->Tr, &ker_gemm_I_O_Tr_, false);
    bind_gemm_kernel(this->O, this->T, &ker_gemm_IrO_T_, this->Ir != V);
    bind_gemm_kernel(this->O, this->Tr, &ker_gemm_IrO_Tr_, this->Ir != V);

    bind_gemm_kernel(this->O, this->T - 1, &ker_gemm_left_I_O_T_, false);
    bind_gemm_kernel(this->O, this->Tr - 1, &ker_gemm_left_I_O_Tr_, false);
    bind_gemm_kernel(this->O, this->T - 1, &ker_gemm_left_IrO_T_, this->Ir != V);
    bind_gemm_kernel(this->O, this->Tr - 1, &ker_gemm_left_IrO_Tr_, this->Ir != V);

    bind_gemm_kernel(this->O, this->T - 1, &ker_gemm_right_I_O_T_, false);
    bind_gemm_kernel(this->O, this->Tr - 1, &ker_gemm_right_I_O_Tr_, false);
    bind_gemm_kernel(this->O, this->T - 1, &ker_gemm_right_IrO_T_, this->Ir != V);
    bind_gemm_kernel(this->O, this->Tr - 1, &ker_gemm_right_IrO_Tr_, this->Ir != V);
  }

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    printf("execute_opt=" #n "\n");                                            \
    execute_opt_ = &Instance_elx_conv_direct_t::__execute_##n;                 \
    break

  switch (xopt_) {
    EXECUTE_CASE(a060);
    EXECUTE_CASE(d060);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
