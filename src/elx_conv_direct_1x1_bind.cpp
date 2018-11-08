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

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::bind_execute_functions()
{
#define BIND_KERNEL_2(S, F)                                                    \
  if (has_Ir) {                                                                \
    gemm_kernel_binder::bind<float, float, V, 1, I, S, F, true>(O, T, func);   \
  } else {                                                                     \
    gemm_kernel_binder::bind<float, float, V, 1, I, S, F, false>(O, T, func);  \
  }

#define BIND_KERNEL_1(S, F)                                                    \
  gemm_kernel_binder::bind<float, float, V, 1, I, S, F, false>(O, T, func);

  auto bind_kernel = [&](int O, int T,
      gemm_kernel_binder::ker<float, float> **func, bool has_Ir) {
    switch (xopt_) {
    case (0xa061):
      BIND_KERNEL_2(1, GKF_CCC)
      break;
    case (0xf061):
      BIND_KERNEL_2(1, GKF_CCC)
      break;
    case (0xb061):
      BIND_KERNEL_1(1, GKF_CCD)
      break;
    case (0xe060):
      BIND_KERNEL_1(2, GKF_DCD)
      break;
    case (0xc060):
      BIND_KERNEL_2(1, GKF_DDD)
      break;
    case (0xd060):
      BIND_KERNEL_1(2, GKF_DDD)
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  bind_kernel(this->O, this->T, &ker_gemm_I_O_T_, false);
  bind_kernel(this->O, this->Tr, &ker_gemm_I_O_Tr_, false);
  if (xopt_ == 0xc060 || xopt_ == 0xd060) {
    bind_kernel(this->O2r, this->T, &ker_gemm_I_OrT_, false);
    bind_kernel(this->O2r, this->Tr, &ker_gemm_I_OrTr_, false);
  }
  // Ir != V
  if (xopt_ == 0xa061 || xopt_ == 0xf061 || xopt_ == 0xc060) {
    bind_kernel(this->O, this->T, &ker_gemm_IrO_T_, true);
    bind_kernel(this->O, this->Tr, &ker_gemm_IrO_Tr_, true);
  }

#define EXECUTE_CASE(n)                                                     \
  case 0x##n:                                                               \
    printf("execute_opt=" #n "\n");                                         \
    execute_opt_ = &elx_conv_direct_1x1_t<Type, V, I>::__execute_##n;       \
    break

  switch (xopt_) {
    EXECUTE_CASE(a061);
    EXECUTE_CASE(f061);
    EXECUTE_CASE(b061);
    EXECUTE_CASE(e060);
    EXECUTE_CASE(c060);
    EXECUTE_CASE(d060);
  default:
    el_error("Unimplemented xopt");
    break;
  }
}

} // namespace euler
