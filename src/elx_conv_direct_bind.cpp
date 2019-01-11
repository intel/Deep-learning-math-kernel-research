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
#define BIND_GEMM_KERNEL(S, F)                                                 \
  gemm_kernel_binder::bind<TarrayTypes, V, 1, I, S, F>(O, T, func);

#define BIND_CONV_KERNEL(S, F, K)                                              \
  if (K == 3) {                                                                \
    gemm_kernel_binder::bind<TarrayTypes, V, 1, I, S, F, 3>(O, T, func);       \
  } else if (K == 5) {                                                         \
    gemm_kernel_binder::bind<TarrayTypes, V, 1, I, S, F, 5>(O, T, func);       \
  } else if (K == 7) {                                                         \
    gemm_kernel_binder::bind<TarrayTypes, V, 1, I, S, F, 7>(O, T, func);       \
  }

  auto bind_gemm_kernel = [&](int O, int T,
      gemm_kernel_binder::kgemm<TarrayTypes> **func) {
    switch (xopt_) {
    case (0xd060):
      if (this->ws == 1) {
        BIND_GEMM_KERNEL(1, GKF_DCD)
      } else if (this->ws == 2) {
        BIND_GEMM_KERNEL(2, GKF_DCD)
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  auto bind_conv_kernel = [&](int O, int T,
      gemm_kernel_binder::kconv<TarrayTypes> **func, int K) {
    switch (xopt_) {
    case (0xa060):
      if (this->input_fmt == nchw) {
        BIND_CONV_KERNEL(1, GKF_ECD, K);
      } else {
        BIND_CONV_KERNEL(1, GKF_DCD, K);
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  if (xopt_ == 0xa060) {
    bind_conv_kernel(this->O, this->T, &ker_conv_, this->kw);
    bind_conv_kernel(this->O, this->Tr, &ker_conv_Tr_, this->kw);
  } else if (xopt_ == 0xd060) {
    iter_each (_wt, this->wt) {
      int Tz = _wt == this->wt - 1 ? this->Tr : this->T;
      for (int _kw = 0; _kw < this->kw; ++_kw) {
        // _iws, _iwe
        // _iw = ws * _ow + _kw - lp
        auto ows0 = _wt * this->T;
        auto owe0 = _wt * this->T + Tz - 1;
        auto _iws = this->ws * ows0 + _kw - this->lp;
        while (_iws < 0)
          _iws += this->ws;
        auto _iwe = this->ws * owe0 + _kw - this->lp;
        while (_iwe > this->iw - 1)
          _iwe -= this->ws;
        auto _ows = (_iws + this->lp - _kw) / this->ws;
        auto _owe = (_iwe + this->lp - _kw) / this->ws;
        bind_gemm_kernel(this->O, _owe - _ows + 1, &ker_gemm_[_wt][_kw]);
      }
    }
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
