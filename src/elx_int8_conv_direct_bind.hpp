#pragma once

#include "elx_int8_conv_direct.hpp"

namespace euler {

Template_elx_int8_conv_direct_t void
Instance_elx_int8_conv_direct_t::bind_execute_functions()
{
#define BIND_GEMM_KERNEL(S, F)                                                 \
  u8s8_gemm_kernel_binder::bind<S, F>(O, T, func);

#define BIND_CONV_KERNEL(S, F, K)                                              \
  if (K == 3) {                                                                \
    u8s8_conv_kernel_binder::bind<S, F, 3>(O, T, func);                        \
  } else if (K == 5) {                                                         \
    u8s8_conv_kernel_binder::bind<S, F, 5>(O, T, func);                        \
  } else if (K == 7) {                                                         \
    u8s8_conv_kernel_binder::bind<S, F, 7>(O, T, func);                        \
  }

  auto bind_gemm_kernel = [&](int O, int T,
      u8s8_gemm_kernel_binder::kgemm<TarrayTypes, float> **func) {
    switch (xopt_) {
    case (0xa160):
      if (ep.input_fmt == nhwc && ep.output_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_GEMM_KERNEL(1, GKF_FCF)
        } else if (ep.ws == 2) {
          BIND_GEMM_KERNEL(2, GKF_FCF)
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (ep.input_fmt == nhwc && ep.output_fmt == nChw16c) {
        if (ep.ws == 1) {
          BIND_GEMM_KERNEL(1, GKF_FCD)
        } else if (ep.ws == 2) {
          BIND_GEMM_KERNEL(2, GKF_FCD)
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (ep.input_fmt == nChw16c && ep.output_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_GEMM_KERNEL(1, GKF_DCF)
        } else if (ep.ws == 2) {
          BIND_GEMM_KERNEL(2, GKF_DCF)
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else { // blocked
        if (ep.ws == 1) {
          BIND_GEMM_KERNEL(1, GKF_DCD)
        } else if (ep.ws == 2) {
          BIND_GEMM_KERNEL(2, GKF_DCD)
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      }
      break;
    default:
      el_error("Unknown direct gemm kernel xopt");
      break;
    }
  };

  auto bind_conv_kernel = [&](int O, int T,
      u8s8_conv_kernel_binder::kconv<TarrayTypes, OutputType> **func, int K) {
    switch (xopt_) {
    case (0xc160):
      if (compact_ir_weights_ && ep.input_fmt == nhwc && ep.output_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_FBF, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_FBF, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_FBF, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (ep.input_fmt == nhwc && ep.output_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_FCF, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_FCF, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_FCF, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (compact_ir_weights_ && ep.input_fmt == nhwc &&
                 ep.output_fmt == nChw16c) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_FBD, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_FBD, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_FBD, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (!compact_ir_weights_ && ep.input_fmt == nChw16c &&
                 ep.output_fmt == nChw16c) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_DCD, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_DCD, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_DCD, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (ep.input_fmt == nhwc && ep.output_fmt == nChw16c) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_FCD, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_FCD, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_FCD, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else if (ep.input_fmt == nChw16c && ep.output_fmt == nhwc) {
        if (ep.ws == 1) {
          BIND_CONV_KERNEL(1, GKF_DCF, K);
        } else if (ep.ws == 2) {
          if (ep.lp > ep.rp) {
            BIND_CONV_KERNEL(S2_LLP, GKF_DCF, K);
          } else {
            BIND_CONV_KERNEL(2, GKF_DCF, K);
          }
        } else {
          el_error("Stride > 2 not yet bounded");
        }
      } else {
        el_error("direct:int8: kernel fmt not supported");
      }
      break;
    default:
      el_error("Unknown xopt");
      break;
    }
  };

  if (xopt_ == 0xc160 /*|| xopt_ == 0xb160*/) {
    bind_conv_kernel(ep.O, ep.T, &ker_conv_, ep.kw);
    bind_conv_kernel(ep.O, ep.Tr, &ker_conv_Tr_, ep.kw);
  } else if (xopt_ == 0xa160) {
    if (ep.wt > 128) {
      el_error("direct: a160: wt > max-kernel-slot:128");
    }
    iter_each (_wt, ep.wt) {
      int Tz = _wt == ep.wt - 1 ? ep.Tr : ep.T;
      for (int _kw = 0; _kw < ep.kw; ++_kw) {
        // _iws, _iwe
        // _iw = ws * _ow + _kw - lp
        auto ows0 = _wt * ep.T;
        auto owe0 = _wt * ep.T + Tz - 1;
        auto _iws = ep.ws * ows0 + _kw - ep.lp;
        while (_iws < 0)
          _iws += ep.ws;
        auto _iwe = ep.ws * owe0 + _kw - ep.lp;
        while (_iwe > ep.iw - 1)
          _iwe -= ep.ws;
        auto _ows = (_iws + ep.lp - _kw) / ep.ws;
        auto _owe = (_iwe + ep.lp - _kw) / ep.ws;
        bind_gemm_kernel(ep.O, _owe - _ows + 1, &ker_gemm_[_wt][_kw]);
      }
    }
  }

#define EXECUTE_CASE(n)                                                        \
  case 0x##n:                                                                  \
    execute_opt_ = &Instance_elx_int8_conv_direct_t::__execute_##n;            \
    break

  switch (xopt_) {
    EXECUTE_CASE(c160);
    EXECUTE_CASE(a160);
  default:
    el_error("Unimplemented direct lp xopt");
    break;
  }
}

} // namespace euler
