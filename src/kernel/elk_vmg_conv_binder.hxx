#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_VMG_KCONV_TBL(type, V, Vx, I, S, F)                               \
  static kconv<conv_impl::type>                                                \
      *kconv_##type##_##V##_##Vx##_##I##_##S##_##F[1][32][3][5]
#else
#define DECL_VMG_KCONV_TBL(type, V, Vx, I, S, F)                               \
  __vmg_kconv_generate_inst__ vmg_conv type V Vx I S F
#endif

#define _LOG2(x) (x) == 16 ? 4 : (x) == 8 ? 3 : (x) == 4 ? 2 : (x) == 2 ? 1 : 0
#define LOOKUP_VMG_KCONV_TBL(type, V, Vx, I, S, F, O, T, K, G)                 \
  kconv_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][K/2-1][_LOG2(G)]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_vmg_conv.hxx"

namespace euler {

struct vmg_conv_kernel_binder {
  template <typename GarrayTypes, int V, int Vx, int I, int... Kp>
  using conv_ker_cls = typename euler::vmg_conv_kernel<GarrayTypes,
      V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes>
  using kconv
      = decltype(conv_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, 1, 1>::conv);

#endif // BUILD_OTJ_TBL

  DECL_VMG_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, blocked
  DECL_VMG_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, nhwc
  DECL_VMG_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, blocked, f16c
  DECL_VMG_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, nhwc input, f16c
  //DECL_VMG_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, f16c

#ifdef ENABLE_USER_FP16
  DECL_VMG_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_EBD); // direct, nchw input, f16c
#endif

#if !defined(BUILD_OTJ_TBL)

#ifdef ENABLE_USER_FP16
  template <int S, int F, int K, int G>
  static inline void bind(int O, int T, kconv<conv_impl::FP32_F16o> **func)
  {
    switch (F) {
    //case GKF_DCD:
    //  if (S == 1)
    //    *func = LOOKUP_VMG_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K, G);
    //  break;
    case GKF_EBD:
      if (S == 1)
        *func = LOOKUP_VMG_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_EBD, O, T, K, G);
      break;
    default:
      break;
    }
  }
#endif

  template <int S, int F, int K, int G>
  static inline void bind(int O, int T, kconv<conv_impl::FP32> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_VMG_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K, G);
      break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_VMG_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T, K, G);
      break;
    default:
      break;
    }
  }

  template <int S, int F, int K, int G>
  static inline void bind(int O, int T, kconv<conv_impl::FP32_F16w> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_VMG_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K, G);
      break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_VMG_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T, K, G);
      break;
    default:
      break;
    }
  }

#endif // BUILD_OTJ_TBL
};

} // namespace euler
