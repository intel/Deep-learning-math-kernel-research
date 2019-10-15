#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_KCONV_TBL(type, V, Vx, I, S, F)                                   \
  static kconv<conv_impl::type>                                                \
      *kconv_##type##_##V##_##Vx##_##I##_##S##_##F[8][32][3]
#else
#define DECL_KCONV_TBL(type, V, Vx, I, S, F)                                   \
  __kconv_generate_inst__ conv type V Vx I S F
#endif

#define LOOKUP_KCONV_TBL(type, V, Vx, I, S, F, O, T, K)                        \
  kconv_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][K/2-1]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_conv.hxx"

namespace euler {

struct conv_kernel_binder {
  template <typename GarrayTypes, int V, int Vx, int I, int... Kp>
  using conv_ker_cls = typename euler::conv_kernel<GarrayTypes,
      V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes>
  using kconv
      = decltype(conv_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, 1>::conv);

#endif // BUILD_OTJ_TBL

  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, blocked
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_DCD); // direct, blocked
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_DCD); // direct, blocked
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_EBD); // direct, nchw input
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_EBD); // direct, nchw input
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_EBD); // direct, nchw input
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, nhwc
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_FCF); // direct, nhwc
  DECL_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_FCF); // direct, nhwc
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, blocked, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_DCD); // direct, blocked, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_DCD); // direct, blocked, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_EBD); // direct, nchw input, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_EBD); // direct, nchw input, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_EBD); // direct, nchw input, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, nhwc input, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_FCF); // direct, nhwc input, f16c
  DECL_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_FCF); // direct, nhwc input, f16c
  //DECL_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, f16c

#ifdef ENABLE_USER_FP16
  DECL_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_EBD); // direct, nchw input, f16c
#endif

#if !defined(BUILD_OTJ_TBL)

#ifdef ENABLE_USER_FP16
  template <int S, int F, int K>
  static inline void bind(int O, int T, kconv<conv_impl::FP32_F16o> **func)
  {
    switch (F) {
    //case GKF_DCD:
    //  if (S == 1)
    //    *func = LOOKUP_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K);
    //  break;
    case GKF_EBD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_EBD, O, T, K);
      break;
    default:
      break;
    }
  }
#endif

  template <int S, int F, int K>
  static inline void bind(int O, int T, kconv<conv_impl::FP32> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_DCD, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_DCD, O, T, K);
      break;
    case GKF_EBD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_EBD, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_EBD, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_EBD, O, T, K);
      break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_FCF, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_AVX512, S2_LLP, GKF_FCF, O, T, K);
      break;
    default:
      break;
    }
  }

  template <int S, int F, int K>
  static inline void bind(int O, int T, kconv<conv_impl::FP32_F16w> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_DCD, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_DCD, O, T, K);
      break;
    case GKF_EBD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_EBD, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_EBD, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_EBD, O, T, K);
      break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T, K);
      else if (S == 2)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_FCF, O, T, K);
      else if (S == S2_LLP)
        *func = LOOKUP_KCONV_TBL(FP32_F16w, 16, 1, ISA_AVX512, S2_LLP, GKF_FCF, O, T, K);
      break;
    default:
      break;
    }
  }

#endif // BUILD_OTJ_TBL
};

} // namespace euler
