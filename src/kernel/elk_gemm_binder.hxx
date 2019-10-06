#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_KGEMM_TBL(type, V, Vx, I, S, F)                                   \
  static kgemm<conv_impl::type>                                                \
      *kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[8][32]
#else
#define DECL_KGEMM_TBL(type, V, Vx, I, S, F)                                   \
  __kgemm_generate_inst__ gemm type V Vx I S F
#endif

#define LOOKUP_KGEMM_TBL(type, V, Vx, I, S, F, O, T)                           \
  kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_gemm.hxx"

namespace euler {

struct gemm_kernel_binder {
  template <typename GarrayTypes, int V, int Vx, int I, int... Kp>
  using gemm_ker_cls = typename euler::gemm_kernel<GarrayTypes,
      V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes>
  using kgemm
      = decltype(gemm_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, 1>::gemm);

#endif // BUILD_OTJ_TBL

  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_CCC); // wino, 1x1
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_CCD); // 1x1
  //DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_ECD); // direct, nchw input
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, 1x1, blocked
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, nhwc
  //DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_CCC); // 1x1
  //DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_CCD); // 1x1
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_DCD); // direct, blocked
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_FCF); // direct, nhwc

  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_CCC); // 1x1, f16c
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_CCD); // 1x1, f16c
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, 1x1, f16c
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF); // direct, 1x1, f16c
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_FCF); // direct, 1x1, f16c
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_DCD); // direct, 1x1, f16c
  //DECL_KGEMM_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_ECD); // direct, nchw input, f16c
  DECL_KGEMM_TBL(FP32_F16iwo, 16, 1, ISA_AVX512, 1, GKF_CCC); // wino, f16c

#ifdef ENABLE_USER_FP16
  DECL_KGEMM_TBL(FP32_F16b, 16, 1, ISA_AVX512, 1, GKF_CCC); // wino, user f16
  DECL_KGEMM_TBL(FP32_F16wob, 16, 1, ISA_AVX512, 1, GKF_CCC); // wino, f16c + user f16
  DECL_KGEMM_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD); // direct, 1x1, f16c
#endif

#if !defined(BUILD_OTJ_TBL)
  // GarrayTypes->f32f32f32f32, used by WINO with f32 UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_CCC, O, T);
      //else if (S == 2)
      //  *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_CCC, O, T);
      break;
    case GKF_CCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_CCD, O, T);
      //else if (S == 2)
      //  *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_CCD, O, T);
      break;
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_DCD, O, T);
      break;
    //case GKF_ECD:
    //  if (S == 1)
    //    *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_ECD, O, T);
    //  break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_AVX512, 2, GKF_FCF, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f16f32, used by WINO with f32 UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16iwo> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16iwo, 16, 1, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f32f32, used by CONV 1x1 with f32 UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16w> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    case GKF_CCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_CCD, O, T);
      break;
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_DCD, O, T);
      break;
    case GKF_FCF:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 1, GKF_FCF, O, T);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_AVX512, 2, GKF_FCF, O, T);
      break;
    default:
      break;
    }
  }

#ifdef ENABLE_USER_FP16
  // GarrayTypes->f32f32f32f16, used by WINO with f16 UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16b> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16b, 16, 1, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f16f16, used by WINO with f16 UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16wob> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16wob, 16, 1, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f32f16f32, used by DIRECT CONV with f16o UserTypes
  template <int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16o> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_DCD, O, T);
      break;
    //case GKF_ECD:
    //  if (S == 1)
    //    *func = LOOKUP_KGEMM_TBL(FP32_F16o, 16, 1, ISA_AVX512, 1, GKF_ECD, O, T);
    //  break;
    default:
      break;
    }
  }

#endif

#endif // BUILD_OTJ_TBL
};

} // namespace euler
