#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_U8S8_KGEMM_TBL(type, otype, fmaopt, V, Vx, I, S, F)               \
  static kgemm<conv_impl::type, otype>                                 \
      *kgemm_##type##_##otype##_##fmaopt##_##V##_##Vx##_##I##_##S##_##F[8][32]
#else
#define DECL_U8S8_KGEMM_TBL(type, otype, fmaopt, V, Vx, I, S, F)               \
  __u8s8_kgemm_generate_inst__ u8s8_gemm type otype fmaopt V Vx I S F
#endif

#define LOOKUP_U8S8_KGEMM_TBL(type, otype, fmaopt, V, Vx, I, S, F, O, T)       \
  kgemm_##type##_##otype##_##fmaopt##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_u8s8_gemm_otj.hxx"

namespace euler {

struct u8s8_gemm_kernel_binder {
  template <typename GarrayTypes, typename OoutputType,
            bool FmaOpt, int V, int Vx, int I, int... Kp>
  using gemm_ker_cls = typename euler::u8s8_gemm_kernel_otj<GarrayTypes,
      OoutputType, FmaOpt, V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes, typename OoutputType>
  using kgemm = decltype(gemm_ker_cls<
      GarrayTypes, OoutputType, false, 1, 1, 1, 1, 1, 1, 1, 1>::gemm);

#endif // BUILD_OTJ_TBL

  DECL_U8S8_KGEMM_TBL(INT8_F32, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);

  DECL_U8S8_KGEMM_TBL(INT8_F16o, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c
  DECL_U8S8_KGEMM_TBL(INT8_F16o, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c
  DECL_U8S8_KGEMM_TBL(INT8_F16o, int8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F16o, int8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F16o, uint8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F16o, uint8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);

  DECL_U8S8_KGEMM_TBL(INT8_F32, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, false, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, true, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm

  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, false, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, true, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, false, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, true, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, false, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, true, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output

#ifdef ENABLE_USER_FP16
  DECL_U8S8_KGEMM_TBL(INT8_F16b, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, user f16
  DECL_U8S8_KGEMM_TBL(INT8_F16ob, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c + user f16
#endif

#if !defined(BUILD_OTJ_TBL)

#define DEF_BIND_INT8_F32(otype)                                               \
  /* GarrayTypes->u8s8f32f32, used by WINO with f32 UserTypes */               \
  template <int S, int F, bool fmaopt>                                         \
  static inline void bind(                                                     \
      int O, int T, kgemm<conv_impl::INT8_F32, otype> **func)                  \
  {                                                                            \
    if (fmaopt) {                                                              \
      switch (F) {                                                             \
      case GKF_CCC:                                                            \
        if (S == 1)                                                            \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T); \
        break;                                                                 \
      case GKF_DCD:                                                            \
        if (S == 1)                                                            \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, true, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD, O, T); \
        else if (S == 2)                                                       \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, true, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD, O, T); \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
      }                                                                        \
    } else {                                                                   \
      switch (F) {                                                             \
      case GKF_CCC:                                                            \
        if (S == 1)                                                            \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);\
        break;                                                                 \
      case GKF_DCD:                                                            \
        if (S == 1)                                                            \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, false, 16, 4, ISA_SKX_AVX512, 1, GKF_DCD, O, T);\
        else if (S == 2)                                                       \
          *func = LOOKUP_U8S8_KGEMM_TBL(                                       \
              INT8_F32, otype, false, 16, 4, ISA_SKX_AVX512, 2, GKF_DCD, O, T);\
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

  DEF_BIND_INT8_F32(float)
  DEF_BIND_INT8_F32(int8_t)
  DEF_BIND_INT8_F32(uint8_t)

  // GarrayTypes->u8s8f16f32, used by WINO with f32 UserTypes
  template <int S, int F, bool fmaopt>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16o, float> **func)
  {
    if (fmaopt) {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16o, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    } else {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16o, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    }
  }

#ifdef ENABLE_USER_FP16
  // GarrayTypes->u8s8f32f16, used by WINO with f16 UserTypes
  template <int S, int F, bool fmaopt>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16b, float> **func)
  {
    if (fmaopt) {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16b, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    } else {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16b, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    }
  }

  // GarrayTypes->u8s8f16f16, used by WINO with f16 UserTypes
  template <int S, int F, bool fmaopt>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16ob, float> **func)
  {
    if (fmaopt) {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16ob, float, true, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    } else {
      switch (F) {
      case GKF_CCC:
        if (S == 1)
          *func = LOOKUP_U8S8_KGEMM_TBL(
              INT8_F16ob, float, false, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T);
        break;
      default:
        break;
      }
    }
  }
#endif

#endif // BUILD_OTJ_TBL
};

} // namespace euler
