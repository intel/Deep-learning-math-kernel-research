#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_U8S8_KGEMM_TBL(type, otype, V, Vx, I, S, F)                       \
  static kgemm<conv_impl::type, otype>                                         \
      *kgemm_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[8][32]
#else
#define DECL_U8S8_KGEMM_TBL(type, otype, V, Vx, I, S, F)                       \
  __u8s8_kgemm_generate_inst__ u8s8_gemm type otype V Vx I S F
#endif

#define LOOKUP_U8S8_KGEMM_TBL(type, otype, V, Vx, I, S, F, O, T)               \
  kgemm_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_u8s8_gemm.hxx"

namespace euler {

struct u8s8_gemm_kernel_binder {
  template <typename GarrayTypes, typename OoutputType, int V, int Vx, int I, int... Kp>
  using gemm_ker_cls = typename euler::u8s8_gemm_kernel<GarrayTypes,
      OoutputType, V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes, typename OoutputType>
  using kgemm
      = decltype(gemm_ker_cls<GarrayTypes, OoutputType, 1, 1, 1, 1, 1, 1, 1, 1>::gemm);

#endif // BUILD_OTJ_TBL

  // wino
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_CCC); // wino, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_CCC);

  DECL_U8S8_KGEMM_TBL(INT8_F16o, float, 16, 4, ISA_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c
  DECL_U8S8_KGEMM_TBL(INT8_F16o, int8_t, 16, 4, ISA_AVX512, 1, GKF_CCC);
  DECL_U8S8_KGEMM_TBL(INT8_F16o, uint8_t, 16, 4, ISA_AVX512, 1, GKF_CCC);

  // direct, blocked
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm

  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_DCD); // direct, blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_DCD); // direct, blocked, int8-gemm, uint8-output

  // direct, nhwc
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FCF); // direct, nhwc, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FCF); // direct, nhwc, int8-gemm

  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FCF); // direct, nhwc, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FCF); // direct, nhwc, int8-gemm, int8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FCF); // direct, nhwc, int8-gemm, uint8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FCF); // direct, nhwc, int8-gemm, uint8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FCD); // direct, nhwc, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FCD); // direct, nhwc, int8-gemm

  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FCD); // direct, nhwc->blocked, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FCD); // direct, nhwc->blocked, int8-gemm, int8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FCD); // direct, nhwc->blocked, int8-gemm, uint8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FCD); // direct, nhwc->blocked, int8-gemm, uint8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_DCF); // direct, nhwc, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_DCF); // direct, nhwc, int8-gemm

  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_DCF); // direct, blocked->nhwc, int8-gemm, int8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_DCF); // direct, blocked->nhwc, int8-gemm, int8-output

  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_DCF); // direct, blocked->nhwc, int8-gemm, uint8-output
  DECL_U8S8_KGEMM_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_DCF); // direct, blocked->nhwc, int8-gemm, uint8-output


#ifdef ENABLE_USER_FP16
  DECL_U8S8_KGEMM_TBL(INT8_F16b, float, 16, 4, ISA_AVX512, 1, GKF_CCC); // wino, int8-gemm, user f16
  DECL_U8S8_KGEMM_TBL(INT8_F16ob, float, 16, 4, ISA_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c + user f16
#endif

#if !defined(BUILD_OTJ_TBL)

#define DEF_BIND_INT8_F32(otype)                                               \
  /* GarrayTypes->u8s8f32f32, used by WINO with f32 UserTypes */               \
  template <int S, int F>                                                      \
  static inline void bind(                                                     \
      int O, int T, kgemm<conv_impl::INT8_F32, otype> **func)                  \
  {                                                                            \
    switch (F) {                                                               \
    case GKF_CCC:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_CCC, O, T);             \
      break;                                                                   \
    case GKF_DCD:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_DCD, O, T);             \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 2, GKF_DCD, O, T);             \
      break;                                                                   \
    case GKF_FCF:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_FCF, O, T);             \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 2, GKF_FCF, O, T);             \
      break;                                                                   \
    case GKF_DCF:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_DCF, O, T);             \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 2, GKF_DCF, O, T);             \
      break;                                                                   \
    case GKF_FCD:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_FCD, O, T);             \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KGEMM_TBL(                                         \
            INT8_F32, otype, 16, 4, ISA_AVX512, 2, GKF_FCD, O, T);             \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  }

  DEF_BIND_INT8_F32(float)
  DEF_BIND_INT8_F32(int8_t)
  DEF_BIND_INT8_F32(uint8_t)

  // GarrayTypes->u8s8f16f32, used by WINO with f32 UserTypes
  template <int S, int F>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16o, float> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(
            INT8_F16o, float, 16, 4, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

#ifdef ENABLE_USER_FP16
  // GarrayTypes->u8s8f32f16, used by WINO with f16 UserTypes
  template <int S, int F>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16b, float> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(
            INT8_F16b, float, 16, 4, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f16f16, used by WINO with f16 UserTypes
  template <int S, int F>
  static inline void bind(
      int O, int T, kgemm<conv_impl::INT8_F16ob, float> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(
            INT8_F16ob, float, 16, 4, ISA_AVX512, 1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }
#endif

#endif // BUILD_OTJ_TBL
};

} // namespace euler
