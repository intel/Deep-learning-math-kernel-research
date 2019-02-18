#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_U8S8_KGEMM_TBL(type, V, Vx, I, S, F)                              \
  static kgemm<conv_impl::type>                                                \
      *kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[8][32]
#else
#define DECL_U8S8_KGEMM_TBL(type, V, Vx, I, S, F)                                   \
  __kgemm_generate_inst__ u8s8_gemm type V Vx I S F
#endif

#define LOOKUP_U8S8_KGEMM_TBL(type, V, Vx, I, S, F, O, T)                      \
  kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1]

#if !defined(BUILD_OTJ_TBL)
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_u8s8_gemm_otj.hxx"

namespace euler {

struct u8s8_gemm_kernel_binder {
  template <typename GarrayTypes, int V, int Vx, int I, int... Kp>
  using gemm_ker_cls = typename euler::u8s8_gemm_kernel_otj<GarrayTypes,
      V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes>
  using kgemm
      = decltype(gemm_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, 1>::gemm);

#endif // BUILD_OTJ_TBL

  DECL_U8S8_KGEMM_TBL(INT8_F32, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm
  DECL_U8S8_KGEMM_TBL(INT8_F16o, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c

#ifdef ENABLE_USER_FP16
  DECL_U8S8_KGEMM_TBL(INT8_F16b, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, user f16
  DECL_U8S8_KGEMM_TBL(INT8_F16ob, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC); // wino, int8-gemm, f16c + user f16
#endif

#if !defined(BUILD_OTJ_TBL)

  // GarrayTypes->u8s8f32f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F32> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(INT8_F32, 16, 4, ISA_SKX_AVX512,
            1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f16f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16o> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(INT8_F16o, 16, 4, ISA_SKX_AVX512,
            1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

#ifdef ENABLE_USER_FP16
  // GarrayTypes->u8s8f32f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16b> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(INT8_F16b, 16, 4, ISA_SKX_AVX512,
            1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f16f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16ob> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_U8S8_KGEMM_TBL(INT8_F16ob, 16, 4, ISA_SKX_AVX512,
            1, GKF_CCC, O, T);
      break;
    default:
      break;
    }
  }
#endif

#endif // BUILD_OTJ_TBL
};

} // namespace euler
