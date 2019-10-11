#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_U8S8_KCONV_TBL(type, otype, V, Vx, I, S, F)                       \
  static kconv<conv_impl::type, otype>                                         \
      *kconv_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[8][32][3]
#else
#define DECL_U8S8_KCONV_TBL(type, otype, V, Vx, I, S, F)                       \
  __u8s8_kconv_generate_inst__ u8s8_conv type otype V Vx I S F
#endif

#define LOOKUP_U8S8_KCONV_TBL(type, otype, V, Vx, I, S, F, O, T, K)            \
  kconv_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][K/2-1]

#if !defined(BUILD_OTJ_TBL)
#include "el_def.hpp"
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_u8s8_conv.hxx"

namespace euler {

struct u8s8_conv_kernel_binder {
  template <typename GarrayTypes, typename RoutputType, int V, int Vx, int I, int... Kp>
  using conv_ker_cls = typename euler::u8s8_conv_kernel<GarrayTypes,
      RoutputType, V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes, typename RoutputType>
  using kconv
      = decltype(conv_ker_cls<GarrayTypes, RoutputType, 1, 1, 1, 1, 1, 1, 1, 1>::conv);

#endif // BUILD_OTJ_TBL

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_DCD); // direct, blocked, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_DCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_DCD);

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FCF); // direct, nhwc, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FCF);

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FCD); // direct, nhwc->blocked, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FCD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FCD);

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_DCF); // direct, blocked->nhwc, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_DCF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_DCF);

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FBD); // direct, nhwc->blocked, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FBD);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FBD);

  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 1, GKF_FBF); // direct, nhwc->nhwc, int8-conv
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, 2, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, float, 16, 4, ISA_AVX512, S2_LLP, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_FBF);
  DECL_U8S8_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, S2_LLP, GKF_FBF);



#if !defined(BUILD_OTJ_TBL)

#define DEF_CONV_BIND_INT8_F32(otype)                                          \
  template <int S, int F, int K>                                               \
  static inline void bind(                                                     \
      int O, int T, kconv<conv_impl::INT8_F32, otype> **func)                  \
  {                                                                            \
    switch (F) {                                                               \
    case GKF_DCD:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_DCD, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_DCD, O, T, K);                                              \
      else if (S == S2_LLP)                                                    \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_DCD, O, T, K);                                         \
      break;                                                                   \
    case GKF_FBF:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_FBF, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_FBF, O, T, K);                                              \
      else if (S == S2_LLP)                                                    \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_FBF, O, T, K);                                         \
      break;                                                                   \
    case GKF_FCF:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_FCF, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_FCF, O, T, K);                                              \
      else if (S == S2_LLP)                                                    \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_FCF, O, T, K);                                         \
      break;                                                                   \
    case GKF_DCF:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_DCF, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_DCF, O, T, K);                                              \
      else if (S == S2_LLP)                                                    \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_DCF, O, T, K);                                         \
      break;                                                                   \
    case GKF_FCD:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_FCD, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_FCD, O, T, K);                                              \
      else if (S == S2_LLP)                                                    \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_FCD, O, T, K);                                         \
      break;                                                                   \
    case GKF_FBD:                                                              \
      if (S == 1)                                                              \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            1, GKF_FBD, O, T, K);                                              \
      else if (S == 2)                                                         \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            2, GKF_FBD, O, T, K);                                              \
       else if (S == S2_LLP)                                                   \
        *func = LOOKUP_U8S8_KCONV_TBL(INT8_F32, otype, 16, 4, ISA_AVX512,      \
            S2_LLP, GKF_FBD, O, T, K);                                         \
      break;                                                                   \
    default:                                                                   \
      el_error("Unimlemented conv kernel format");                             \
      break;                                                                   \
    }                                                                          \
  }

  DEF_CONV_BIND_INT8_F32(float)
  DEF_CONV_BIND_INT8_F32(int8_t)
  DEF_CONV_BIND_INT8_F32(uint8_t)
#endif // BUILD_OTJ_TBL
};

} // namespace euler
