#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_U8S8_DEPTHWISE_KCONV_TBL(type, otype, V, Vx, I, S, F)             \
  static kconv<conv_impl::type, otype>                                         \
      *kconv_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[1][32][1]
#else
#define DECL_U8S8_DEPTHWISE_KCONV_TBL(type, otype, V, Vx, I, S, F)             \
  __u8s8_depthwise_kconv_generate_inst__ u8s8_depthwise_conv type otype V Vx I S F
#endif

#define LOOKUP_U8S8_DEPTHWISE_KCONV_TBL(type, otype, V, Vx, I, S, F, O, T, K)  \
  kconv_##type##_##otype##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][0]

#if !defined(BUILD_OTJ_TBL)
#include "el_def.hpp"
#include "src/kernel/elk_def.hpp"
#include "src/kernel/elk_u8s8_depthwise_conv.hxx"

namespace euler {

struct u8s8_depthwise_conv_kernel_binder {
  template <typename GarrayTypes, typename RoutputType, int V, int Vx, int I,
            int... Kp>
  using conv_ker_cls = typename euler::u8s8_depthwise_conv_kernel<
      GarrayTypes, RoutputType, V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes, typename RoutputType>
  using kconv = decltype(
      conv_ker_cls<GarrayTypes, RoutputType, 1, 1, 1, 1, 1, 1, 1, 1>::conv);

#endif // BUILD_OTJ_TBL

  DECL_U8S8_DEPTHWISE_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 1, GKF_DCD);
  DECL_U8S8_DEPTHWISE_KCONV_TBL(INT8_F32, int8_t, 16, 4, ISA_AVX512, 2, GKF_DCD);
  DECL_U8S8_DEPTHWISE_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 1, GKF_DCD);
  DECL_U8S8_DEPTHWISE_KCONV_TBL(INT8_F32, uint8_t, 16, 4, ISA_AVX512, 2, GKF_DCD);

#if !defined(BUILD_OTJ_TBL)

#  define DEF_DEPTHWISE_CONV_BIND_INT8_F32(otype)                           \
    template <int S, int F, int K>                                          \
    static inline void bind(int O, int T,                                   \
                            kconv<conv_impl::INT8_F32, otype> **func) {     \
      switch (F) {                                                          \
      case GKF_DCD:                                                         \
        if (S == 1)                                                         \
          *func = LOOKUP_U8S8_DEPTHWISE_KCONV_TBL(                          \
              INT8_F32, otype, 16, 4, ISA_AVX512, 1, GKF_DCD, O, T, K);     \
        else if (S == 2)                                                    \
          *func = LOOKUP_U8S8_DEPTHWISE_KCONV_TBL(                          \
              INT8_F32, otype, 16, 4, ISA_AVX512, 2, GKF_DCD, O, T, K);     \
        break;                                                              \
      default:                                                              \
        el_error("Unimlemented conv kernel format");                        \
        break;                                                              \
      }                                                                     \
    }

  DEF_DEPTHWISE_CONV_BIND_INT8_F32(int8_t)
  DEF_DEPTHWISE_CONV_BIND_INT8_F32(uint8_t)
#endif // BUILD_OTJ_TBL
};

} // namespace euler
