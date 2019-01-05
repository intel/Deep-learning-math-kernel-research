#pragma once

#if !defined(BUILD_OTJ_TBL)
#define DECL_KGEMM_TBL(type, V, Vx, I, S, F)                                   \
  static kgemm<conv_impl::type>                                                \
      *kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[8][32][2]
#define DECL_KCONV_TBL(type, V, Vx, I, S, F)                                   \
  static kconv<conv_impl::type>                                                \
      *kconv_##type##_##V##_##Vx##_##I##_##S##_##F[8][32][3]
#else
#define DECL_KGEMM_TBL(type, V, Vx, I, S, F)                                   \
  __kgemm_generate_inst__ gemm type V Vx I S F
#define DECL_KCONV_TBL(type, V, Vx, I, S, F)                                   \
  __kconv_generate_inst__ conv type V Vx I S F
#endif

#define LOOKUP_KGEMM_TBL(type, V, Vx, I, S, F, O, T, has_Ir)                   \
  kgemm_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][has_Ir]
#define LOOKUP_KCONV_TBL(type, V, Vx, I, S, F, O, T, K)                        \
  kconv_##type##_##V##_##Vx##_##I##_##S##_##F[O - 1][T - 1][K/2-1]

#if !defined(BUILD_OTJ_TBL)
#include "el_intrin.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "el_stl.hpp"
#include "elx_conv.hpp"
#include "src/kernel/elk_gemm_otj.hxx"

namespace euler {

struct gemm_kernel_binder {
  template <typename GarrayTypes, int V, int Vx, int I, int... Kp>
  using gemm_ker_cls = typename euler::gemm_kernel_otj<GarrayTypes,
      V, Vx, I, estl::integer_sequence<Kp...>>;

  template <typename GarrayTypes>
  using kgemm
      = decltype(gemm_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, false>::gemm);

  template <typename GarrayTypes>
  using kconv
      = decltype(gemm_ker_cls<GarrayTypes, 1, 1, 1, 1, 1, 1, 1, false>::conv);

#endif // BUILD_OTJ_TBL

  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_CCD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DDD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_CCC);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_CCD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_DCD);
  DECL_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_DDD);

  DECL_KGEMM_TBL(FP32_F16b, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(FP32_F16w, 16, 1, ISA_SKX_AVX512, 1, GKF_CCD);
  DECL_KGEMM_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD);
  DECL_KGEMM_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD);
  DECL_KGEMM_TBL(FP32_F16wo, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(FP32_F16wob, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC);

  DECL_KGEMM_TBL(INT8_F32, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(INT8_F16b, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(INT8_F16o, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);
  DECL_KGEMM_TBL(INT8_F16ob, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC);

  DECL_KCONV_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD);
  DECL_KCONV_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD);
  DECL_KCONV_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD);
  DECL_KCONV_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD);

#if !defined(BUILD_OTJ_TBL)
  // GarrayTypes->f32f32f32f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_CCC, O, T, has_Ir);
      break;
    case GKF_CCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_CCD, O, T, has_Ir);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_CCD, O, T, has_Ir);
      break;
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD, O, T, has_Ir);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_DCD, O, T, has_Ir);
      break;
    case GKF_ECD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD, O, T, has_Ir);
      break;
    case GKF_DDD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DDD, O, T, has_Ir);
      else if (S == 2)
        *func = LOOKUP_KGEMM_TBL(FP32, 16, 1, ISA_SKX_AVX512, 2, GKF_DDD, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f32f32f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16b> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16b, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f16f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16wo> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16wo, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f16f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16wob> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16wob, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f32f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F32> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(INT8_F32, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f32f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16b> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(INT8_F16b, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f16f32, used by WINO with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16o> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(INT8_F16o, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->u8s8f16f16, used by WINO with f16 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::INT8_F16ob> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(INT8_F16ob, 16, 4, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f16f32f32, used by CONV 1x1 with f32 UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16w> **func)
  {
    switch (F) {
    case GKF_CCC:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_SKX_AVX512, 1, GKF_CCC, O, T, has_Ir);
      break;
    case GKF_CCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16w, 16, 1, ISA_SKX_AVX512, 1, GKF_CCD, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  // GarrayTypes->f32f32f16f32, used by DIRECT CONV with f16o UserTypes
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<conv_impl::FP32_F16o> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD, O, T, has_Ir);
      break;
    case GKF_ECD:
      if (S == 1)
        *func = LOOKUP_KGEMM_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD, O, T, has_Ir);
      break;
    default:
      break;
    }
  }

  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, int K>
  static inline void bind(int O, int T, kconv<conv_impl::FP32> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD, O, T, K);
      break;
    case GKF_ECD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD, O, T, K);
      break;
    default:
      break;
    }
  }

  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, int K>
  static inline void bind(int O, int T, kconv<conv_impl::FP32_F16o> **func)
  {
    switch (F) {
    case GKF_DCD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_DCD, O, T, K);
      break;
    case GKF_ECD:
      if (S == 1)
        *func = LOOKUP_KCONV_TBL(FP32_F16o, 16, 1, ISA_SKX_AVX512, 1, GKF_ECD, O, T, K);
      break;
    default:
      break;
    }
  }

#endif // BUILD_OTJ_TBL

// Implicit instantiation is disabled b/c slow build. This build path can be
// broken. Code here is for reference only.
// TODO: code cleanup.
#if 0
  template <typename GarrayTypes, int V, int Vx, int I, int S, int F, bool has_Ir>
  static inline void bind(int O, int T, kgemm<GarrayTypes> **func)
  {
    switch (O) {
    case 1:
      LOOP_FROM_TO(_T, 1, 32, {
        if (T == _T)
          (*func = gemm_ker_cls< GarrayTypes, V, Vx, I,
               S, F, 1, _T, has_Ir>::gemm);
      });
      if (T >= 32)
        el_error("gemm_kernel: O = 1, T >= 32 not supported");
      break;
    case 2:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 2, _T, has_Ir>::gemm);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 2, T >= 15 not supported");
      break;
    case 3:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 3, _T, has_Ir>::gemm);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 3, T >= 15 not supported");
      break;
    case 4:
      LOOP_FROM_TO(_T, 1, 15, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 4, _T, has_Ir>::gemm);
      });
      if (T >= 15)
        el_error("gemm_kernel: O = 4, T >= 15 not supported");
      break;
    case 5:
      LOOP_FROM_TO(_T, 1, 6, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 5, _T, has_Ir>::gemm);
      });
      if (T >= 6)
        el_error("gemm_kernel: O = 5, T >= 6 not supported");
      break;
    case 6:
      LOOP_FROM_TO(_T, 1, 5, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 6, _T, has_Ir>::gemm);
      });
      if (T >= 5)
        el_error("gemm_kernel: O = 6, T >= 5 not supported");
      break;
    case 7:
      LOOP_FROM_TO(_T, 1, 4, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 7, _T, has_Ir>::gemm);
      });
      if (T >= 4)
        el_error("gemm_kernel: O = 7, T >= 4 not supported");
      break;
    case 8:
      LOOP_FROM_TO(_T, 1, 9, {
        if (T == _T)
          (*func = gemm_ker_cls<GarrayTypes, V, Vx, I,
               S, F, 8, _T, has_Ir>::gemm);
      });
      if (T >= 9)
        el_error("gemm_kernel: O = 8, T >= 9 not supported");
      break;
    default:
      el_error("gemm_kenrel: O > 8 unsupported");
    }
  }
#endif
};

} // namespace euler
