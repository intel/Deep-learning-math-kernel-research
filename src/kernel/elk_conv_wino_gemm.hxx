#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

#ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
#error "Don't include this file directly"
#endif

// blocking -
// oc3, ic3, A * A, O2, I2, V, V
// t2, A*A, ic3, I2, T, V
// t2, A*A, oc3, O2, T, V

namespace euler {

template <typename Type>
void elk_gemm_ker(
    Type* mxp, Type* mxn, Type* nxp, int m, int n, int p, bool zero_out)
{
  MD(Type, amxn, [m][n], mxn);
  MD(Type, anxp, [n][p], nxp);
  MD(Type, amxp, [m][p], mxp);

  for_each (_m, m) {
    for_each (_p, p) {
      if (zero_out)
        amxp[_m][_p] = 0.0f;
      for_each (_n, n) {
        amxp[_m][_p] += amxn[_m][_n] * anxp[_n][_p];
      }
    }
  }
}

template <D_GEMM(typename Type, const int T, const int V, const int I)>
template <const int T_>
void convolution_winograd_kernel<R_GEMM(Type, T, V, I)>::__gemm(
    winograd_template_parameter_t<S_GEMM(float, T_, 16, ISA_GENERIC)>,
    elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
    bool zero_out)
{
  MD(Type, atoutput, [xc.O2][T_][V], toutput);
  MD(Type, atinput, [xc.I2][T_][V], tinput);
  MD(Type, atweights, [xc.O2][xc.I2][V][V], tweights);

#pragma omp parallel for collapse(1)
  for_each (_O2, xc.O2) {
    for_each (_I2, xc.I2) {
      elk_gemm_ker<Type>((Type *)atoutput[_O2], (Type *)atinput[_I2],
          (Type *)atweights[_O2][_I2], T_, V, V, zero_out && _I2 == 0);
    }
  }
}

#define AVX512_DEF(z, n, nil) __m512 t##n;
#define AVX512_ZERO(z, n, nil) t##n = _mm512_setzero_ps();
#define AVX512_LOAD(z, n, nil) t##n = _mm512_load_ps(atoutput[_O2][n]);
#define AVX512_STORE(z, n, nil) _mm512_store_ps(atoutput[_O2][n], t##n);
#define AVX512_FMA(z, n, nil)                                                  \
  x = _mm512_set1_ps(*(x_ptr + n * 16));                                       \
  t##n = _mm512_fmadd_ps(w, x, t##n);

#define DEF_function_gemm(z, n, nil)                                         \
  template <D_GEMM(typename Type, const int T, const int V, const int I)>    \
  void convolution_winograd_kernel<R_GEMM(Type, T, V, I)>::__gemm(           \
      winograd_template_parameter_t<S_GEMM(float, n, 16, ISA_SKX_AVX512)>,   \
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights, \
      bool zero_out)                                                         \
  {                                                                          \
    ENABLE_AVX512F();                                                        \
                                                                             \
    MD(float, atweights, [xc.O2][xc.I2][16][16], tweights);                  \
    MD(float, atinput, [xc.I2][n][16], tinput);                              \
    MD(float, atoutput, [xc.O2][n][16], toutput);                            \
                                                                             \
    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {                                  \
      BOOST_PP_REPEAT(n, AVX512_DEF, nil);                                   \
                                                                             \
      if (zero_out) {                                                        \
        BOOST_PP_REPEAT(n, AVX512_ZERO, nil);                                \
      } else {                                                               \
        BOOST_PP_REPEAT(n, AVX512_LOAD, nil);                                \
      }                                                                      \
                                                                             \
      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {                                \
        for (int _V = 0; _V < 16; ++_V) {                                    \
          __m512 x;                                                          \
          __m512 w = _mm512_load_ps(atweights[_O2][_I2][_V]);                \
          float *x_ptr = &atinput[_I2][0][_V];                               \
          BOOST_PP_REPEAT(n, AVX512_FMA, nil);                               \
        }                                                                    \
      }                                                                      \
      BOOST_PP_REPEAT(n, AVX512_STORE, nil);                                 \
    }                                                                        \
  }

BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, DEF_function_gemm, nil);

#define INST_C_gemm(z, n, data)                                                \
  template void                                                                \
  convolution_winograd_kernel<S_GEMM(float, n, 16, ISA_GENERIC)>::gemm(        \
      elx_conv_t<float> &, float *, float *, float *, bool zero_out);

BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, INST_C_gemm, nil);

#define INST_V_gemm(z, n, nil)                                                 \
  template void                                                                \
  convolution_winograd_kernel<S_GEMM(float, n, 16, ISA_SKX_AVX512)>::gemm(     \
      elx_conv_t<float> &, float *, float *, float *, bool zero_out);

BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, INST_V_gemm, nil);

} // namespace euler
