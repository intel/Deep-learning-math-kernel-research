#include <assert.h>
#include <x86intrin.h>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

namespace euler {

template <typename Type>
void elk_gemm_ker(Type *mxp, Type *mxn, Type *nxp, int m, int n, int p,
                  bool zero_out) {
  mdarray<Type, 2> amxn(mxn, m, n);
  mdarray<Type, 2> anxp(nxp, n, p);
  mdarray<Type, 2> amxp(mxp, m, p);

  for_each(_m, m) {
    for_each(_p, p) {
      if (zero_out) amxp(_m, _p) = 0.0f;
      for_each(_n, n) { amxp(_m, _p) += amxn(_m, _n) * anxp(_n, _p); }
    }
  }
}

template <typename Type, const int T, const int V>
void elk_gemm(elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
              bool zero_out) {
  mdarray<Type, 3> atoutput(toutput, xc.O2, T, V);
  mdarray<Type, 3> atinput(tinput, xc.I2, T, V);
  mdarray<Type, 4> atweights(tweights, xc.O2, xc.I2, V, V);

#pragma omp parallel for collapse(1)
  for_each(_O2, xc.O2) {
    for_each(_I2, xc.I2) {
      elk_gemm_ker<Type>(&atoutput(_O2, 0, 0), &atinput(_I2, 0, 0),
                         &atweights(_O2, _I2, 0, 0), T, V, V,
                         zero_out && _I2 == 0);
    }
  }
}

template <>
void elk_gemm<float, 25, 16, ISA_GENERIC>(elx_conv_t<float> &xc, float *toutput,
                                          float *tinput, float *tweights,
                                          bool zero_out) {
  elk_gemm<float, 25, 16>(xc, toutput, tinput, tweights, zero_out);
}

template <>
void elk_gemm<float, 25, 16, ISA_SKX_AVX512>(elx_conv_t<float> &xc,
                                             float *toutput, float *tinput,
                                             float *tweights, bool zero_out) {
  ENABLE_AVX512F();

  mdarray<float, 4> atweights(tweights, xc.O2, xc.I2, 16, 16);
  mdarray<float, 3> atinput(tinput, xc.I2, 25, 16);
  mdarray<float, 3> atoutput(toutput, xc.O2, 25, 16);

  for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
#undef OP
#define OP(x) __m512 t##x
    OP_0_to_24();

    if (zero_out) {
#undef OP
#define OP(x) t##x = _mm512_setzero_ps();
      OP_0_to_24();
    } else {
#undef OP
#define OP(x) t##x = _mm512_load_ps(&atoutput(_O2, x, 0));
      OP_0_to_24();
    }

    for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
      for (int _V = 0; _V < 16; ++_V) {
        __m512 x;
        __m512 w = _mm512_load_ps(&atweights(_O2, _I2, _V, 0));
        float *x_ptr = &atinput(_I2, 0, _V);
#undef OP
#define OP(n)                            \
  x = _mm512_set1_ps(*(x_ptr + n * 16)); \
  t##n = _mm512_fmadd_ps(w, x, t##n);
        OP_0_to_24();
      }
    }
#undef OP
#define OP(n) _mm512_store_ps(&atoutput(_O2, n, 0), t##n);
    OP_0_to_24();
  }
}

// elk_trans_weights
// oc3, ic3, A * A, O2, I2, V, V
// t2, A*A, ic3, I2, T, V
// t2, A*A, oc3, O2, T, V
}  // namespace euler
