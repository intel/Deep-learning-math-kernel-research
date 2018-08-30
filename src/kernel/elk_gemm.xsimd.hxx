#pragma once
#include <assert.h>
#include <xintrin.hpp>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

// #ifndef INCLUDE_WINOGRAD_CONVOLUTION_KERNEL
// #error "Don't include this file directly"
// #endif

// blocking -
// oc3, ic3, A * A, O2, I2, V, V
// t2, A*A, ic3, I2, T, V
// t2, A*A, oc3, O2, T, V

namespace euler {

template <typename Type>
static inline void elk_gemm_ker(
    Type* mxp, Type* mxn, Type* nxp, int m, int n, int p, bool zero_out) {
  MD2(Type, amxn, mxn, m, n);
  MD2(Type, anxp, nxp, n, p);
  MD2(Type, amxp, mxp, m, p);

  iter_each (_m, m) {
    iter_each (_p, p) {
      if (zero_out)
        md2(amxp, _m, _p) = 0.0f;
      iter_each (_n, n) {
        md2(amxp, _m, _p) += md2(amxn, _m, _n) * md2(anxp, _n, _p);
      }
    }
  }
}

template <typename Type, int ...configs>
void inline gemm_kernel_base<Type, configs...>::__gemm(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out) {
  MD3(Type, atoutput, toutput, xc.O2, T, V);
  MD3(Type, atinput, tinput, xc.I2, T, V);
  MD4(Type, atweights, tweights, xc.O2, xc.I2, V, V);

#pragma omp parallel for collapse(1)
  iter_each (_O2, xc.O2) {
    iter_each (_I2, xc.I2) {
      elk_gemm_ker<Type>(&md3(atoutput, _O2, 0, 0), &md3(atinput, _I2, 0, 0),
          &md4(atweights, _O2, _I2, 0, 0), T, V, V, zero_out && _I2 == 0);
    }
  }
}

template <typename Type, int ...configs>
void inline gemm_kernel_base<Type, configs...>::__gemm_tail(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out) {
  MD3(Type, atoutput, toutput, xc.O2, T, V);
  MD3(Type, atinput, tinput, xc.I2, T, V);
  MD4(Type, atweights, tweights, xc.O2, xc.I2, V, V);

#pragma omp parallel for collapse(1)
  iter_each (_O2, xc.O2) {
    iter_each (_I2, xc.I2 - 1) {
      elk_gemm_ker<Type>(&md3(atoutput, _O2, 0, 0), &md3(atinput, _I2, 0, 0),
          &md4(atweights, _O2, _I2, 0, 0), T, V, V, zero_out && _I2 == 0);
    }
    elk_gemm_ker<Type>(&md3(atoutput, _O2, 0, 0),
        &md3(atinput, xc.I2 - 1, 0, 0), &md4(atweights, _O2, xc.I2 - 1, 0, 0),
        T, xc.Ir, V, zero_out && xc.I2 == 1);
  }
}

template <int T, int V>
class gemm_kernel_base<float, ISA_SKX_AVX512, V, T> {
  template <typename Type, int ...configs>
    friend class gemm_kernel_base;
public:
  static void inline __gemm(
    elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
    bool zero_out) {
    if (1 <= T && T <= 28)
      __gemm_impl_1_28(xc, toutput, tinput, tweights, zero_out);
    else if (29 <= T && T <= 30)
      __gemm_impl_29_30(xc, toutput, tinput, tweights, zero_out);
    else /*if (31 <= T)*/
      __gemm_impl_31(xc, toutput, tinput, tweights, zero_out);
    /* else throw ??? */
  }

  static void inline __gemm_tail(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    MD4(float, atweights, tweights, xc.O2, xc.I2, V, V);
    MD3(float, atinput, tinput, xc.I2, T, V);
    MD3(float, atoutput, toutput, xc.O2, T, V);

    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
      __m<V> t[T];
 
      if (zero_out) {
#       pragma unroll (T)
        for (int i =0; i < T; i++) {
          t[i] = _mm<V>::setzero_ps();
        }
      } else {
#       pragma unroll (T)
        for (int i =0; i < T; i++) {
          t[i] = _mm<V>::load_ps(&md3(atoutput, _O2, i, 0));
        }
      }

      for (int _I2 = 0; _I2 < xc.I2 -1; ++_I2) {
#       pragma unroll (V)
        for (int _V= 0; _V < V; ++_V) {
          auto w = _mm<V>::load_ps(&md4(atweights, _O2, _I2, _V, 0));
#         pragma unroll (T)
          for (int i=0; i < T; i++) {
            auto x = _mm<V>::set1_ps(md3(atinput, _I2, i, _V));
            t[i] = _mm<V>::fmadd_ps(w, x, t[i]);
          }
        }
      }

      for (int _v = 0; _v < xc.Ir; ++_v) {
        auto w = _mm<V>::load_ps(&md4(atweights, _O2, xc.I2 - 1, _v, 0));
#       pragma unroll (T)
        for (int i = 0; i < T; i ++) {
          auto x = _mm<V>::set1_ps(md3(atinput, xc.I2 - 1, i, _v));
          t[i] = _mm<V>::fmadd_ps(w, x, t[i]);
        }
      }
#     pragma unroll (T)
      for (int i =0; i < T; i ++) {
        _mm<V>::store_ps(&md3(atoutput, _O2, i, 0), t[i]);
      }
    }
  }

private:
  static inline void __gemm_impl_1_28(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    MD4(float, atweights, tweights, xc.O2, xc.I2, V, V);
    MD4(float, atinput, tinput, xc.I2, T, 4, 4);
    MD3(float, atoutput, toutput, xc.O2, T, V);

    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
      __m<V> t[T];
      float *w_ptr = &md4(atweights, _O2, 0, 0, 0);
      __m512 w0, w1, w2, w3;
      w0 = _mm<V>::load_ps(w_ptr);
      w_ptr += V;
      w1 = _mm<V>::load_ps(w_ptr);
      w_ptr += V;
      asm volatile("" : : : "memory");

      if (zero_out) {
#       pragma unroll (T)
        for(int i =0; i < T; i++)
          t[i] = _mm<V>::setzero_ps();
      } else {
#       pragma unroll (T)
        for(int i =0; i < T; i++)
          t[i] = _mm<V>::load_ps(&md3(atoutput, _O2, i, 0));
      }

      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#       pragma unroll (4)
        for (int _V = 0; _V < V/4; ++_V) {
          w2 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i = 0; i < T; i++) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 0));
            t[i] = _mm<V>::fmadd_ps(w0, x, t[i]);
          }
          asm volatile("" : : : "memory");

          w3 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i =0; i < T; i++) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 1));
            t[i] = _mm<V>::fmadd_ps(w1, x, t[i]);
          }
          asm volatile("" : : : "memory");

          w0 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i =0; i < T; i ++) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 2));
            t[i] = _mm<V>::fmadd_ps(w2, x, t[i]);
          }
          asm volatile("" : : : "memory");

          w1 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i =0; i < T; i ++) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 3));
            t[i] = _mm<V>::fmadd_ps(w3, x, t[i]);
          }
        }
      }
#     pragma unroll (T)
      for (int i = 0; i < T; i ++) {
        _mm<V>::store_ps(&md3(atoutput, _O2, i, 0), t[i]);
      }
    }
  }

  static void inline __gemm_impl_29_30(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    MD4(float, atweights, tweights, xc.O2, xc.I2, V, V);
    MD4(float, atinput, tinput, xc.I2, T, 8, 2);
    MD3(float, atoutput, toutput, xc.O2, T, V);

    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
      __m<V> t[T];
      auto *w_ptr = &md4(atweights, _O2, 0, 0, 0);
      __m<V> w0, w1;
      w0 = _mm<V>::load_ps(w_ptr);
      w_ptr += V;

      asm volatile("" : : : "memory");

      if (zero_out) {
#       pragma unroll (T)
        for(int i =0; i < T; i++)
          t[i] = _mm<V>::setzero_ps();
      } else {
#       pragma unroll (T)
        for(int i =0; i < T; i++)
          t[i] = _mm<V>::load_ps(&md3(atoutput, _O2, i, 0));
      }

      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#       pragma unroll (8)
        for (int _V = 0; _V < V/4; ++_V) {
          w1 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i =0; i < T; ++i) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 0));
            t[i] = _mm<V>::fmadd_ps(w0, x, t[i]);
          }

          asm volatile("" : : : "memory");
          w0 = _mm<V>::load_ps(w_ptr);
          w_ptr += V;
#         pragma unroll (T)
          for (int i =0; i < T; ++i) {
            auto x = _mm<V>::set1_ps(md4(atinput, _I2, i, _V, 1));
            t[i] = _mm<V>::fmadd_ps(w1, x, t[i]);
          }
        }
      }
#     pragma unroll (T)
      for (int i =0; i < T; i++) {
        _mm<V>::store_ps(&md3(atoutput, _O2, i, 0), t[i]);
      }
    }
  }

  static void inline __gemm_impl_31(
      elx_conv_t<float> &xc, float *toutput, float *tinput, float *tweights,
      bool zero_out) {
    MD4(float, atweights, tweights, xc.O2, xc.I2, V, V);
    MD3(float, atinput, tinput, xc.I2, T, V);
    MD3(float, atoutput, toutput, xc.O2, T, V);

    for (int _O2 = 0; _O2 < xc.O2; ++ _O2) {
      __m512 t[T];

      if (zero_out) {
#       pragma unroll (T)
        for (int i =0; i<T; i++) {
          t[i] = _mm<V>::setzero_ps();
        }
      } else {
#       pragma unroll (T)
        for (int i =0; i<T; i++) {
          t[i] = _mm<V>::load_ps(&md3(atoutput, _O2, i, 0));
        }
      }

      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#       pragma unroll (V)
        for (int _V = 0; _V < V; ++_V) {
          auto w = _mm<V>::load_ps(&md4(atweights, _O2, _I2, _V, 0));
#         pragma unroll (T)
          for (int i =0; i < T; ++i) {
            auto x = _mm<V>::set1_ps(md3(atinput, _I2, i, _V));
            t[i] = _mm<V>::fmadd_ps(w, x, t[i]);
          }
        }
      }
#     pragma unroll (T)
      for (int i=0; i < T; i++) {
        _mm<V>::store_ps(&md3(atoutput, _O2, i, 0), t[i]);
      }
    }
  }
};

} // namespace euler
