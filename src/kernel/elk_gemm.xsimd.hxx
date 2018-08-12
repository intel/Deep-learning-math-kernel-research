#pragma once
#include <assert.h>
#include <x86intrin.h>
#include <xsimd/xsimd.hpp>
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv_wino.hpp"

using namespace euler;
template <typename Type, int ...configs>
class gemm_kernel_algo {
public:
  constexpr static int T = gemm_traits<configs...>::T;
  constexpr static int V = gemm_traits<configs...>::V;
  using bundle = xsimd::batch<Type, V>;

  static inline bundle __bfma(bundle x, Type y, bundle z) {
    bundle _y{y};
    return xsimd::fma(x, _y, z);
  }
  static inline bundle __bfma(Type x, bundle y, bundle z) {
    bundle _x{x};
    return xsimd::fma(_x, y, z);
  }
  static inline bundle __bfma(bundle x, bundle y, Type z) {
    bundle _z{z};
    return xsimd::fma(x, y, _z);
  }

  template <int G/*, std::enable_if<G%2 == 0 && V%G == 0> */>
  static void inline __gemm_impl(elx_conv_t<Type> &xc, Type *toutput,
      Type *tinput, Type *tweights, bool zero_out) {
    MD4(Type, atweights, tweights, xc.O2, xc.I2, V, V);
    MD4(Type, atinput, tinput, xc.I2, T, 4, 4);
    MD3(Type, atoutput, toutput, xc.O2, T, V);

    for (int _O2 = 0; _O2 < xc.O2; ++_O2) {
      auto *t = new (&md3(atoutput, _O2, 0, 0)) bundle [T];
      auto *w_ptr = &md4(atweights, _O2, 0, 0, 0);
      bundle w[G];
      constexpr int p = G/2;

#     pragma unroll (p)
      for (int i = 0; i < p; i ++) {
        w[i].load_aligned(w_ptr);
        w_ptr += V;
      }

      if (zero_out) {
#       pragma unroll (T)
        for(int i =0; i < T; i ++)
          t[i] ^= t[i];
      }

      for (int _I2 = 0; _I2 < xc.I2; ++_I2) {
#       pragma unroll (V/G)
        for (int _V = 0; _V < V/G; ++ _V) {
#         pragma unroll (G)
          for (int j = 0; j < G; j ++) {
            auto p_idx = (p + j) % G;
            w[p_idx].load_aligned(w_ptr);
            w_ptr += V;
#           pragma unroll (T)
            for (int i = 0; i < T; i ++)
              t[i] = __bfma(w[p_idx], md4(atinput, _I2, i, _V, j), t[i]);
          }
        }
      }

#     pragma unroll (T)
      for (int i = 0; i < T; ++i)
        t[i].store_aligned(&md3(atoutput, _O2, i, 0));
    }
  }
};

template <typename Type, int ...configs>
void inline gemm_kernel_base<Type, configs...>::__gemm(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out) {
  if (1 <= T && T <= 28)
    gemm_kernel_algo<Type, configs...>::template __gemm_impl<4>(
        xc, toutput, tinput, tweights, zero_out);
  else if (29 <= T && T <=30)
    gemm_kernel_algo<Type, configs...>::template __gemm_impl<2>(
        xc, toutput, tinput, tweights, zero_out);
  else
    gemm_kernel_algo<Type, configs...>::template __gemm_impl<1>(
        xc, toutput, tinput, tweights, zero_out);
}

template <typename Type, int ...configs>
void inline gemm_kernel_base<Type, configs...>::__gemm_tail(
    elx_conv_t<Type> &xc, Type *toutput, Type *tinput, Type *tweights,
    bool zero_out) {
  MD4(float, atweights, tweights, xc.O2, xc.I2, 16, 16);
  MD3(float, atinput, tinput, xc.I2, T, 16);
  MD3(float, atoutput, toutput, xc.O2, T, 16);
  using bundle = xsimd::batch<Type, V>;

  for (int _O2 = 0; _O2 < xc.O2; ++ _O2) {
    bundle t[T];

    if (zero_out) {
#     pragma unroll
      for (int i = 0; i < T; i ++)
        t[i] ^= t[i];
    } else { 
#     pragma unroll
      for (int i = 0; i < T; i ++)
        t[i].load_aligned(&md3(atoutput, _O2, i, 0));
    }

    for (int _I2 = 0; _I2 < xc.I2 -1; ++_I2) {
#     pragma unroll
      for (int _V=0; _V < V; ++_V) {
        auto w = xsimd::load_aligned(&md4(atweights, _O2, _I2, _V, 0));
#       pragma unroll
        for (int i = 0; i < V; i ++)
          t[i] = gemm_kernel_algo<Type, configs...>::__bfma(w, md3(atinput, xc.I2 -1, i, _V), t[i]);
      }
    }
#   pragma unroll
    for (int i = 0; i < T; i ++)
      t[i].store_aligned(&md3(atoutput, _O2, i, 0));
  }
}
