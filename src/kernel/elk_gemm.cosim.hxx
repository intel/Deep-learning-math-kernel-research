#pragma once
#include "elk_cosim.hpp"
#include "elk_conv_wino.hpp"

namespace euler {

template <typename Type, int V, int T>
class gemm_kernel_base<Type, ISA_COSIM_AVX512, V, T> :
  public cosim_base<Type> {
public:
  using target = gemm_kernel_base<Type, ISA_COSIM_AVX512, V, T>;
  using cosim = gemm_kernel_base<Type, ISA_GENERIC, V, T>;

  static void inline __gemm(elx_conv_t<Type> &xc, Type *toutput, Type *tinput
      , Type *tweights, bool zero_out) {
    Type *dup_toutput = new Type [xc.O2 * T * V];
    std::memcpy(dup_toutput, toutput, xc.O2 * T * V * sizeof(Type));

    target::__gemm(xc, toutput, tinput, tweights, zero_out);
    cosim::__gemm(xc, dup_toutput, tinput, tweights, zero_out);
    cosim_base<Type>::compare_small(dup_toutput, toutput, xc.O2 * T * V);

    delete [] dup_toutput;
  }

  static void inline __gemm_tail(elx_conv_t<Type> &xc, Type *toutput,
      Type *tinput, Type *tweights, bool zero_out) {
    Type *dup_toutput = new Type [xc.O2 * T * V];
    std::memcpy(dup_toutput, toutput, xc.O2 * T * V * sizeof(Type));

    target::__gemm_tail(xc, toutput, tinput, tweights, zero_out);
    cosim::__gemm_tail(xc, dup_toutput, tinput, tweights, zero_out);
    cosim_base<Type>::compare_small(dup_toutput, toutput, xc.O2 * T * V);

    delete [] dup_toutput;
  }
};

};
