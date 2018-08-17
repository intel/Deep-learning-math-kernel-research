#pragma once
#include "elk_cosim.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
template <typename Type, int V, int A, int K>
class convolution_winograd_kernel_base<Type, ISA_COSIM_AVX512, V, A, K> :
  public cosim_base<Type> {
    using target =
      convolution_winograd_kernel_base<Type, ISA_SKX_AVX512, V, A, K>;
    using cosim =
      convolution_winograd_kernel_base<Type, ISA_GENERIC, V, A, K>;
protected:
  constexpr static int I = ISA_COSIM_AVX512;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end) {
    Type dup_atinput[A][A][V];

    target::template __trans_input<is_border>(xc, atinput, input, hT_start,
        hT_end, wT_start, wT_end);
    cosim::template __trans_input<is_border>(xc, dup_atinput, input, hT_start,
        hT_end, wT_start, wT_end);

    cosim_base<Type>::compare_small(dup_atinput, atinput, A * A * V);
  }

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<float> &xc, Type atinput[A][A][V],
      Type *input, int _wA, int hA_start, int hA_end, int wA_start,
      int wA_end) {
    Type dup_atinput[A][A][V];

    target::template __trans_inputa<is_border>(xc, atinput, input, hA_start,
        hA_end, wA_start, wA_end);
    cosim::template __trans_inputa<is_border>(xc, dup_atinput, input, hA_start,
        hA_end, wA_start, wA_end);

    cosim_base<Type>::compare_small(dup_atinput, atinput, A * A * V);
  }

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<float> &xc, Type *output,
      Type atoutput[A][A][V], Type *bias, int hOA_end, int wOA_end) {
    Type dup_output[A - K + 1][A - K + 1][V] = {0};
    std::memset(atoutput, 0, sizeof(dup_output));

    target::template __trans_output<conditions...>(xc, output, atoutput,
        bias, hOA_end, wOA_end);
    cosim::template __trans_output<conditions...>(xc, dup_output, atoutput,
        bias, hOA_end, wOA_end);

    cosim_base<Type>::compare_small(dup_output, atoutput, (A-K+1)*(A-K+1)*V);
  }

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<float> &xc, float *toutputa,
      float *toutput, int Tz, bool stream_out) {
    Type dup_toutputa [A-K+1][V] = {0};
    std::memset(toutputa, 0, sizeof(dup_toutputa));

    target::template __trans_outputa_th<conditions...>(xc, toutputa, toutput,
        Tz, stream_out);
    cosim::template __trans_outputa_th<conditions...>(xc, dup_toutputa, toutput,
        Tz, stream_out);

    cosim_base<Type>::compare_small(dup_toutputa, toutputa, (A - K + 1) * V);
  }

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<float> &xc, float *output,
      float aoutputa[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
    Type dup_output[A-K+1][A-K+1][V];
    std::memset(output, 0, sizeof(dup_output));

    target::template __trans_outputa_bh<conditions...>(xc, output, aoutputa,
        bias, hOA_end, wOA_end);
    cosim::template __trans_outputa_bh<conditions...>(xc, dup_output, aoutputa,
        bias, hOA_end, wOA_end);

    cosim_base<Type>::compare_small(dup_output, output, (A-K+1)*(A-K+1)*V);
  }

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]) {
    Type dup_atweights[A][A][V][V];

    target::trans_weights(atweights, aweights);
    cosim::trans_weights(dup_atweights, aweights);

    cosim_base<Type>::compare_small(dup_atweights, aweights, A * A * V * V);
  }
};
};
