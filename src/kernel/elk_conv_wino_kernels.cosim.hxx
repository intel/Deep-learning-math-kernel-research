#pragma once
#include "elk_cosim.hpp"
#include "elk_conv_wino.hpp"

namespace euler {
template <typename Type, int ...configs> 
class convolution_winograd_kernel_base<Type, ISA_COSIM_AVX512, configs...> :
  public cosim_base<Type> {
protected:
    constexpr static int configs_[] {configs...};
    constexpr static int I = ISA_COSIM_AVX512;
    constexpr static int V = configs_[0];
    constexpr static int A = configs_[1];
    constexpr static int K = configs_[2];
    using target =
      convolution_winograd_kernel_base<Type, ISA_SKX_AVX512, V, A, K>;
    using cosim =
      convolution_winograd_kernel_base<Type, ISA_GENERIC, V, A, K>;

  template <bool is_border> static inline
  void __trans_input(elx_conv_t<float> &xc, float atinput[A][A][V],
      float *input, int hT_start, int hT_end, int wT_start,
      int wT_end) {
    alignas(64) Type dup_atinput[A][A][V];

    target::template __trans_input<is_border>(xc, atinput, input, hT_start,
        hT_end, wT_start, wT_end);
    cosim::template __trans_input<is_border>(xc, dup_atinput, input, hT_start,
        hT_end, wT_start, wT_end);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_atinput),
        reinterpret_cast<Type *>(atinput), A * A * V);
  }

  template <bool is_border>
  static void __trans_inputa(elx_conv_t<float> &xc, Type atinput[A][A][V],
      Type *input, int _wA, int hA_start, int hA_end, int wA_start,
      int wA_end) {
    alignas(64) Type dup_atinput[A][A][V];

    target::template __trans_inputa<is_border>(xc, atinput, input, _wA, hA_start,
        hA_end, wA_start, wA_end);
    cosim::template __trans_inputa<is_border>(xc, dup_atinput, input, _wA, hA_start,
        hA_end, wA_start, wA_end);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_atinput),
        reinterpret_cast<Type *>(atinput), A * A * V);
  }

  template <bool ...conditions>
  static inline void __trans_output(elx_conv_t<float> &xc, Type *output,
      Type atoutput[A][A][V], Type *bias, int hOA_end, int wOA_end) {
    auto *dup_output = new Type [xc.oh * xc.ow * V];
    std::memcpy(dup_output, output, sizeof(Type) * xc.oh * xc.ow * V);

    target::template __trans_output<conditions...>(xc, output, atoutput,
        bias, hOA_end, wOA_end);
    cosim::template __trans_output<conditions...>(xc,
        reinterpret_cast<Type *>(dup_output), atoutput, bias, hOA_end, wOA_end);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_output),
        reinterpret_cast<Type *>(output), xc.oh * xc.ow * V);
    delete [] dup_output;
  }

  template <bool ...conditions>
  static inline void __trans_outputa_th(elx_conv_t<float> &xc, float *toutputa,
      float *toutput, int Tz, bool stream_out) {
    alignas(64) Type dup_toutputa [A-K+1][V] = {0};
    std::memset(toutputa, 0, sizeof(dup_toutputa));

    target::template __trans_outputa_th<conditions...>(xc, toutputa, toutput,
        Tz, stream_out);
    cosim::template __trans_outputa_th<conditions...>(xc,
        reinterpret_cast<Type *>(dup_toutputa), toutput, Tz, stream_out);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_toutputa),
        reinterpret_cast<Type *>(toutputa), (A - K + 1) * V);
  }

  template <bool ...conditions>
  static inline void __trans_outputa_bh(elx_conv_t<float> &xc, float *output,
      float aoutputa[A][A - K + 1][V], float *bias, int hOA_end, int wOA_end) {
    auto *dup_output = new Type [xc.oh * xc.ow * V];
    std::memcpy(dup_output, output, sizeof(Type) * xc.oh * xc.ow * V);

    target::template __trans_outputa_bh<conditions...>(xc, output, aoutputa,
        bias, hOA_end, wOA_end);
    cosim::template __trans_outputa_bh<conditions...>(xc,
        reinterpret_cast<Type *>(dup_output), aoutputa, bias, hOA_end, wOA_end);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_output),
        reinterpret_cast<Type *>(output), (A-K+1)*(A-K+1)*V);
    delete [] dup_output;
  }

  static inline void __trans_weights(float atweights[A][A][V][V],
      float aweights[K][K][V][V]) {
    alignas(64) Type dup_atweights[A][A][V][V];

    target::__trans_weights(atweights, aweights);
    cosim::__trans_weights(dup_atweights, aweights);

    cosim_base<Type>::compare_small(reinterpret_cast<Type *>(dup_atweights),
        reinterpret_cast<Type *>(atweights), A * A * V * V);
  }
};
};
