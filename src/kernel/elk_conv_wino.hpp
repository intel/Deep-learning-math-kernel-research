#pragma once

#include "elx_conv.hpp"

namespace euler {

// A: tile size
// K: kernel size
// V: vector size
// I: ISA
// format: C/D/E/F
// is_border
// with_bias: has bias
// with_relu
// with_ip_sum

template <typename TinputType, typename InputType, int format, bool is_border,
    int I, int A, int V>
struct elk_conv_wino_trans_input {
  static void execute(elx_param_t &ep, TinputType *tinput,
      InputType *input, int hA_start, int hA_end, int wA_start, int wA_end);
};

template <typename ToutputType, typename OutputType, typename BiasType,
    int format, bool is_border, bool with_bias, bool with_relu,
    bool with_ip_sum, int I, int A, int K, int V>
struct elk_conv_wino_trans_output {
  static void execute(elx_param_t &ep, OutputType *output,
      ToutputType *toutput, BiasType *bias, int hOA_end, int wOA_end);
};

template <typename TweightsType, typename WeightsType, int I, int A, int K,
    int V>
struct elk_conv_wino_trans_weights {
  static void execute(
      TweightsType atweights[A][A][V][V], WeightsType aweights[K][K][V][V]);
};

} // namespace euler
