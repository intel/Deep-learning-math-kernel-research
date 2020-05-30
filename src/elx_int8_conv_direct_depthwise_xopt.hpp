#pragma once

#include "elx_int8_conv_direct_depthwise.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  c160 |conv |   t+o  |  -  | blocked/nhwc, Tr, K=3 S=1,2
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::__execute_c160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_3x3(
          weights_scale_, weights_shift_, tweights_s8_, weights, bias);
      if (ep.sampling_kind == CALIBRATED) {
        MD2(float, atinput_scale, input_scale_, 2, ep.T);
        iter_each(_T, ep.T) {
          md2(atinput_scale, 0, _T) = ep.input_quant_S;
          md2(atinput_scale, 1, _T) = ep.input_quant_z;
        }
      }
    });
  }

  estl::parallel_for<4>([&](int _n, int _G3, int _ht, int _wt) {
    MD3(int8_t, atweights_s8, tweights_s8_, ep.G3, ep.G2, ep.kh * KW * V);
    MD3(BiasType, abias, bias, ep.G3, ep.G2, V);
    MD3(float, atweights_scale, weights_scale_, ep.G3, ep.G2, V);
    MD3(float, aweights_shift, weights_shift_, ep.G3, ep.G2, V);
    // blocked input
    MD4(InputType, ainput_blocked, input, ep.n, ep.G3, ep.G2, ep.ih * ep.iw * V);
    // blocked output
    MD5(OutputType, aoutput0_blocked, output,
        ep.n, ep.G3, ep.G2, ep.ht, ep.ow * V);
    MD3(OutputType, aoutput1_blocked,
        &md5(aoutput0_blocked, _n, _G3, 0, _ht, 0), ep.wt, ep.T, V);
    // blocked toutput
    MD5(ToutputType, atoutput0_blocked, toutput_,
        ep.n, ep.G3, ep.G2, ep.ht, ep.ow * V);
    MD3(ToutputType, atoutput1_blocked,
        &md5(atoutput0_blocked, _n, _G3, 0, _ht, 0), ep.wt, ep.T, V);

    auto ainput = &md4(ainput_blocked, _n, _G3, 0, 0);
    auto aoutput = &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = &md3(atoutput1_blocked, _wt, 0, 0);
    conv_c160(aoutput, atoutput, ainput,
              &md3(atweights_s8, _G3, 0, 0),
              &md3(abias, _G3, 0, 0), input_scale_,
              &md3(atweights_scale, _G3, 0, 0),
              &md3(aweights_shift, _G3, 0, 0), _ht, _wt);
  }, ep.n, ep.G3, ep.ht, ep.wt);

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
