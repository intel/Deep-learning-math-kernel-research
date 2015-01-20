#include "elx_conv_direct_depthwise_lp.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a160 |conv |   t+o  |  -  | blocked/nhwc, Tr, K=3 S=1,2
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::__execute_a160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  if (is_first_run_) {
    trans_weights_3x3(weights_scale_, weights_factor_, tweights_s8_, weights, bias);
    if (this->sampling_kind == CALIBRATED) {
      MD2(TscaleType, atinput_scale, input_scale_, 2, this->T);
      iter_each(_T, this->T) {
        md2(atinput_scale, 0, _T) = this->input_quant_S;
        md2(atinput_scale, 1, _T) = this->input_quant_z;
      }
    }
  }

  parallel_for<3>(mthr_, [&](int _t3, int _ht, int _wt) {
    MD2(int8_t, atweights_s8, tweights_s8_, this->g2, this->kh * this->KW * V);
    MD2(BiasType, abias, bias, this->g2, V);
    MD2(TscaleType, atweights_scale, weights_scale_, this->g2, V);
    MD2(TscaleType, aweights_factor, weights_factor_, this->g2, V);
    // blocked input
    MD3(InputType, ainput_blocked, input,
        this->t3, this->g2, this->ih * this->iw * V);
    // blocked output
    MD4(OutputType, aoutput0_blocked, output,
        this->t3, this->g2, this->ht, this->ow * V);
    MD3(OutputType, aoutput1_blocked, &md4(aoutput0_blocked, _t3, 0, _ht, 0),
        this->wt, this->T, V);
    // blocked toutput
    MD4(ToutputType, atoutput0_blocked, toutput_,
        this->t3, this->g2, this->ht, this->ow * V);
    MD3(ToutputType, atoutput1_blocked, &md4(atoutput0_blocked, _t3, 0, _ht, 0),
        this->wt, this->T, V);

    auto ainput = &md3(ainput_blocked, _t3, 0, 0);
    auto aoutput = &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = &md3(atoutput1_blocked, _wt, 0, 0);
    conv_a160(aoutput, atoutput, ainput,
              &md2(atweights_s8, 0, 0),
              &md2(abias, 0, 0), input_scale_, &md2(atweights_scale, 0, 0),
              &md2(aweights_factor, 0, 0), _ht, _wt);
  }, this->t3, this->ht, this->wt);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  set_trans_buffers();

  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
