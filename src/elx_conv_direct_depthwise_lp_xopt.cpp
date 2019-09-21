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
    setup_workspace([&]() {
      trans_weights_3x3(
          weights_scale_, weights_factor_, tweights_s8_, weights, bias);
      if (this->sampling_kind == CALIBRATED) {
        MD2(TscaleType, atinput_scale, input_scale_, 2, this->T);
        iter_each(_T, this->T) {
          md2(atinput_scale, 0, _T) = this->input_quant_S;
          md2(atinput_scale, 1, _T) = this->input_quant_z;
        }
      }
    });
  }

  parallel_for<4>(mthr_, [&](int _t3, int _g3, int _ht, int _wt) {
    MD3(int8_t, atweights_s8, tweights_s8_,
        this->g3, this->g2, this->kh * this->KW * V);
    MD3(BiasType, abias, bias, this->g3, this->g2, V);
    MD3(TscaleType, atweights_scale, weights_scale_, this->g3, this->g2, V);
    MD3(TscaleType, aweights_factor, weights_factor_, this->g3, this->g2, V);
    // blocked input
    MD4(InputType, ainput_blocked, input,
        this->t3, this->g3, this->g2, this->ih * this->iw * V);
    // blocked output
    MD5(OutputType, aoutput0_blocked, output,
        this->t3, this->g3, this->g2, this->ht, this->ow * V);
    MD3(OutputType, aoutput1_blocked,
        &md5(aoutput0_blocked, _t3, _g3, 0, _ht, 0), this->wt, this->T, V);
    // blocked toutput
    MD5(ToutputType, atoutput0_blocked, toutput_,
        this->t3, this->g3, this->g2, this->ht, this->ow * V);
    MD3(ToutputType, atoutput1_blocked,
        &md5(atoutput0_blocked, _t3, _g3, 0, _ht, 0), this->wt, this->T, V);

    auto ainput = &md4(ainput_blocked, _t3, _g3, 0, 0);
    auto aoutput = &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = &md3(atoutput1_blocked, _wt, 0, 0);
    conv_a160(aoutput, atoutput, ainput,
              &md3(atweights_s8, _g3, 0, 0),
              &md3(abias, _g3, 0, 0), input_scale_,
              &md3(atweights_scale, _g3, 0, 0),
              &md3(aweights_factor, _g3, 0, 0), _ht, _wt);
  }, this->t3, this->g3, this->ht, this->wt);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
