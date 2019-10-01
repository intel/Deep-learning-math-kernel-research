#include "elx_int8_conv_direct_depthwise.hpp"
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

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::__execute_a160(
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

  estl::parallel_for<4>(mthr_, [&](int _n, int _G3, int _ht, int _wt) {
    MD3(int8_t, atweights_s8, tweights_s8_,
        this->G3, this->G2, this->kh * this->KW * V);
    MD3(BiasType, abias, bias, this->G3, this->G2, V);
    MD3(TscaleType, atweights_scale, weights_scale_, this->G3, this->G2, V);
    MD3(TscaleType, aweights_factor, weights_factor_, this->G3, this->G2, V);
    // blocked input
    MD4(InputType, ainput_blocked, input,
        this->n, this->G3, this->G2, this->ih * this->iw * V);
    // blocked output
    MD5(OutputType, aoutput0_blocked, output,
        this->n, this->G3, this->G2, this->ht, this->ow * V);
    MD3(OutputType, aoutput1_blocked,
        &md5(aoutput0_blocked, _n, _G3, 0, _ht, 0), this->wt, this->T, V);
    // blocked toutput
    MD5(ToutputType, atoutput0_blocked, toutput_,
        this->n, this->G3, this->G2, this->ht, this->ow * V);
    MD3(ToutputType, atoutput1_blocked,
        &md5(atoutput0_blocked, _n, _G3, 0, _ht, 0), this->wt, this->T, V);

    auto ainput = &md4(ainput_blocked, _n, _G3, 0, 0);
    auto aoutput = &md3(aoutput1_blocked, _wt, 0, 0);
    auto atoutput = &md3(atoutput1_blocked, _wt, 0, 0);
    conv_a160(aoutput, atoutput, ainput,
              &md3(atweights_s8, _G3, 0, 0),
              &md3(abias, _G3, 0, 0), input_scale_,
              &md3(atweights_scale, _G3, 0, 0),
              &md3(aweights_factor, _G3, 0, 0), _ht, _wt);
  }, this->n, this->G3, this->ht, this->wt);

  if (inference_acc_)
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
