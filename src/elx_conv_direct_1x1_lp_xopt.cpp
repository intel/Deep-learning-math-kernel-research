#include <string.h>
#include "el_parallel.hpp"
#include "elx_conv_direct_1x1_lp.hpp"


// XOPT
// kernel options:
//   - a: CCC, s1
//   - b: CCD, s1
//   - c: DCD: s1
// fusion:  same as winograd
// dup:     same as winograd
//
// ------+-----+--------+-----+--------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+--------------------------------------
//  c060 |  c  |   t+o  |  -  | blocked/nhwc, Tr, Or, stride=1
// ------+-----+--------+-----+--------------------------------------
//  b061 |  b  |   t+o  |  I  | blocked, stride>=1, ow = wt*T
// ------+-----+--------+-----+--------------------------------------
//

namespace euler {

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::__execute_c160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2, t2*, T(Tr), V
  if (is_first_run_) {
    setup_workspace([&]() {
      if (weights_is_bfmt_ || weights_as_bfmt_)
        trans_weights_s8_blocked_oc(
            weights_scale_, tweights_s8_, weights, bias);
      else
        el_error("Unimplement format");
    });
  }

  INIT_LOOP_ORDER(4);
  CREATE_LOOP_ORDER(4, t3, oc4, ic4, t2);
  CREATE_LOOP_ORDER(4, oc4, ic4, t3, t2);

  auto loop_for = [&](int a0, int a1, int a2, int a3) {
    auto ithr = el_get_thread_num();
    int _t3, _oc4, _ic4, _t2;
    if (CHECK_LOOP_ORDER(4, t3, oc4, ic4, t2)) {
      _t3 = a0; _oc4 = a1; _ic4 = a2; _t2 = a3;
    } else {
      _oc4 = a0; _ic4 = a1; _t3 = a2; _t2 = a3;
    }

    // input
    MD2(uint8_t, ainput_blocked, input,
        this->t3, this->ih * this->iw * this->IC);
    MD2(uint8_t, ainput_nhwc, input,
        this->t3, this->ih * this->iw * this->ic);
    // output
    MD2(OutputType, aoutput_blocked, output,
        this->t3, this->oh * this->ow * this->OC);
    MD2(OutputType, aoutput_nhwc, output,
        this->t3, this->oh * this->ow * this->oc);
    // toutput
    MD2(ToutputType, atoutput_blocked, toutput_,
        this->t3, this->oh * this->ow * this->OC);
    MD2(ToutputType, atoutput_nhwc, toutput_,
        this->t3, this->oh * this->ow * this->oc);
    MD2(ToutputType, atoutput_opt, toutput_,
        mthr_, this->oc3 * this->O2 * V * this->oh * this->ow);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    MD2(TscaleType, ainput_scale, input_scale_, 2, this->T);
    MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
        this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
    MD2(TscaleType, aweights_scale, weights_scale_,
        this->oc4, this->oc3 * 2 * this->O2 * V);

    auto ain = this->input_fmt == nhwc
             ? &md2(ainput_nhwc, _t3, 0) : &md2(ainput_blocked, _t3, 0);
    auto aout = this->output_fmt == nhwc
             ? &md2(aoutput_nhwc, _t3, 0) : &md2(aoutput_blocked, _t3, 0);
    auto atout = toutput_opt_
             ? &md2(atoutput_opt, ithr, 0) : this->output_fmt == nhwc
               ? &md2(atoutput_nhwc, _t3, 0) : &md2(atoutput_blocked, _t3, 0);

    gemm_c160(atout, aout, ain,
        &md3(atweights_s8, _oc4, _ic4, 0),
        &md2(ainput_scale, 0, 0),
        &md2(aweights_scale, _oc4, 0),
        &md2(abias, _oc4, 0),
        _ic4, _oc4, _t2);
  };

  if (this->oh <= 14 && this->ow <= 14) {
    SET_LOOP_ORDER(4, oc4, ic4, t3, t2);
    parallel_for<4, 1>(mthr_, loop_for,
                       this->oc4, this->ic4, this->t3, this->t2);
  } else {
    SET_LOOP_ORDER(4, t3, oc4, ic4, t2);
    parallel_for<4, 2>(mthr_, loop_for,
                       this->t3, this->oc4, this->ic4, this->t2);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::__execute_b161(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2, t2*, T(Tr), V

  if (is_first_run_) {
    setup_workspace([&]() {
      if (weights_is_bfmt_ || weights_as_bfmt_)
        trans_weights_s8_blocked_oc(
            weights_scale_, tweights_s8_, weights, bias);
      else
        el_error("Unimplement format");
    });
  }

  INIT_LOOP_ORDER(5);
  CREATE_LOOP_ORDER(5, t3, oc4, ic4, ht, wt);
  CREATE_LOOP_ORDER(5, oc4, ic4, t3, ht, wt);

  auto loop_for = [&](int a0, int a1, int a2, int a3, int a4) {
    auto ithr = el_get_thread_num();
    int _t3, _oc4, _ic4, _ht, _wt;
    if (CHECK_LOOP_ORDER(5, t3, oc4, ic4, ht, wt)) {
      _t3 = a0, _oc4 = a1, _ic4 = a2, _ht = a3, _wt = a4;
    } else {
      _oc4 = a0, _ic4 = a1, _t3 = a2, _ht = a3, _wt = a4;
    }
    // blocked
    MD7(uint8_t, ainput_blocked, input,
        this->t3, this->ic4, this->ic3 * this->I2,
        this->ht, this->hs, this->wt, this->T * this->ws * V);
    MD2(OutputType, aoutput0_blocked, output,
        this->t3, this->OC * this->oh * this->ow);
    MD5(OutputType, aoutput1_blocked, &md2(aoutput0_blocked, _t3, 0), this->oc4,
        this->oc3 * this->O2, this->ht, this->wt, this->T * V);
    MD6(ToutputType, atoutput_blocked, toutput_,
        this->t3, this->oc4, this->oc3 * this->O2, this->ht, this->wt, this->T * V);
    MD5(ToutputType, atoutput_opt, toutput_,
        mthr_, this->oc3 * this->O2, this->ht, this->wt, this->T * V);
    // nhwc
    MD6(uint8_t, ainput0_nhwc, input,
        this->t3, this->ht, this->hs, this->wt, this->T * this->ws, this->ic);
    MD2(uint8_t, ainput1_nhwc, &md6(ainput0_nhwc, _t3, _ht, 0, _wt, 0, 0),
        this->ic4, this->ic3 * this->I2 * V);
    MD5(OutputType, aoutput0_nhwc, output,
        this->t3, this->ht, this->wt, this->T, this->oc);
    MD2(OutputType, aoutput1_nhwc, &md5(aoutput0_nhwc, _t3, _ht, _wt, 0, 0),
        this->oc4, this->oc3 * this->O2 * V);
    MD5(ToutputType, atoutput0_nhwc, toutput_,
        this->t3, this->ht, this->wt, this->T, this->oc);
    MD2(ToutputType, atoutput1_nhwc, &md5(atoutput0_nhwc, _t3, _ht, _wt, 0, 0),
        this->oc4, this->oc3 * this->O2 * V);

    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
    MD2(TscaleType, ainput_scale, input_scale_, 2, this->T);
    MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
        this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
    MD2(TscaleType, aweights_scale, weights_scale_,
        this->oc4, this->oc3 * 2 * this->O2 * V);

    auto ain = this->input_fmt == nhwc
             ? &md2(ainput1_nhwc, _ic4, 0)
             : &md7(ainput_blocked, _t3, _ic4, 0, _ht, 0, _wt, 0);
    auto aout = this->output_fmt == nhwc
             ? &md2(aoutput1_nhwc, _oc4, 0)
             : &md5(aoutput1_blocked, _oc4, 0, _ht, _wt, 0);
    auto atout = toutput_opt_
             ? &md5(atoutput_opt, ithr, 0, _ht, _wt, 0)
             : this->output_fmt == nhwc
               ? &md2(atoutput1_nhwc, _oc4, 0)
               : &md6(atoutput_blocked, _t3, _oc4, 0, _ht, _wt, 0);
    gemm_b161(atout, aout, ain,
        &md3(atweights_s8, _oc4, _ic4, 0),
        &md2(ainput_scale, 0, 0),
        &md2(aweights_scale, _oc4, 0),
        &md2(abias, _oc4, 0),
        _ic4);
  };

  if (this->oh <= 7 && this->ow <= 7) {
    SET_LOOP_ORDER(5, oc4, ic4, t3, ht, wt);
    parallel_for<5, 1>(mthr_, loop_for,
                       this->oc4, this->ic4, this->t3, this->ht, this->wt);
  } else {
    SET_LOOP_ORDER(5, t3, oc4, ic4, ht, wt);
    parallel_for<5, 2>(mthr_, loop_for,
                       this->t3, this->oc4, this->ic4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_lp_t
void Instance_elx_conv_direct_1x1_lp_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
