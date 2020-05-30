#pragma once

#include <string.h>
#include "el_parallel.hpp"
#include "elx_int8_conv_direct_1x1.hpp"


// XOPT
// --------+-----+--------+-----------+--------------------------------------
//         | ker | fusion | transform |             notes
// --------+-----+--------+-----------+--------------------------------------
// a160_s1 | gemm|   t+o  |     -     | blocked/nhwc, Ir, Or, Tr, stride=1
// --------+-----+--------+-----------+--------------------------------------
// a160_s2 | gemm|   t+o  |     -     | blocked/nhwc, Ir, stride>=1, ow = wt*T
// --------+-----+--------+-----------+--------------------------------------
//

namespace euler {

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::__execute_a160_s1(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: O4*, O3, O2, I4*, I3, I2, V, V
  // input:   n*, I4*, I3, I2, t2*, T(Tr), V
  // output:  n*, O4*, O3, O2, t2*, T(Tr), V
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
  CREATE_LOOP_ORDER(4, n, O4, I4, t2);
  CREATE_LOOP_ORDER(4, O4, I4, n, t2);

  auto loop_for = [&](int a0, int a1, int a2, int a3) {
    auto ithr = estl::current_thread_index();
    int _n, _O4, _I4, _t2;
    if (CHECK_LOOP_ORDER(4, n, O4, I4, t2)) {
      _n = a0; _O4 = a1; _I4 = a2; _t2 = a3;
    } else {
      _O4 = a0; _I4 = a1; _n = a2; _t2 = a3;
    }

    // input
    MD2(uint8_t, ainput_blocked, input,
        ep.n, ep.ih * ep.iw * ep.IC);
    MD2(uint8_t, ainput_nhwc, input,
        ep.n, ep.ih * ep.iw * ep.ic);
    // output
    MD2(OutputType, aoutput_blocked, output,
        ep.n, ep.oh * ep.ow * ep.OC);
    MD2(OutputType, aoutput_nhwc, output,
        ep.n, ep.oh * ep.ow * ep.oc);
    // toutput
    MD2(ToutputType, atoutput_blocked, toutput_,
        ep.n, ep.oh * ep.ow * ep.OC);
    MD2(ToutputType, atoutput_nhwc, toutput_,
        ep.n, ep.oh * ep.ow * ep.oc);
    MD2(ToutputType, atoutput_opt, toutput_,
        mthr_, ep.O3 * ep.O2 * V * ep.oh * ep.ow);
    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);

    MD2(float, ainput_scale, input_scale_, 2, ep.T);
    MD3(int8_t, atweights_s8, tweights_s8_, ep.O4, ep.I4,
        ep.O3 * ep.I3 * ep.O2 * ep.I2 * V * V);
    MD2(float, aweights_scale, weights_scale_,
        ep.O4, ep.O3 * 2 * ep.O2 * V);

    auto ain = ep.input_fmt == nhwc
             ? &md2(ainput_nhwc, _n, 0) : &md2(ainput_blocked, _n, 0);
    auto aout = ep.output_fmt == nhwc
             ? &md2(aoutput_nhwc, _n, 0) : &md2(aoutput_blocked, _n, 0);
    auto atout = toutput_opt_
             ? &md2(atoutput_opt, ithr, 0) : ep.output_fmt == nhwc
               ? &md2(atoutput_nhwc, _n, 0) : &md2(atoutput_blocked, _n, 0);

    gemm_a160_s1(atout, aout, ain,
        &md3(atweights_s8, _O4, _I4, 0),
        &md2(ainput_scale, 0, 0),
        &md2(aweights_scale, _O4, 0),
        &md2(abias, _O4, 0),
        _I4, _O4, _t2);
  };

  if (ep.oh <= 14 && ep.ow <= 14) {
    SET_LOOP_ORDER(4, O4, I4, n, t2);
    estl::parallel_for<4, 1>(loop_for, ep.O4, ep.I4, ep.n, ep.t2);
  } else {
    SET_LOOP_ORDER(4, n, O4, I4, t2);
    estl::parallel_for<4, 2>(loop_for, ep.n, ep.O4, ep.I4, ep.t2);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::__execute_a160_s2(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: O4*, O3, O2, I4*, I3, I2, V, V
  // input:   n*, I4*, I3, I2, t2*, T(Tr), V
  // output:  n*, O4*, O3, O2, t2*, T(Tr), V

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
  CREATE_LOOP_ORDER(5, n, O4, I4, ht, wt);
  CREATE_LOOP_ORDER(5, O4, I4, n, ht, wt);

  auto loop_for = [&](int a0, int a1, int a2, int a3, int a4) {
    auto ithr = estl::current_thread_index();
    int _n, _O4, _I4, _ht, _wt;
    if (CHECK_LOOP_ORDER(5, n, O4, I4, ht, wt)) {
      _n = a0, _O4 = a1, _I4 = a2, _ht = a3, _wt = a4;
    } else {
      _O4 = a0, _I4 = a1, _n = a2, _ht = a3, _wt = a4;
    }
    // blocked
    MD7(uint8_t, ainput_blocked, input,
        ep.n, ep.I4, ep.I3 * ep.I2,
        ep.ht, ep.hs, ep.wt, ep.T * ep.ws * V);
    MD2(OutputType, aoutput0_blocked, output,
        ep.n, ep.OC * ep.oh * ep.ow);
    MD5(OutputType, aoutput1_blocked, &md2(aoutput0_blocked, _n, 0), ep.O4,
        ep.O3 * ep.O2, ep.ht, ep.wt, ep.T * V);
    MD6(ToutputType, atoutput_blocked, toutput_,
        ep.n, ep.O4, ep.O3 * ep.O2, ep.ht, ep.wt, ep.T * V);
    MD5(ToutputType, atoutput_opt, toutput_,
        mthr_, ep.O3 * ep.O2, ep.ht, ep.wt, ep.T * V);
    // nhwc
    MD6(uint8_t, ainput0_nhwc, input,
        ep.n, ep.ht, ep.hs, ep.wt, ep.T * ep.ws, ep.ic);
    MD2(uint8_t, ainput1_nhwc, &md6(ainput0_nhwc, _n, _ht, 0, _wt, 0, 0),
        ep.I4, ep.I3 * ep.I2 * V);
    MD5(OutputType, aoutput0_nhwc, output,
        ep.n, ep.ht, ep.wt, ep.T, ep.oc);
    MD2(OutputType, aoutput1_nhwc, &md5(aoutput0_nhwc, _n, _ht, _wt, 0, 0),
        ep.O4, ep.O3 * ep.O2 * V);
    MD5(ToutputType, atoutput0_nhwc, toutput_,
        ep.n, ep.ht, ep.wt, ep.T, ep.oc);
    MD2(ToutputType, atoutput1_nhwc, &md5(atoutput0_nhwc, _n, _ht, _wt, 0, 0),
        ep.O4, ep.O3 * ep.O2 * V);

    MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
    MD2(float, ainput_scale, input_scale_, 2, ep.T);
    MD3(int8_t, atweights_s8, tweights_s8_, ep.O4, ep.I4,
        ep.O3 * ep.I3 * ep.O2 * ep.I2 * V * V);
    MD2(float, aweights_scale, weights_scale_,
        ep.O4, ep.O3 * 2 * ep.O2 * V);

    auto ain = ep.input_fmt == nhwc
             ? &md2(ainput1_nhwc, _I4, 0)
             : &md7(ainput_blocked, _n, _I4, 0, _ht, 0, _wt, 0);
    auto aout = ep.output_fmt == nhwc
             ? &md2(aoutput1_nhwc, _O4, 0)
             : &md5(aoutput1_blocked, _O4, 0, _ht, _wt, 0);
    auto atout = toutput_opt_
             ? &md5(atoutput_opt, ithr, 0, _ht, _wt, 0)
             : ep.output_fmt == nhwc
               ? &md2(atoutput1_nhwc, _O4, 0)
               : &md6(atoutput_blocked, _n, _O4, 0, _ht, _wt, 0);
    gemm_a160_s2(atout, aout, ain,
        &md3(atweights_s8, _O4, _I4, 0),
        &md2(ainput_scale, 0, 0),
        &md2(aweights_scale, _O4, 0),
        &md2(abias, _O4, 0),
        _I4);
  };

  if (ep.oh <= 7 && ep.ow <= 7) {
    SET_LOOP_ORDER(5, O4, I4, n, ht, wt);
    estl::parallel_for<5, 1>(loop_for, ep.O4, ep.I4, ep.n, ep.ht, ep.wt);
  } else {
    SET_LOOP_ORDER(5, n, O4, I4, ht, wt);
    estl::parallel_for<5, 2>(loop_for, ep.n, ep.O4, ep.I4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::__execute_a160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  if (ep.ws == 1)
    __execute_a160_s1(output, input, weights, bias);
  else
    __execute_a160_s2(output, input, weights, bias);
}

Template_elx_int8_conv_direct_1x1_t
void Instance_elx_int8_conv_direct_1x1_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
