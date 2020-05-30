#pragma once

#include "elx_conv_direct_vmg.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  c060 |conv |   t+o  |  -  | nhwc|blocked, Tr, K=3,5,7 S=1 G=1,2,4,8,16
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::__execute_c060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nhwc): n*, ht*, S, wt*, T, S, I4*, I3, I2, V(Ir)
  // output (blocked):  n*, O4*, O3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  n*, ht*wt*, T, O4*, O3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() { trans_weights_to_compact(tweights_, weights); });
  }

  if (ep.input_fmt == nhwc) { // nhwc => nhwc
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * C * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD5(InputType, ainput0, input, ep.n, ep.ht, ep.hs, ep.iw,
          ep.g * ep.ic);
      MD5(InputType, ainput1, &md5(ainput0, _n, _ht, 0, 0, 0), ep.wt,
          ep.T, ep.ws, ep.g, ep.ic);
      MD2(InputType, ainput2, &md5(ainput1, _wt, 0, 0, _g, 0), ep.I4,
          ep.I3 * ep.I2 * V);
      MD4(OutputType, aoutput0, output, ep.n, ep.ht, ep.ow, ep.g * ep.oc);
      MD4(OutputType, aoutput1, &md4(aoutput0, _n, _ht, 0, 0), ep.wt,
          ep.T, ep.g, ep.oc);
      MD2(OutputType, aoutput2, &md4(aoutput1, _wt, 0, _g, 0), ep.O4,
          ep.O3 * ep.O2 * V);
      conv_c060(&md2(aoutput2, _O4, 0), &md2(ainput2, _I4, 0),
          &md4(atweights, _g, _O4, _I4, 0), &md2(abias1, _O4, 0),
          _I4, _O4, _ht, _wt);
    },  ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  } else { // blocked => blocked
    estl::parallel_for<6, 2>([&](int _n, int _g, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, ep.g, ep.oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), ep.O4, ep.O3 * ep.O2 * V);
      MD4(TweightsType, atweights, tweights_, ep.g, ep.O4, ep.I4,
          V * C * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD7(InputType, ainput0, input, ep.n, ep.g, ep.I4,
          ep.I3 * ep.I2, ep.ht, ep.hs, ep.iw * V);
      MD3(InputType, ainput1, &md7(ainput0, _n, _g, _I4, 0, _ht, 0, 0),
          ep.wt, ep.T * ep.ws, V);
      MD6(OutputType, aoutput0, output, ep.n, ep.g, ep.O4,
          ep.O3 * ep.O2, ep.ht, ep.ow * V);
      MD3(OutputType, aoutput1, &md6(aoutput0, _n, _g, _O4, 0, _ht, 0),
          ep.wt, ep.T, V);
      conv_c060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _wt, 0, 0),
          &md4(atweights, _g, _O4, _I4, 0), &md2(abias1, _O4, 0),
          _I4, _O4, _ht, _wt);
    }, ep.n, ep.g, ep.I4, ep.O4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
