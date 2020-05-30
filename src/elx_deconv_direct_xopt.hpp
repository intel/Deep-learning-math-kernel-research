#pragma once

#include "elx_deconv_direct.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a060 |conv |   t+o  |  -  | nhwc|blocked|nchw-input, Ir/Tr/Or, K=3,5,7 S=1
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_deconv_direct_t
void Instance_elx_deconv_direct_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): n*, I4*, I3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nchw): n*, I4*, I3, I2, V(Ir), ht*, S, wt*, T, S
  // input (nhwc): n*, ht*, S, wt*, T, S, I4*, I3, I2, V(Ir)
  // weights: O4*, O3, O2, I4*, I3, I2, V(Ir), V
  // output (blocked):  n*, O4*, O3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  n*, ht*wt*, T, O4*, O3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() { trans_weights_to_compact(tweights_, weights); });
  }

  if (ep.input_fmt == nchw) { // nchw => blocked
    estl::parallel_for<5, 1>([&](int _n, int _I4, int _O4, int _ht, int _wt) {
      int Vr = ep.ic < V ? ep.Ir : V;
      MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
      MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
          V * Vr * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD2(InputType, ainput0, input, ep.n, ep.ic * ep.ih * ep.iw);
      MD3(InputType, ainput1, &md2(ainput0, _n, 0), ep.I4,
          ep.I3 * ep.I2 * V, ep.ih * ep.iw);
      MD5(OutputType, aoutput0, output, ep.n, ep.O4,
          ep.O3 * ep.O2, ep.ht, ep.ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _n, _O4, 0, _ht, 0), ep.wt,
          ep.T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _I4, 0, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    }, ep.n, ep.I4, ep.O4, ep.ht, ep.wt);
  } else if (ep.input_fmt == nhwc) { // nhwc => nhwc
    estl::parallel_for<5, 1>([&](int _n, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
      MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD4(InputType, ainput0, input, ep.n, ep.ih, ep.iw, ep.ic);
      MD2(InputType, ainput1, &md4(ainput0, _n, 0, 0, 0), ep.I4, ep.I3 * ep.I2 * V);
      MD4(OutputType, aoutput0, output, ep.n, ep.ht, ep.ow, ep.oc);
      MD3(OutputType, aoutput1, &md4(aoutput0, _n, _ht, 0, 0), ep.wt,
          ep.T, ep.oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), ep.O4,
          ep.O3 * ep.O2 * V);
      conv_a060(&md2(aoutput2, _O4, 0), &md2(ainput1, _I4, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    },  ep.n, ep.I4, ep.O4, ep.ht, ep.wt);
  } else { // blocked => blocked
    estl::parallel_for<5, 1>([&](int _n, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias, bias, ep.O4, ep.O3 * ep.O2 * V);
      MD3(TweightsType, atweights, tweights_, ep.O4, ep.I4,
          V * V * ep.kh * ep.kw * ep.I3 * ep.O3 * ep.I2
              * ep.O2);
      MD4(InputType, ainput, input, ep.n, ep.I4,
          ep.I3 * ep.I2, ep.ih * ep.iw * V);
      MD5(OutputType, aoutput0, output, ep.n, ep.O4,
          ep.O3 * ep.O2, ep.ht, ep.ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _n, _O4, 0, _ht, 0), ep.wt,
          ep.T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md4(ainput, _n, _I4, 0, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    }, ep.n, ep.I4, ep.O4, ep.ht, ep.wt);
  }

  if (is_first_run_ && inference_acc_)
    is_first_run_ = false;
}

Template_elx_deconv_direct_t
void Instance_elx_deconv_direct_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
