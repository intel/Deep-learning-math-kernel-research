#include "elx_conv_direct_vmg.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a060 |conv |   t+o  |  -  | nhwc|blocked, Tr, K=3,5,7 S=1 G=1,2,4,8,16
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nhwc): t3*, ht*, S, wt*, T, S, ic4*, ic3, I2, V(Ir)
  // output (blocked):  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  t3*, ht*wt*, T, oc4*, oc3, O2(O2r), V
  if (is_first_run_) {
    setup_workspace([&]() { trans_weights_to_compact(tweights_, weights); });
  }

  if (this->input_fmt == nhwc) { // nhwc => nhwc
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * C * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD5(InputType, ainput0, input, this->t3, this->ht, this->hs, this->iw,
          this->g * this->ic);
      MD5(InputType, ainput1, &md5(ainput0, _t3, _ht, 0, 0, 0), this->wt,
          this->T, this->ws, this->g, this->ic);
      MD2(InputType, ainput2, &md5(ainput1, _wt, 0, 0, _g, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->g * this->oc);
      MD4(OutputType, aoutput1, &md4(aoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->g, this->oc);
      MD2(OutputType, aoutput2, &md4(aoutput1, _wt, 0, _g, 0), this->oc4,
          this->oc3 * this->O2 * V);
      conv_a060(&md2(aoutput2, _oc4, 0), &md2(ainput2, _ic4, 0),
          &md4(atweights, _g, _oc4, _ic4, 0), &md2(abias1, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    },  this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  } else { // blocked => blocked
    parallel_for<6, 2>(mthr_, [&](int _t3, int _g, int _ic4, int _oc4, int _ht, int _wt) {
      MD2(BiasType, abias0, bias, this->g, this->oc);
      MD2(BiasType, abias1, &md2(abias0, _g, 0), this->oc4, this->oc3 * this->O2 * V);
      MD4(TweightsType, atweights, tweights_, this->g, this->oc4, this->ic4,
          V * C * this->kh * this->kw * this->ic3 * this->oc3 * this->I2
              * this->O2);
      MD7(InputType, ainput0, input, this->t3, this->g, this->ic4,
          this->ic3 * this->I2, this->ht, this->hs, this->iw * V);
      MD3(InputType, ainput1, &md7(ainput0, _t3, _g, _ic4, 0, _ht, 0, 0),
          this->wt, this->T * this->ws, V);
      MD6(OutputType, aoutput0, output, this->t3, this->g, this->oc4,
          this->oc3 * this->O2, this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md6(aoutput0, _t3, _g, _oc4, 0, _ht, 0),
          this->wt, this->T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _wt, 0, 0),
          &md4(atweights, _g, _oc4, _ic4, 0), &md2(abias1, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }, this->t3, this->g, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
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
