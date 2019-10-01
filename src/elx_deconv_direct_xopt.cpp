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

  if (this->input_fmt == nchw) { // nchw => blocked
    estl::parallel_for<5, 1>(mthr_, [&](int _n, int _I4, int _O4, int _ht, int _wt) {
      int Vr = this->ic < V ? this->Ir : V;
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          V * Vr * this->kh * this->kw * this->I3 * this->O3 * this->I2
              * this->O2);
      MD2(InputType, ainput0, input, this->n, this->ic * this->ih * this->iw);
      MD3(InputType, ainput1, &md2(ainput0, _n, 0), this->I4,
          this->I3 * this->I2 * V, this->ih * this->iw);
      MD5(OutputType, aoutput0, output, this->n, this->O4,
          this->O3 * this->O2, this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _n, _O4, 0, _ht, 0), this->wt,
          this->T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md3(ainput1, _I4, 0, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    }, this->n, this->I4, this->O4, this->ht, this->wt);
  } else if (this->input_fmt == nhwc) { // nhwc => nhwc
    estl::parallel_for<5, 1>(mthr_, [&](int _n, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          V * V * this->kh * this->kw * this->I3 * this->O3 * this->I2
              * this->O2);
      MD4(InputType, ainput0, input, this->n, this->ih, this->iw, this->ic);
      MD2(InputType, ainput1, &md4(ainput0, _n, 0, 0, 0), this->I4, this->I3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->n, this->ht, this->ow, this->oc);
      MD3(OutputType, aoutput1, &md4(aoutput0, _n, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), this->O4,
          this->O3 * this->O2 * V);
      conv_a060(&md2(aoutput2, _O4, 0), &md2(ainput1, _I4, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    },  this->n, this->I4, this->O4, this->ht, this->wt);
  } else { // blocked => blocked
    estl::parallel_for<5, 1>(mthr_, [&](int _n, int _I4, int _O4, int _ht, int _wt) {
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          V * V * this->kh * this->kw * this->I3 * this->O3 * this->I2
              * this->O2);
      MD4(InputType, ainput, input, this->n, this->I4,
          this->I3 * this->I2, this->ih * this->iw * V);
      MD5(OutputType, aoutput0, output, this->n, this->O4,
          this->O3 * this->O2, this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _n, _O4, 0, _ht, 0), this->wt,
          this->T, V);
      conv_a060(&md3(aoutput1, _wt, 0, 0), &md4(ainput, _n, _I4, 0, 0),
          &md3(atweights, _O4, _I4, 0), &md2(abias, _O4, 0), _I4, _O4, _ht,
          _wt);
    }, this->n, this->I4, this->O4, this->ht, this->wt);
  }

  if (inference_acc_)
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
