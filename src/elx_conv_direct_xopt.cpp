#include <string.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "elx_conv_direct.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a060 |conv |   t+o  |  -  | nhwc|blocked|nchw-input, Ir/Tr/Or, K=3,5,7 S=1
// ------+-----+--------+-----+------------------------------------------------
//  d060 |gemm |   t+o  |  -  | nhwc|blocked, Ir/Tr/Or
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {


Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nchw): t3*, ic4*, ic3, I2, V(Ir), ht*, S, wt*, T, S
  // input (nhwc): t3*, ht*, S, wt*, T, S, ic4*, ic3, I2, V(Ir)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output (blocked):  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  // output (nhwc):  t3*, ht*wt*, T, oc4*, oc3, O2(O2r), V
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
  MD3(TweightsType, atweights3, tweights_, this->ic4, this->oc4,
      this->kh * this->kw * this->ic3 * this->oc3 * this->I2 * this->O2 * V * V);

  if (is_first_run_) {
    trans_weights_to_compact(tweights_, weights);
  }

  if (this->input_fmt == nchw) { // nchw => blocked
    MD2(InputType, ainput2, input, this->t3, this->ic * this->ih * this->iw);
    MD5(OutputType, aoutput, output, this->t3, this->oc4, this->oc3 * this->O2,
        this->ht, this->ow * V);

    iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
    iter_each (_ht, this->ht) {
    iter_each (_wt, this->wt) {
      MD5(InputType, ainput5, &md2(ainput2, _t3, 0), this->ic4,
          this->ic3 * this->I2 * V, this->ht, this->hs, this->iw);
      MD2(InputType, ainput2, &md5(ainput5, _ic4, 0, _ht, 0, 0), this->wt,
          this->T * this->ws);
      MD3(OutputType, aoutput3, &md5(aoutput, _t3, _oc4, 0, _ht, 0), this->wt,
          this->T, V);
      conv_a060(&md3(aoutput3, _wt, 0, 0), &md2(ainput2, _wt, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }}}}}
  } else if (this->input_fmt == nhwc) { // nhwc => nhwc
    MD5(InputType, ainput0, input, this->t3, this->ht, this->hs, this->iw, this->ic);
    MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->oc);

    iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
    iter_each (_ht, this->ht) {
    iter_each (_wt, this->wt) {
      MD4(InputType, ainput1, &md5(ainput0, _t3, _ht, 0, 0, 0), this->wt,
          this->T, this->ws, this->ic);
      MD2(InputType, ainput2, &md4(ainput1, _wt, 0, 0, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD3(OutputType, aoutput1, &md4(aoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);

      conv_a060(&md2(aoutput2, _oc4, 0), &md2(ainput2, _ic4, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }}}}}
  } else { // blocked => blocked
    MD6(InputType, ainput, input, this->t3, this->ic4, this->ic3 * this->I2,
        this->ht, this->hs, this->iw * V);
    MD5(OutputType, aoutput, output, this->t3, this->oc4, this->oc3 * this->O2,
        this->ht, this->ow * V);

    iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
    iter_each (_ht, this->ht) {
    iter_each (_wt, this->wt) {
      MD3(InputType, ainput3, &md6(ainput, _t3, _ic4, 0, _ht, 0, 0), this->wt,
          this->T * this->ws, V);
      MD3(OutputType, aoutput3, &md5(aoutput, _t3, _oc4, 0, _ht, 0), this->wt,
          this->T, V);
      conv_a060(&md3(aoutput3, _wt, 0, 0), &md3(ainput3, _wt, 0, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }}}}}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_d060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ih, iw, V(Ir)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output:  t3*, oc4*, oc3, O2(O2r), ht*wt*, T(Tr), V
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
  MD3(TweightsType, atweights3, tweights_, this->ic4, this->oc4,
      this->kh * this->kw * this->ic3 * this->oc3 * this->I2 * this->O2 * V * V);

  if (is_first_run_) {
    trans_weights_to_compact(tweights_, weights);
  }

  if (this->input_fmt == nhwc) { // nhwc -> nhwc
    MD3(InputType, ainput0, input, this->t3, this->ih * this->iw, this->ic);
    MD3(OutputType, aoutput0, output, this->t3, this->ht * this->ow, this->oc);

    iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
    iter_each (_ht, this->ht) {
    iter_each (_wt, this->wt) {
      MD2(InputType, ainput1, &md3(ainput0, _t3, 0, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD2(OutputType, aoutput1, &md3(aoutput0, _t3, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);
      gemm_d060(&md2(aoutput1, _oc4, 0), &md2(ainput1, _ic4, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }}}}}
  } else { // blocked -> blocked
    MD4(InputType, ainput, input, this->t3, this->ic4, this->ic3 * this->I2,
      this->ih * this->iw * V);
    MD4(OutputType, aoutput, output, this->t3, this->oc4, this->oc3 * this->O2,
      this->ht * this->ow * V);
    iter_each (_ic4, this->ic4) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
#pragma omp for nowait collapse(4)
    iter_each (_t3, this->t3) {
    iter_each (_oc4, this->oc4) {
    iter_each (_ht, this->ht) {
    iter_each (_wt, this->wt) {
      gemm_d060(&md4(aoutput, _t3, _oc4, 0, 0), &md4(ainput, _t3, _ic4, 0, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0),
          _ic4, _oc4, _ht, _wt);
    }}}}}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
