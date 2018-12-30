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
// ------+-----+--------+-----+--------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+--------------------------------------
//  a060 |  d  |   t+o  |  -  | blocked|nchw, blocked, stride=1, Ir
// ------+-----+--------+-----+--------------------------------------
//  d060 |  d  |   t+o  |  -  | blocked|nchw, blocked, stride=1, Ir, Tr
// ------+-----+--------+-----+--------------------------------------
//
namespace euler {


Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_a060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(Ir)
  // input (nchw): t3*, ic4*, ic3, I2, V(Ir), ht*, S, wt*, T, S
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output:  t3*, oc4*, oc3, O2(O2r), ht*wt*, T, V
  MD5(OutputType, aoutput, output, this->t3, this->oc4, this->oc3 * this->O2,
      this->ht, this->ow * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
  MD3(TarrayType, atweights3, tweights_, this->ic4, this->oc4,
      this->kh * this->kw * this->ic3 * this->oc3 * this->I2 * this->O2 * V * V);

  if (is_first_run_) {
    trans_weights_blocked_to_compact(tweights_, weights);
  }

  if (this->input_fmt == nchw) {
    MD2(InputType, ainput2, input, this->t3, this->ic * this->ih * this->iw);

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
  } else {
    MD6(InputType, ainput, input, this->t3, this->ic4, this->ic3 * this->I2,
        this->ht, this->hs, this->iw * V);

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

// nChw16c (padded) | nchw (compact) + OIhw16i16o -> nChw16c, Ir,Tr
// kh|kw are odd
// ih = oh, lp = rp = (kh-1)/2, hs=1
// iw = iw, tp = bp = (kw-1)/2
Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__execute_d060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T(Tr), S, V(Ir)
  // input (nchw): t3*, ic4*, ic3, I2, V(Ir), ht*, S, wt*, T(Tr), S
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V(Ir), V
  // output:  t3*, oc4*, oc3, O2(O2r), ht*wt*, T(Tr), V
  MD5(OutputType, aoutput, output, this->t3, this->oc4, this->oc3 * this->O2,
      this->ht, this->ow * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
  MD3(TarrayType, atweights3, tweights_, this->ic4, this->oc4,
      this->kh * this->kw * this->ic3 * this->oc3 * this->I2 * this->O2 * V * V);

  if (is_first_run_) {
    trans_weights_blocked_to_compact(tweights_, weights);
  }

  if (this->input_fmt == nchw) {
    MD2(InputType, ainput2, input, this->t3, this->ic * this->ih * this->iw);

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
      gemm_d060_nchw_input(&md3(aoutput3, _wt, 0, 0), &md2(ainput2, _wt, 0),
          &md3(atweights3, _ic4, _oc4, 0), &md2(abias, _oc4, 0), _ic4, _oc4,
          _ht, _wt);
    }}}}}
  } else {
    MD6(InputType, ainput, input, this->t3, this->ic4, this->ic3 * this->I2,
        this->ht, this->hs, this->iw * V);

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
      gemm_d060_blocked_input(&md3(aoutput3, _wt, 0, 0),
          &md3(ainput3, _wt, 0, 0), &md3(atweights3, _ic4, _oc4, 0),
          &md2(abias, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }}}}}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::execute(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  (this->*execute_opt_)(output, input, weights, bias);
}

} // namespace euler
