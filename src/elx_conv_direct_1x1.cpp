#include <string.h>
#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_direct_1x1.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {


template <typename Type, const int V, const int I>
elx_conv_direct_1x1_t<Type, V, I>::elx_conv_direct_1x1_t(
    eld_conv_t<Type>& dc)
    : elx_conv_t<Type>(dc)
{
  // TODO: error when V!=16 && fmt=OIhw16i16o

  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  this->V = V;
  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  this->A = A;
  this->ht = (this->oh + A - K) / (A - K + 1);
  this->wt = (this->ow + A - K) / (A - K + 1);
  this->nt = this->ht * this->wt;
  this->t = this->nt * this->n;

  hOA_end_ = this->oh % (A - K + 1) - 1;
  if (hOA_end_ == -1) hOA_end_ = A - K;
  wOA_end_ = this->ow % (A - K + 1) - 1;
  if (wOA_end_ == -1) wOA_end_ = A - K;
  hA_end_ = (this->ih + this->tp) - (this->ht - 1) * (A - K + 1) - 1;
  wA_end_ = (this->iw + this->lp) - (this->wt - 1) * (A - K + 1) - 1;

  // TODO: santize user settings
  if (this->I2 == 0) this->I2 = 4; // TODO: I2 selection
  if (this->O2 == 0) this->O2 = 2; // TODO: O2 selection
  if (this->T == 0)  this->T = 18; // TODO: T selection

  // Tailing
  this->Tr = this->t % this->T ? this->t % this->T : this->T;
  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  if (this->nteams == 0 || this->nthreads == 0
      || this->nteams * this->nthreads > mthr_
      || this->nteams > MAX_THREAD_TEAMS) {
    this->nteams = 1;
    this->nthreads = mthr_;
  } else {
    mthr_ = this->nteams * this->nthreads;
  }
  inference_acc_ = this->prop_kind == forward_inference;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  // further divide packed oc/ic
  this->oc3 = this->oc2 / this->O2;
  this->ic3 = this->ic2 / this->I2;

  this->t2 = (this->t + this->T - 1) / this->T;

  // In case of Ir != V && blocked-format, assume bias also
  // padded to Vx.

  xopt_ = this->execution_mode;
  if (!(xopt_ & XOPT_MSK)) {
    // TODO: deduce xopt
    xopt_ = TTM_O | FUS_T | DUP_I;
  }

  // dbg
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc4=%d, OC=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc4, this->OC);
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::__execute(Type *output, Type *input, Type *weights, Type *bias) {
  // main alogirhtm frame
  // input: n, ic2, hxw, V
  // weights: oc2, oc2, V, V
  // output:  n, oc2, hxw, V
}

template <typename Type, const int V, const int I>
void elx_conv_direct_1x1_t<Type, V, I>::execute(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  if (is_bfmt_)
    return (this->*execute_opt_)(output, input, weights, bias);
  else {
    Type *in = input;
    Type *wei = weights;
    Type *out = output_as_bfmt_ ? boutput_ : output;

    if (input_as_bfmt_) {
      MD5(Type, abinput, binput_, this->n, this->ic2, this->ih, this->iw, V);
      MD4(Type, ainput, input, this->n, this->ic, this->ih, this->iw);

#pragma omp parallel for collapse(3)
      for_each (_n, this->n) {
      for_each (_ic2, this->ic2) {
      for_each (_ih, this->ih) {
        int v = _ic2 == this->ic2 - 1 ? this->Ir : V;
        for_each (_iw, this->iw) {
#pragma omp simd
          for_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }}}
      in = binput_;
    }

    if (weights_as_bfmt_) {
      MD6(Type, abweights, bweights_, this->oc2, this->ic2, this->kh, this->kw, V, V);
      MD4(Type, aweights, weights, this->oc, this->ic, this->kh, this->kw);

#pragma omp parallel for collapse(3)
      for_each (_oc2, this->oc2) {
      for_each (_ic2, this->ic2) {
      for_each (_kh, this->kh) {
        int iv = _ic2 == this->ic2 - 1 ? this->Ir : V;
        int ov = _oc2 == this->oc2 - 1 ? this->Or : V;
        for_each (_kw, this->kw) {
          for_each (_iv, iv) {
#pragma omp simd
            for_each (_ov, ov) {
              md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
                = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
            }
          }
        }
      }}}
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)(out, in, wei, bias);

    if (output_as_bfmt_) {
      MD5(Type, aboutput, boutput_, this->n, this->oc2, this->oh, this->ow, V);
      MD4(Type, aoutput, output, this->n, this->oc, this->oh, this->ow);

#pragma omp parallel for collapse(3)
      for_each (_n, this->n) {
      for_each (_oc2, this->oc2) {
      for_each (_oh, this->oh) {
        int v = _oc2 == this->oc2 - 1 ? this->Or : V;
        for_each (_V, v) {
          for_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }
        }
      }}}
    }
  }
}

} // namespace euler
