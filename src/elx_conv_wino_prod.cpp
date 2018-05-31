#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"
#include "elx_conv_wino_prod.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_prod_t<Type, A, K, V, I>::elx_conv_wino_prod_t(
    eld_conv_t<Type> &dc)
    : elx_conv_t<Type>(dc)
{
  this->V = V;
  this->ic2 = this->ic / V;
  this->oc2 = this->oc / V;

  this->A = A;
  this->ht = (this->oh + A - 3) / (A - K + 1);
  this->wt = (this->ow + A - 3) / (A - K + 1);

  size_t tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
  size_t tinput_size
      = sizeof(Type) * A * A * this->ht * this->wt * this->ic * this->n;
  tweights_ = (Type *)memalign(64, tweights_size);
  tinput_ = (Type *)memalign(64, tinput_size);
  toutput_ = nullptr;

  is_first_run_ = true;
  inference_acc_ = this->prop_kind == forward_inference ? true : false;
}

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_prod_t<Type, A, K, V, I>::~elx_conv_wino_prod_t()
{
  if (tweights_ != nullptr) {
    free(tweights_);
    tweights_ = nullptr;
  }
  if (tinput_ != nullptr) {
    free(tinput_);
    tinput_ = nullptr;
  }
  if (toutput_ != nullptr) {
    free(toutput_);
    toutput_ = nullptr;
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<Type, A, K, V, I>::trans_weights(
    Type *tweights, Type *weights)
{
#pragma omp parallel for collapse(2)
  for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
    for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
      int d = 16 * 16 * (_ic2 + _oc2 * this->ic2);
      MD(Type, aweights, [K][K][V][V], weights + K * K * d);
      MD(Type, atweights, [A][A][V][V], tweights + A * A * d);

      convolution_winograd_kernel<S_WEIGHTS(Type, A, K, V, I)>::trans_weights(
          atweights, aweights);
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<Type, A, K, V, I>::trans_input(
    Type *tinput, Type *input)
{
  MD(Type, atinput, [this->n][this->ic2][this->ht][this->wt][A][A][V],
      tinput);
  MD(Type, ainput, [this->n][this->ic2][this->ih][this->iw][V], input);
  int hA_end = (this->ih + this->lp) - (this->ht - 1) * (A - K + 1) - 1;
  int wA_end = (this->iw + this->tp) - (this->wt - 1) * (A - K + 1) - 1;

#pragma omp parallel for collapse(4)
  for (int _n = 0; _n < this->n; ++_n) {
    for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
      for (int _ht = 0; _ht < this->ht; ++_ht) {
        for (int _wt = 0; _wt < this->wt; ++_wt) {
          int _ih = _ht * (A - K + 1) - this->tp;
          int _iw = _wt * (A - K + 1) - this->lp;
          int _hA_start = (_ht > 0) ? 0 : this->tp;
          int _wA_start = (_wt > 0) ? 0 : this->lp;
          int _hA_end = (_ht < this->ht - 1) ? A - 1 : hA_end;
          int _wA_end = (_wt < this->wt - 1) ? A - 1 : wA_end;

          if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
              && _wA_end == A - 1) {
            convolution_winograd_kernel<S_INPUT(
                Type, A, K, V, I, BORDER(false))>::trans_input(*this,
                atinput[_n][_ic2][_ht][_wt],
                (Type *)ainput[_n][_ic2][_ih][_iw], 0, A - 1, 0, A - 1);
          } else {
            convolution_winograd_kernel<S_INPUT(
                Type, A, K, V, I, BORDER(true))>::trans_input(*this,
                atinput[_n][_ic2][_ht][_wt],
                (Type *)ainput[_n][_ic2][_ih][_iw], _hA_start, _hA_end,
                _wA_start, _wA_end);
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<Type, A, K, V, I>::product_trans_output(
    Type *output, Type *tinput, Type *tweights, Type *bias)
{
  auto ker_trans_output = this->with_bias
      ? convolution_winograd_kernel<S_OUTPUT(
            float, 5, 3, 16, I, BORDER(false), BIAS(true))>::trans_output
      : convolution_winograd_kernel<S_OUTPUT(
            float, 5, 3, 16, I, BORDER(false), BIAS(false))>::trans_output;
  auto ker_trans_output0 = this->with_bias
      ? convolution_winograd_kernel<S_OUTPUT(
            float, 5, 3, 16, I, BORDER(true), BIAS(true))>::trans_output
      : convolution_winograd_kernel<S_OUTPUT(
            float, 5, 3, 16, I, BORDER(true), BIAS(false))>::trans_output;

  MD(Type, atweights, [this->oc2][this->ic2][A][A][V][V], tweights);
  MD(Type, atinput, [this->n][this->ic2][this->ht][this->wt][A][A][V],
      tinput);
  MD(Type, aoutput, [this->n][this->oc2][this->oh][this->ow][V], output);
  MD(Type, abias, [this->oc2][V], bias);

  int hOA_end = this->oh % (A - K + 1) - 1;
  if (hOA_end == -1)
    hOA_end = A - K;
  int wOA_end = this->ow % (A - K + 1) - 1;
  if (wOA_end == -1)
    wOA_end = A - K;

#pragma omp parallel for collapse(4)
  for (int _n = 0; _n < this->n; ++_n) {
    for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
      for (int _ht = 0; _ht < this->ht; ++_ht) {
        for (int _wt = 0; _wt < this->wt; ++_wt) {
          int _oh = _ht * (A - K + 1);
          int _ow = _wt * (A - K + 1);
          int _hOA_end = (_ht < this->ht - 1) ? A - K : hOA_end;
          int _wOA_end = (_wt < this->wt - 1) ? A - K : wOA_end;

          Type tout[A][A][V];
          memset(&tout, 0, sizeof(tout));

          for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
            for (int _hA = 0; _hA < A; _hA++) {
              for (int _wA = 0; _wA < A; _wA++) {
                for (int _IV = 0; _IV < V; ++_IV) {
#pragma omp simd
                  for (int _OV = 0; _OV < V; ++_OV) {
                    tout[_hA][_wA][_OV]
                        += atweights[_oc2][_ic2][_hA][_wA][_IV][_OV]
                        * atinput[_n][_ic2][_ht][_wt][_hA][_wA][_IV];
                  }
                }
              }
            }
          }

          if (_hOA_end < A - K || _wOA_end < A - K) {
            ker_trans_output0(*this, (Type *)aoutput[_n][_oc2][_oh][_ow],
                tout, (Type *)abias[_oc2], _hOA_end, _wOA_end);
          } else {
            ker_trans_output(*this, (Type *)aoutput[_n][_oc2][_oh][_ow], tout,
                (Type *)abias[_oc2], _hOA_end, _wOA_end);
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<Type, A, K, V, I>::execute(
    Type *output, Type *input, Type *weights, Type *bias)
{
  if (is_first_run_)
    trans_weights(tweights_, weights);
  trans_input(tinput_, input);
  product_trans_output(output, tinput_, tweights_, bias);
  if (inference_acc_)
    is_first_run_ = false;
}

} // namespace euler
