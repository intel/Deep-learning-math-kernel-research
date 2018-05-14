#include <assert.h>
#include <stdlib.h>
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
    : elx_conv_t<Type>(dc) {
  this->V = V;
  this->ic2 = this->ic / V;
  this->oc2 = this->oc / V;

  this->A = A;
  this->ht = (this->oh + A - 3) / (A - K + 1);
  this->wt = (this->ow + A - 3) / (A - K + 1);

  int tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
  int tinput_size =
      sizeof(Type) * A * A * this->ht * this->wt * this->ic * this->n;
  // int toutput_size  = sizeof(Type) * A * A * this->ht * this->wt
  //                                         * this->oc * this->n;
  tweights_ = (Type *)malloc(tweights_size);
  tinput_ = (Type *)malloc(tinput_size);
  toutput_  = nullptr;
}

template <typename T, const int A, const int K, const int V, const int I>
elx_conv_wino_prod_t<T, A, K, V, I>::~elx_conv_wino_prod_t() {
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

template <typename T, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<T, A, K, V, I>::trans_weights(T *tweights,
                                                        T *weights) {
#pragma omp parallel for collapse(2)
  for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
    for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
      int d = 16 * 16 * (_ic2 + _oc2 * this->ic2);
      MD(T, aweights, [K][K][V][V], weights + K * K * d);
      MD(T, atweights, [A][A][V][V], tweights + A * A * d);

      elk_trans_weights<T, A, K, V, I>(atweights, aweights);
    }
  }
}

template <typename T, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<T, A, K, V, I>::trans_input(T *tinput, T *input) {
  MD(T, atinput, [this->n][this->ic2][this->ht][this->wt][A][A][V], tinput);
  MD(T, ainput, [this->n][this->ic2][this->ih][this->iw][V], input);

#pragma omp parallel for collapse(4)
  for (int _n = 0; _n < this->n; ++_n) {
    for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
      for (int _ht = 0; _ht < this->ht; ++_ht) {
        for (int _wt = 0; _wt < this->wt; ++_wt) {
          int _ih = _ht * (A - K + 1) - this->lp;
          int _iw = _wt * (A - K + 1) - this->tp;
          int _wT_start = 0;
          int _hT_start = 0;
          int _wT_end = this->iw - _iw - 1;
          int _hT_end = this->ih - _ih - 1;
          if (_ih < 0) {
            _hT_start = -_ih;
            _ih = 0;
          }
          if (_iw < 0) {
            _wT_start = -_iw;
            _iw = 0;
          }

          if (_hT_start == 0 && _wT_start == 0 && _hT_end == A - 1 &&
              _wT_end == A - 1) {
            elk_trans_input<T, A, K, V, I>(*this, atinput[_n][_ic2][_ht][_wt],
                                           (T *)ainput[_n][_ic2][_ih][_iw]);
          } else {
            elk_trans_input<T, A, K, V, I>(*this, atinput[_n][_ic2][_ht][_wt],
                                           (T *)ainput[_n][_ic2][_ih][_iw],
                                           _hT_start, _hT_end, _wT_start,
                                           _wT_end);
          }
        }
      }
    }
  }
}

template <typename T, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<T, A, K, V, I>::product_trans_output(T *tinput,
                                                               T *tweights,
                                                               T *output) {
  MD(T, atweights, [this->oc2][this->ic2][A][A][V][V], tweights);
  MD(T, atinput, [this->n][this->ic2][this->ht][this->wt][A][A][V], tinput);
  MD(T, aoutput, [this->n][this->oc2][this->oh][this->ow][V], output);

#pragma omp parallel for collapse(4)
  for (int _n = 0; _n < this->n; ++_n) {
    for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
      for (int _ht = 0; _ht < this->ht; ++_ht) {
        for (int _wt = 0; _wt < this->wt; ++_wt) {
          elk_product_trans_output<T, A, K, V, I>(
              *this, (T *)atinput[_n], (T *)atweights[_oc2],
              (T *)aoutput[_n][_oc2], _ht, _wt);
        }
      }
    }
  }
}

template <typename T, const int A, const int K, const int V, const int I>
void elx_conv_wino_prod_t<T, A, K, V, I>::winograd(T *input, T *weights,
                                                   T *output, T *bias) {
  // TODO: support bias
  if (bias != nullptr) return;
  trans_weights(tweights_, weights);
  trans_input(tinput_, input);
  product_trans_output(tinput_, tweights_, output);
}

}  // namespace euler
