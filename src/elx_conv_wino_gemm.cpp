#include "el_utils.hpp"
#include "elx_conv_wino_gemm.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_gemm_t<Type, A, K, V, I>::elx_conv_wino_gemm_t(
    eld_conv_t<Type> &dc)
    : elx_conv_t<Type>(dc) {
  this->V   = V;
  this->ic2 = this->ic / V;
  this->oc2 = this->oc / V;

  this->A  = A;
  this->ht = (this->oh + A - 3) / (A - K + 1);
  this->wt = (this->ow + A - 3) / (A - K + 1);
  this->nt = this->ht * this->wt;
  this->t  = this->nt * this->n;

  // trans-buffer blocking
  // ic: ic3, I2, V
  // oc: oc3, O2, V
  // t : t2,  T
  // I2, O2
  // tweights + pt-tinputs + pt-toutput ~ L2
  // tweights:gemm + tinputs:gemm + toutput:gemm ~ L1
  this->T  = 25;  // TODO: T selection
  this->O2 = 2;   // TODO: O2 selection
  this->I2 = 4;   // TODO: I2 selection

  this->oc3 = this->oc / (this->O2 * V);
  this->ic3 = this->ic / (this->I2 * V);
  this->t2  = this->t / this->T;

  // Tailing
  this->Ir = this->ic % V;
  this->Or = this->oc % V;
  this->Tr = this->t % this->T;
  // TODO: support tailing
  assert(this->Ir == 0);
  assert(this->Or == 0);
  assert(this->Tr == 0);

  size_t tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
  size_t tinput_size   = sizeof(Type) * A * A * this->t * this->ic;
  size_t toutput_size  = sizeof(Type) * A * A * this->t * this->oc;

  this->tweights = (Type *)malloc(tweights_size);
  this->tinput   = (Type *)malloc(tinput_size);
  this->toutput  = (Type *)malloc(toutput_size);

  if (this->input_fmt == nChw16c) {
    this->input_strides[0] = 1;
    this->input_strides[1] = V;
    this->input_strides[2] = V * this->iw;
    this->input_strides[3] = V * this->iw * this->ih;
    this->input_strides[4] = V * this->iw * this->ih * this->ic2;
  } else {
    // TODO
  }
  if (this->weights_fmt == OIhw16i16o) {
    this->weights_strides[0] = 1;
    this->weights_strides[1] = V;
    this->weights_strides[2] = V * V;
    this->weights_strides[3] = V * V * this->kw;
    this->weights_strides[4] = V * V * this->kw * this->kh;
    this->weights_strides[5] = V * V * this->kw * this->kh * this->ic2;
  } else {
    // TODO
  }
  if (this->output_fmt == nChw16c) {
    this->output_strides[0] = 1;
    this->output_strides[1] = V;
    this->output_strides[2] = V * this->ow;
    this->output_strides[3] = V * this->ow * this->oh;
    this->output_strides[4] = V * this->ow * this->oh * this->oc2;
  } else {
    // TODO
  }

  ker_trans_input_   = elk_trans_input<Type, A, K, V, I>;
  ker_trans_inputX_  = elk_trans_input<Type, A, K, V, I>;
  ker_trans_weights_ = elk_trans_weights<Type, A, K, V, I>;
  ker_trans_output_  = elk_trans_output<Type, A, K, V, I>;
  ker_trans_outputX_ = elk_trans_output<Type, A, K, V, I>;

#define CASE(z, T, data) case T:  ker_gemm_ = elk_gemm<Type, T,  V, I>; break;

  switch (this->T) {
    BOOST_PP_REPEAT_FROM_TO(1, 29, CASE, xxx)
    default:
      elx_error("Unimplemented");
      break;
  }

}

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_gemm_t<Type, A, K, V, I>::~elx_conv_wino_gemm_t() {
  if (this->tweights != nullptr) {
    free(this->tweights);
    this->tweights = nullptr;
  }
  if (this->tinput != nullptr) {
    free(this->tinput);
    this->tinput = nullptr;
  }
  if (this->toutput != nullptr) {
    free(this->toutput);
    this->toutput = nullptr;
  }
}

pragma_opt_core_avx512 template <typename Type, const int A, const int K,
                                 const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_weights(Type *tweights,
                                                           Type *weights) {
  // oc2, ic2, K, K, V, V => oc3, ic3, A, A, O2, I2, V, V
  mdarray<Type, 6> aweights(weights, this->oc2, this->ic2, K, K, V, V);
  mdarray<Type, 8> atweights(tweights, this->oc3, this->ic3, A, A, this->O2,
                             this->I2, V, V);
#pragma omp parallel for collapse(4)
  for_each(_oc3, this->oc3) {
    for_each(_ic3, this->ic3) {
      for_each(_O2, this->O2) {
        for_each(_I2, this->I2) {
          Type aout[A][A][V][V];
          Type in = aweights(_oc3 * this->O2 + _O2, _ic3 * this->I2 + _I2, 0, 0,
                             0, 0);
          MD(Type, ain, [K][K][V][V], &in);
          ker_trans_weights_(aout, ain);

          for_each(_hA, A) {
            for_each(_wA, A) {
              for_each(_iV, V) {
#pragma omp simd
                for_each(_oV, V) {
                  atweights(_oc3, _ic3, _hA, _wA, _O2, _I2, _iV, _oV) =
                    aout[_hA][_wA][_iV][_oV];
                }
              }
            }
          }
        }
      }
    }
  }
}

pragma_opt_core_avx512 template <typename Type, const int A, const int K,
                                 const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_input(Type *tinput,
                                                         Type *input, int _t2) {
  // n, ic2, ih, iw, V => t2=1, A, A, ic3, I2, T, V
  mdarray<Type, 5> ainput(input, this->n, this->ic2, this->ih, this->iw, V);
  mdarray<Type, 7> atinput(tinput, this->t2, A, A, this->ic3, this->I2, this->T,
                           V);

  Type aout[A][A][V];
  int _hA_start = this->lp; // first
  int _wA_start = this->tp;
  int _hA_end = (this->ih + this->lp) % (A - K + 1) - 1; // last
  int _wA_end = (this->iw + this->tp) % (A - K + 1) - 1;

  for_each(_ic3, this->ic3) {
    for_each(_I2, this->I2) {
      for_each(_T, this->T) {
        int _t = _t2 * this->T + _T;
        int _nt = _t % this->nt;
        int _ht = _nt / this->wt;
        int _wt = _nt % this->wt;
        int _ih = _ht * (A - K + 1) - this->lp;
        int _iw = _wt % (A - K + 1) - this->tp;

        if (_ht > 0) _hA_start = 0;
        if (_wt > 0) _wA_start = 0;
        if (_ht < this->ht - 1) _hA_end = A - 1;
        if (_wt < this->wt - 1) _wA_end = A - 1;

        Type *in = &ainput(_t / this->nt, _ic3 * this->I2 + _I2, _ih, _iw, 0);
        if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1 &&
            _wA_end == A - 1) {
          ker_trans_input_(*this, aout, in);
        } else
          ker_trans_inputX_(*this, aout, in, _hA_start, _hA_end, _wA_start,
                            _wA_end);

        for_each(_hA, A) {
          for_each(_wA, A) {
#pragma omp simd
            for_each(_V, V) {
              atinput(_t2, _hA, _wA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::gemm(Type *tinput, Type *tweights,
                                                  Type *toutput) {
  mdarray<Type, 8> atweights(tweights, this->oc3, this->ic3, A, A, this->O2,
                             this->I2, V, V);
  mdarray<Type, 6> atinput(tinput, A, A, this->ic3, this->I2, V);
  mdarray<Type, 6> atoutput(toutput, A, A, this->oc3, this->O2, V);

  for_each(_hA, A) {
    for_each(_wA, A) {
      for_each(_oc3, this->oc3) {
        for_each(_ic3, this->ic3) {
          ker_gemm_(*this, &atoutput(_hA, _wA, _oc3, 0, 0, 0),
                    &atinput(_hA, _wA, _ic3, 0, 0, 0),
                    &atweights(_oc3, _ic3, _hA, _wA, 0, 0, 0, 0), _ic3 == 0);
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_output(Type *output,
                                                          Type *toutput,
                                                          int _t2) {
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  mdarray<Type, 6> atoutput(toutput, A, A, this->oc3, this->O2, this->T, V);
  mdarray<Type, 5> aoutput(output, this->n, this->oc2, this->oh, this->ow, V);

  Type ain[A][A][V];
  int _hOA_end = this->oh % (A - K + 1) - 1;
  int _wOA_end = this->ow % (A - K + 1) - 1;

  for_each(_oc3, this->oc3) {
    for_each(_O2, this->O2) {
      for_each(_T, this->T) {
        for_each(_hA, A) {
          for_each(_wA, A) {
#pragma omp simd
            for_each(_V, V) {
              ain[_hA][_wA][_V] = atoutput(_hA, _wA, _oc3, _O2, _T, _V);
            }
          }
        }

        int _t = _t2 * this->T + _T;
        int _nt = _t % this->nt;
        int _ht = _nt / this->wt;
        int _wt = _nt % this->wt;
        int _oh = _ht * (A - K + 1);
        int _ow = _wt * (A - K + 1);
        Type *out = &aoutput(_t / this->nt, _oc3 * this->O2 + _O2, _oh, _ow, 0);
        if (_ht < this->ht - 1)
          _hOA_end = A - K; // A - K + 1 - 1
        if (_wt < this->wt - 1)
          _wOA_end = A - K;

        if (_hOA_end < A - K || _wOA_end < A - K) {
          ker_trans_outputX_(*this, out, ain, _hOA_end, _wOA_end);
        } else {
          ker_trans_output_(*this, out, ain);
        }
      }
    }
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::winograd(Type *input,
                                                      Type *weights,
                                                      Type *output,
                                                      Type *bias) {
  // TODO: support bias
  if (bias == nullptr) return;
  trans_weights(this->tweights, weights);

#pragma omp parallel for
  for_each(_t2, this->t2) {
    trans_input(this->tinput, input, _t2);
    gemm(this->tinput, this->tweights, this->toutput);
    trans_output(output, this->toutput, _t2);
  }
}

}  // namespace euler
