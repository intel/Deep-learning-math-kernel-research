#include <x86intrin.h>
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
    eld_conv_t<Type>& dc)
    : elx_conv_t<Type>(dc)
{
  this->V = V;
  this->ic2 = this->ic / V;
  this->oc2 = this->oc / V;

  this->A = A;
  this->ht = (this->oh + A - K) / (A - K + 1);
  this->wt = (this->ow + A - K) / (A - K + 1);
  this->nt = this->ht * this->wt;
  this->t = this->nt * this->n;

  // trans-buffer blocking
  // ic: ic3, I2, V
  // oc: oc3, O2, V
  // t : t2,  T
  // I2, O2
  // tweights + pt-tinputs + pt-toutput ~ L2
  // tweights:gemm + tinputs:gemm + toutput:gemm ~ L1
  this->T = 18; // TODO: T selection
  this->O2 = 4; // TODO: O2 selection
  this->I2 = 4; // TODO: I2 selection

  // Tailing
  this->Ir = this->ic % V;
  this->Or = this->oc % V;
  this->Tr = this->t % this->T;
  if (this->Tr == 0) this->Tr = this->T;
  // TODO: support tailing
  assert(this->Ir == 0);
  assert(this->Or == 0);

  // TODO: add tailing?
  this->oc3 = this->oc / (this->O2 * V);
  this->ic3 = this->ic / (this->I2 * V);
  this->t2 = (this->t + this->T - 1) / this->T;

  // dbg
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc);

  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  if (this->nteams == 0 || this->nthreads == 0
      || this->nteams * this->nthreads > mthr_
      || this->nteams > MAX_THREAD_TEAMS) {
    this->nteams = 1;
    this->nthreads = mthr_;
  }
  inference_acc_ = this->prop_kind == forward_inference;

  int ntasks_base = this->t2 / this->nteams;
  int rem = this->t2 - this->nteams * ntasks_base;
  for_each (s, this->nteams) {
    if (s < rem) {
      ttm_[s].start = (ntasks_base + 1) * s;
      ttm_[s].end = ttm_[s].start + ntasks_base;
    } else {
      ttm_[s].start = rem * (ntasks_base + 1) + (s - rem) * ntasks_base;
      ttm_[s].end = ttm_[s].start + ntasks_base - 1;
    }
    // dbg
    printf("ttm_[%d]=[%d,%d]\n", s, ttm_[s].start, ttm_[s].end);
  }

  size_t tweights_size
      = sizeof(Type) * A * A * this->ic * this->oc * this->nteams;
  size_t tinput_size = sizeof(Type) * A * A * this->T * this->ic * mthr_;
  size_t toutput_size = sizeof(Type) * A * A * this->T * this->oc * mthr_;

  tweights_ = (Type *)memalign(64, tweights_size);
  tinput_ = (Type *)memalign(64, tinput_size);
  toutput_ = (Type *)memalign(64, toutput_size);

  // dbg
  size_t l2_usage = tweights_size / this->nteams + sizeof(Type) * A * A * this->T * (this->ic + this->oc);
  size_t l1_usage = sizeof(Type) * this->O2 * this->I2 * V * V + sizeof(Type) * this->T * V * (this->I2 + this->O2);
  printf("l2_usage=%ld, l1_usage=%ld\n", l2_usage, l1_usage);

  ker_trans_input_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(false))>::trans_input;
  ker_trans_input0_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(true))>::trans_input;
  ker_trans_weights_ = convolution_winograd_kernel<S_WEIGHTS(
      Type, A, K, V, I)>::trans_weights;
  if (this->with_bias) {
    ker_trans_output_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(true))>::trans_output;
    ker_trans_output0_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(true))>::trans_output;
  } else {
    ker_trans_output_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(false))>::trans_output;
    ker_trans_output0_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(false))>::trans_output;
  }

#define GEMM_CASE(z, n, data)                                                \
  case n:                                                                    \
    ker_gemm_ = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm;    \
    break;

  switch (this->T) {
    BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_CASE, nil)
  default:
    el_error("Unimplemented");
    break;
  }
#define GEMM_CASE0(z, n, data)                                               \
  case n:                                                                    \
    ker_gemm0_ = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm;   \
    break;
  switch (this->Tr) {
    BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_CASE0, nil);
  default:
    el_error("Unimplemented");
    break;
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_gemm_t<Type, A, K, V, I>::~elx_conv_wino_gemm_t()
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
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_weights(
    Type *tweights, Type *weights)
{
  // oc2, ic2, K, K, V, V => oc3, ic3, A, A, O2, I2, V, V
  mdarray<Type, 6> aweights(weights, this->oc2, this->ic2, K, K, V, V);
  mdarray<Type, 8> atweights(
      tweights, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);
#pragma omp for nowait collapse(4) schedule(static)
  for_each (_oc3, this->oc3) {
    for_each (_ic3, this->ic3) {
      for_each (_O2, this->O2) {
        for_each (_I2, this->I2) {
          Type aout[A][A][V][V];
          Type *in = &aweights(
              _oc3 * this->O2 + _O2, _ic3 * this->I2 + _I2, 0, 0, 0, 0);
          using Array = Type[K][K][V][V];
          ker_trans_weights_(aout, *(Array *)in);

          for_each (_hA, A) {
            for_each (_wA, A) {
              for_each (_iV, V) {
#pragma omp simd
                for_each (_oV, V) {
                  atweights(_oc3, _ic3, _hA, _wA, _O2, _I2, _iV, _oV)
                      = aout[_hA][_wA][_iV][_oV];
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_input(
    Type *tinput, Type *input, int _t2)
{
  int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;

  // n, ic2, ih, iw, V => t2=1, A, A, ic3, I2, T, V
  mdarray<Type, 5> ainput(input, this->n, this->ic2, this->ih, this->iw, V);
  mdarray<Type, 6> atinput(tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) Type aout[A][A][V];
  int hA_end = (this->ih + this->lp) - (this->ht - 1) * (A - K + 1) - 1;
  int wA_end = (this->iw + this->tp) - (this->wt - 1) * (A - K + 1) - 1;

  for_each (_ic3, this->ic3) {
    for_each (_I2, this->I2) {
      for_each (_T, Tz) {
        int _t = _t2 * this->T + _T;
        int _nt = _t % this->nt;
        int _ht = _nt / this->wt;
        int _wt = _nt % this->wt;
        int _ih = _ht * (A - K + 1) - this->lp; // may < 0
        int _iw = _wt * (A - K + 1) - this->tp;
        int _hA_start = (_ht > 0) ? 0 : this->lp;
        int _wA_start = (_wt > 0) ? 0 : this->tp;
        int _hA_end = (_ht < this->ht - 1) ? A - 1 : hA_end;
        int _wA_end = (_wt < this->wt - 1) ? A - 1 : wA_end;

        Type *in = &ainput(_t / this->nt, _ic3 * this->I2 + _I2, _ih, _iw, 0);
        if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
            && _wA_end == A - 1) {
          ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
        } else {
          ker_trans_input0_(
              *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
        }

        for_each (_hA, A) {
          for_each (_wA, A) {
#pragma omp simd
            for_each (_V, V) {
              atinput(_hA, _wA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
            }
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::gemm(
    Type *toutput, Type *tweights, Type *tinput, int _t2)
{
  int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  mdarray<Type, 8> atweights(
      tweights, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);
  mdarray<Type, 6> atinput(tinput, A, A, this->ic3, this->I2, Tz, V);
  mdarray<Type, 6> atoutput(toutput, A, A, this->oc3, this->O2, Tz, V);


  for_each (_hA, A) {
    for_each (_wA, A) {
      for_each (_oc3, this->oc3) {
        for_each (_ic3, this->ic3) {
          ker_gemm(*this, &atoutput(_hA, _wA, _oc3, 0, 0, 0),
              &atinput(_hA, _wA, _ic3, 0, 0, 0),
              &atweights(_oc3, _ic3, _hA, _wA, 0, 0, 0, 0), _ic3 == 0);
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _t2)
{
  int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;

  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  mdarray<Type, 6> atoutput(toutput, A, A, this->oc3, this->O2, Tz, V);
  mdarray<Type, 5> aoutput(output, this->n, this->oc2, this->oh, this->ow, V);
  mdarray<Type, 3> abias(bias, this->oc3, this->O2, V);

  Type ain[A][A][V];
  int hOA_end = this->oh % (A - K + 1) - 1;
  if (hOA_end == -1) hOA_end = A - K;
  int wOA_end = this->ow % (A - K + 1) - 1;
  if (wOA_end == -1) wOA_end = A - K;

  for_each (_oc3, this->oc3) {
    for_each (_O2, this->O2) {
      for_each (_T, Tz) {
        for_each (_hA, A) {
          for_each (_wA, A) {
#pragma omp simd
            for_each (_V, V) {
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
        Type* out = &aoutput(_t / this->nt, _oc3 * this->O2 + _O2, _oh, _ow, 0);
        Type* b = &abias(_oc3, _O2, 0);

        int _hOA_end = (_ht < this->ht - 1) ? A - K : hOA_end;
        int _wOA_end = (_wt < this->wt - 1) ? A - K : wOA_end;

        if (_hOA_end < A - K || _wOA_end < A - K) {
          ker_trans_output0_(*this, out, ain, b, _hOA_end, _wOA_end);
        } else {
          ker_trans_output_(*this, out, ain, b, A - K, A - K);
        }
      }
    }
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute(
    Type *output, Type *input, Type *weights, Type *bias)
{
  mdarray<Type, 2> atinput(tinput_, mthr_, A * A * this->T * this->ic);
  mdarray<Type, 2> atoutput(toutput_, mthr_, A * A * this->T * this->oc);

#pragma omp parallel proc_bind(close)
  {
    trans_weights(tweights_, weights);
#pragma omp barrier

#pragma omp for nowait collapse(1)
    for_each (_t2, this->t2) {
      size_t ithr = omp_get_thread_num();
      trans_input(&atinput(ithr, 0), input, _t2);
      gemm(&atoutput(ithr, 0), tweights_, &atinput(ithr, 0), _t2);
      trans_output(output, &atoutput(ithr, 0), bias, _t2);
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute_inf(
    Type *output, Type *input, Type *weights, Type *bias)
{
  mdarray<Type, 3> atinput(
      tinput_, this->nteams, this->nthreads, A * A * this->T * this->ic);
  mdarray<Type, 3> atoutput(
      toutput_, this->nteams, this->nthreads, A * A * this->T * this->oc);
  mdarray<Type, 2> atweights(
      tweights_, this->nteams, A * A * this->ic * this->oc);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(&atweights(s, 0), weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1) schedule(static)
    for (int _t2 = ttm_[s].start; _t2 <= ttm_[s].end; _t2++) {
      size_t ithr = omp_get_thread_num();
      trans_input(&atinput(s, ithr, 0), input, _t2);
      gemm(&atoutput(s, ithr, 0), &atweights(s, 0), &atinput(s, ithr, 0), _t2);
      trans_output(output, &atoutput(s, ithr, 0), bias, _t2);
    }
  }
  is_first_run_ = false;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::execute(
    Type *output, Type *input, Type *weights, Type *bias)
{
  if (inference_acc_)
    __execute_inf(output, input, weights, bias);
  else
    __execute(output, input, weights, bias);
}

} // namespace euler
