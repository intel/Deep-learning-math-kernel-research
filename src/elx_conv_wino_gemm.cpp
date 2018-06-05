#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_wino_gemm.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

//
// -------------+-------------------+--------------+---------------
//  execute-opt | thread-teaming-by | fusion-along | duplication
// -------------+-------------------+--------------+---------------
//      848     |        _          |      t       |    _
// -------------+-------------------+--------------+---------------
//      442     |        t          |      t       |    W
// -------------+-------------------+--------------+---------------
//      241     |        o          |      t       |    I
// -------------+-------------------+--------------+---------------
//      888     |        _          |      _       |    _
// -------------+-------------------+--------------+---------------
//      281     |        o          |      _       |    I
// -------------+-------------------+--------------+---------------
//      828*    |        _          |      o       |    _
// -------------+-------------------+--------------+---------------
//  *: TODO
//

const unsigned TTM_I = 0x100;
const unsigned TTM_O = 0x200;
const unsigned TTM_T = 0x400;
const unsigned TTM_N = 0x800;

const unsigned FUS_I = 0x10;
const unsigned FUS_O = 0x20;
const unsigned FUS_T = 0x40;
const unsigned FUS_N = 0x80;

const unsigned DUP_I = 0x1;
const unsigned DUP_W = 0x2;
const unsigned DUP_N = 0x8;

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

  // TODO: santize user settings
  if (this->I2 == 0) this->I2 = 4; // TODO: I2 selection
  if (this->O2 == 0) this->O2 = 2; // TODO: O2 selection
  if (this->T == 0)  this->T = 18; // TODO: T selection

  // Tailing
  this->Ir = this->ic % V;
  this->Or = this->oc % V;
  this->Tr = this->t % this->T;
  if (this->Tr == 0) this->Tr = this->T;
  // TODO: support tailing
  assert(this->Ir == 0);
  assert(this->Or == 0);

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

  // TODO: add tailing?
  this->oc3 = this->oc / (this->O2 * V);
  this->ic3 = this->ic / (this->I2 * V);
  this->t2 = (this->t + this->T - 1) / this->T;

  xopt_ = this->execution_mode;
  if (xopt_ == 0) {
    // TODO: deduce xopt
    xopt_ = TTM_O | FUS_T | DUP_I;
  }

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::prepare_execute_opt()
{
  size_t tweights_size, tinput_size, toutput_size;
  size_t in_ndup = 1, wei_ndup = 1;
  size_t nt;

  auto divide_tasks_ttm = [this](size_t tasks) {
    size_t ntasks_base = tasks / this->nteams;
    size_t rem = tasks - this->nteams * ntasks_base;
    for (size_t s = 0; s < this->nteams; s++) {
      if (s < rem) {
        ttm_[s].start = (ntasks_base + 1) * s;
        ttm_[s].end = ttm_[s].start + ntasks_base;
      } else {
        ttm_[s].start = rem * (ntasks_base + 1) + (s - rem) * ntasks_base;
        ttm_[s].end = ttm_[s].start + ntasks_base - 1;
      }
      // dbg
      printf("ttm_[%ld]=[%d,%d]\n", s, ttm_[s].start, ttm_[s].end);
    }
  };

  stream_in_ = false;
  stream_wei_ = false;

  if (xopt_ & TTM_T) {
    divide_tasks_ttm(this->t2);
  }
  if (xopt_ & TTM_O) {
    this->oc3 /= this->nteams;
  }
  if (xopt_ & TTM_N) {
    this->nthreads = mthr_;
    this->nteams = 1;
  }
  if (xopt_ & FUS_N) {
    nt = this->t2;
    stream_in_ = true;
    stream_wei_ = true;
  }
  if (xopt_ & FUS_T) {
    nt = this->nteams * this->nthreads;
  }
  if (xopt_ & DUP_I) {
    if (!(xopt_ & FUS_T))
      in_ndup = this->nteams;
  }
  if (xopt_ & DUP_W) {
    wei_ndup = this->nteams;
  }

  tweights_size = sizeof(Type) * A * A * this->ic * this->oc * wei_ndup;
  tinput_size = sizeof(Type) * A * A * this->T * this->ic * nt * in_ndup;
  toutput_size = sizeof(Type) * A * A * this->T * this->oc * nt;

  tweights_ = (Type *)memalign(64, tweights_size);
  tinput_ = (Type *)memalign(64, tinput_size);
  toutput_ = (Type *)memalign(64, toutput_size);

  // dbg
  printf("wei_ndup=%ld, in_ndup=%ld, nt=%ld\n", wei_ndup, in_ndup, nt);
  size_t l2_usage = tweights_size / this->nteams + sizeof(Type) * A * A * this->T * (this->ic + this->oc);
  size_t l1_usage = sizeof(Type) * this->O2 * this->I2 * V * V + sizeof(Type) * this->T * V * (this->I2 + this->O2);
  printf("l2_usage=%ld, l1_usage=%ld\n", l2_usage, l1_usage);
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::bind_execute_functions()
{
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

#define EXECUTE_CASE(n)                                                      \
  case 0x##n:                                                                \
    printf("execute_opt=" #n "\n");\
    execute_opt_ = &elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute##n;    \
    break

  switch (xopt_) {
  EXECUTE_CASE(241);
  EXECUTE_CASE(281);
  EXECUTE_CASE(442);
  EXECUTE_CASE(848);
  EXECUTE_CASE(888);
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
  MD(Type, aweights, [this->oc2][this->ic2][K][K][V][V], weights);
  MD(Type, atweights, [this->oc3][this->ic3][A][A][this->O2][this->I2][V][V],
      tweights);
#pragma omp for nowait collapse(4) schedule(static)
  for_each (_oc3, this->oc3) {
    for_each (_ic3, this->ic3) {
      for_each (_O2, this->O2) {
        for_each (_I2, this->I2) {
          Type aout[A][A][V][V];
          Type *in = (Type *)
              aweights[_oc3 * this->O2 + _O2][_ic3 * this->I2 + _I2];
          using Array = Type[K][K][V][V];
          ker_trans_weights_(aout, *(Array *)in);

          for_each (_hA, A) {
            for_each (_wA, A) {
              for_each (_iV, V) {
                if (I == ISA_SKX_AVX512) {
                  if (stream_wei_)
                    _mm512_stream_ps(
                        atweights[_oc3][_ic3][_hA][_wA][_O2][_I2][_iV],
                        *((__m512 *)&aout[_hA][_wA][_iV][0]));
                  else
                    _mm512_store_ps(
                        atweights[_oc3][_ic3][_hA][_wA][_O2][_I2][_iV],
                        *((__m512 *)&aout[_hA][_wA][_iV][0]));
                } else {
#pragma omp simd
                  for_each (_oV, V)
                    atweights[_oc3][_ic3][_hA][_wA][_O2][_I2][_iV][_oV]
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
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__trans_input(
    Type *tinput, Type *input, int _t2, int Tz)
{
  // n, ic2, ih, iw, V => t2=1, A, A, ic3, I2, T, V
  MD(Type, ainput, [this->n][this->ic2][this->ih][this->iw][V], input);
  MD(Type, atinput,[A][A][this->ic3][this->I2][Tz][V], tinput);

  alignas(64) Type aout[A][A][V];
  int hA_end = (this->ih + this->lp) - (this->ht - 1) * (A - K + 1) - 1;
  int wA_end = (this->iw + this->tp) - (this->wt - 1) * (A - K + 1) - 1;

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

    Type *in = ainput[_t / this->nt][0][_ih][_iw];
    if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
        && _wA_end == A - 1) {
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    } else {
      ker_trans_input0_(
          *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
    }

    for_each (_hA, A) {
      for_each (_wA, A) {
        if (I == ISA_SKX_AVX512) {
          if (stream_in_)
            _mm512_stream_ps(
                atinput[_hA][_wA][0][0][_T], *((__m512 *)&aout[_hA][_wA][0]));
          else
            _mm512_store_ps(
                atinput[_hA][_wA][0][0][_T], *((__m512 *)&aout[_hA][_wA][0]));
        } else {
#pragma omp simd
          for_each (_V, V)
            atinput[_hA][_wA][0][0][_T][_V] = aout[_hA][_wA][_V];
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_input(
    Type *tinput, Type *input, int _t2, int Tz)
{
  MD(Type, ainput, [this->n][this->ic2][this->ih][this->iw][V], input);
  MD(Type, atinput, [A][A][this->ic3][this->I2][Tz][V], tinput);

  for_each (_ic3, this->ic3) {
    for_each (_I2, this->I2) {
      __trans_input((Type *)atinput[0][0][_ic3][_I2],
                    (Type *)ainput[0][_ic3 * this->I2 + _I2], _t2, Tz);
    }
  }
}


template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_input(
    Type *tinput, Type *input)
{
  MD(Type, ainput, [this->n][this->ic2][this->ih][this->iw][V], input);
  MD(Type, atinput2, [this->t2][A * A * this->T * this->ic], tinput);

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
    for_each (_ic3, this->ic3) {
      for_each (_I2, this->I2) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        MD(Type, atinput6, [A][A][this->ic3][this->I2][Tz][V], atinput2[_t2]);
        __trans_input((Type *)atinput6[0][0][_ic3][_I2],
                      (Type *)ainput[0][_ic3 * this->I2 + _I2], _t2, Tz);
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights, int _t2, int Tz)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;

  MD(Type, atinput, [A][A][this->ic3][this->I2][Tz][V], tinput);
  MD(Type, atoutput, [A][A][this->oc3][this->O2][Tz][V], toutput);
  MD(Type, atweights, [this->oc3][this->ic3][A][A][this->O2][this->I2][V][V],
      tweights);

  for_each (_hA, A) {
    for_each (_wA, A) {
      for_each (_oc3, this->oc3) {
        for_each (_ic3, this->ic3) {
          ker_gemm(*this, (Type *)atoutput[_hA][_wA][_oc3],
              (Type *)atinput[_hA][_wA][_ic3],
              (Type *)atweights[_oc3][_ic3][_hA][_wA], _ic3 == 0);
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::gemm(
    Type *toutput, Type *tinput, Type *tweights)
{
  MD(Type, atinput2, [this->t2][A * A * this->T * this->ic], tinput);
  MD(Type, atoutput2, [this->t2][A * A * this->T * this->oc3 * this->O2 * V], toutput);
  MD(Type, atweights, [this->oc3][this->ic3][A][A][this->O2][this->I2][V][V],
      tweights);

#pragma omp for nowait collapse(4)
  for_each (_t2, this->t2) {
    for_each (_hA, A) {
      for_each (_wA, A) {
        for_each (_oc3, this->oc3) {
          for_each (_ic3, this->ic3) {
            int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
            auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
            MD(Type, atinput6, [A][A][this->ic3][this->I2][Tz][V],
                atinput2[_t2]);
            MD(Type, atoutput6, [A][A][this->oc3][this->O2][Tz][V],
                atoutput2[_t2]);
            ker_gemm(*this, (Type *)atoutput6[_hA][_wA][_oc3],
                (Type *)atinput6[_hA][_wA][_ic3],
                (Type *)atweights[_oc3][_ic3][_hA][_wA], _ic3 == 0);
          }
        }
      }
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__trans_output(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD(Type, atoutput, [A][A][this->oc3][this->O2][Tz][V], toutput);
  MD(Type, aoutput, [this->n][this->oc2][this->oh][this->ow][V], output);

  Type ain[A][A][V];
  int hOA_end = this->oh % (A - K + 1) - 1;
  if (hOA_end == -1) hOA_end = A - K;
  int wOA_end = this->ow % (A - K + 1) - 1;
  if (wOA_end == -1) wOA_end = A - K;

  for_each (_T, Tz) {
    for_each (_hA, A) {
      for_each (_wA, A) {
#pragma omp simd
        for_each (_V, V) {
          ain[_hA][_wA][_V] = atoutput[_hA][_wA][0][0][_T][_V];
        }
      }
    }
    int _t = _t2 * this->T + _T;
    int _nt = _t % this->nt;
    int _ht = _nt / this->wt;
    int _wt = _nt % this->wt;
    int _oh = _ht * (A - K + 1);
    int _ow = _wt * (A - K + 1);
    Type *out = aoutput[_t / this->nt][0][_oh][_ow];

    int _hOA_end = (_ht < this->ht - 1) ? A - K : hOA_end;
    int _wOA_end = (_wt < this->wt - 1) ? A - K : wOA_end;

    if (_hOA_end < A - K || _wOA_end < A - K) {
      ker_trans_output0_(*this, out, ain, bias, _hOA_end, _wOA_end);
    } else {
      ker_trans_output_(*this, out, ain, bias, A - K, A - K);
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias, int _t2, int Tz)
{
  MD(Type, aoutput, [this->n][this->oc2][this->oh][this->ow][V], output);
  MD(Type, atoutput, [A][A][this->oc3][this->O2][Tz][V], toutput);
  MD(Type, abias, [this->oc3][this->O2][V], bias);

  for_each (_oc3, this->oc3) {
    for_each (_O2, this->O2) {
      __trans_output((Type *)aoutput[0][_oc3 * this->O2 + _O2],
          (Type *)atoutput[0][0][_oc3][_O2], (Type *)abias[_oc3][_O2], _t2, Tz);
    }
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::trans_output(
    Type *output, Type *toutput, Type *bias)
{
  MD(Type, aoutput, [this->n][this->oc2][this->oh][this->ow][V], output);
  MD(Type, abias, [this->oc3][this->O2][V], bias);
  MD(Type, atoutput2, [this->t2][A * A * this->T * this->oc3 * this->O2 * V], toutput);

#pragma omp for nowait collapse(3)
  for_each (_t2, this->t2) {
    for_each (_oc3, this->oc3) {
      for_each (_O2, this->O2) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        MD(Type, atoutput6, [A][A][this->oc3][this->O2][Tz][V], atoutput2[_t2]);
        __trans_output((Type *)aoutput[0][_oc3 * this->O2 + _O2],
            (Type *)atoutput6[0][0][_oc3][_O2], (Type *)abias[_oc3][_O2], _t2,
            Tz);
      }
    }
  }
}

// Fuse trans-input, gemm and trans-output along 't' dimension
//
// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute848(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD(Type, atinput2, [mthr_][A * A * this->T * this->ic], tinput_);
  MD(Type, atoutput2, [mthr_][A * A * this->T * this->oc], toutput_);

#pragma omp parallel proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(tweights_, weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1)
    for_each (_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      // TODO: fuse ic3/I2?
      trans_input(atinput2[ithr], input, _t2, Tz);
      gemm(atoutput2[ithr], atinput2[ithr], tweights_, _t2, Tz);
      trans_output(output, atoutput2[ithr], bias, _t2, Tz);
    }
  }
  if (inference_acc_) is_first_run_ = false;
}

// Thread-teaming along 't' dimension. TODO: ttm along 'o'
// Fuse trans-input, gemm and trans-output along 't' dimension
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute442(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD(Type, atinput3, [this->nteams][this->nthreads][A * A * this->T * this->ic], tinput_);
  MD(Type, atoutput3, [this->nteams][this->nthreads][A * A * this->T * this->oc], toutput_);
  MD(Type, atweights2, [this->nteams][this->oc * this->ic * A * A], tweights_);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights((Type *)atweights2[s], weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1) schedule(static)
    for (int _t2 = ttm_[s].start; _t2 <= ttm_[s].end; _t2++) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      trans_input(atinput3[s][ithr], input, _t2, Tz);
      gemm(atoutput3[s][ithr], atinput3[s][ithr], atweights2[s], _t2, Tz);
      trans_output(output, atoutput3[s][ithr], bias, _t2, Tz);
    }
  }
  if (inference_acc_) is_first_run_ = false;
}

// Flat mode
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute888(
    Type *output, Type *input, Type *weights, Type *bias)
{
#pragma omp parallel proc_bind(close)
  {
    if (is_first_run_)
      trans_weights(tweights_, weights);
    trans_input(tinput_, input);
#pragma omp barrier
    gemm(toutput_, tinput_, tweights_);
#pragma omp barrier
    trans_output(output, toutput_, bias);
  }
  if (inference_acc_) is_first_run_ = false;
}

// Thread teaming along 'o' dimension.
// Flat mode (no-fusion)
//
// tweights: nteams, oc3, ic3, A, A, O2, I2, V, V (oc3 /= nteams)
// tinputs:  nteams, t2, A, A, ic3, I2, T, V (dup)
// toutput:  nteams, t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute281(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD(Type, aoutput3, [this->n][this->nteams][this->oc * this->oh * this->ow / this->nteams], output);
  MD(Type, aweights2, [this->nteams][this->oc * this->ic * K * K / this->nteams], weights);
  MD(Type, abias2, [this->nteams][this->oc / this->nteams], bias);
  MD(Type, atinput2, [this->nteams][this->t2 * A * A * this->T * this->ic], tinput_);
  MD(Type, atoutput2, [this->nteams][this->t2 * A * A * this->T * this->oc / this->nteams], toutput_);
  MD(Type, atweights2, [this->nteams][this->oc * this->ic * A * A / this->nteams], tweights_);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_)
      trans_weights((Type *)atweights2[s], (Type *)aweights2[s]);
    trans_input((Type *)atinput2[s], input);
#pragma omp barrier
    gemm((Type *)atoutput2[s], (Type *)atinput2[s], (Type *)atweights2[s]);
#pragma omp barrier
    trans_output((Type *)aoutput3[0][s], (Type *)atoutput2[s], (Type *)abias2[s]);
  }
  if (inference_acc_) is_first_run_ = false;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::__execute241(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD(Type, aoutput3, [this->n][this->nteams][this->oc * this->oh * this->ow / this->nteams], output);
  MD(Type, aweights2, [this->nteams][this->oc * this->ic * K * K / this->nteams], weights);
  MD(Type, abias2, [this->nteams][this->oc / this->nteams], bias);
  MD(Type, atinput3, [this->nteams][this->nthreads][A * A * this->T * this->ic], tinput_);
  MD(Type, atoutput3, [this->nteams][this->nthreads][A * A * this->T * this->oc / this->nteams], toutput_);
  MD(Type, atweights2, [this->nteams][this->oc * this->ic * A * A / this->nteams], tweights_);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights((Type *)atweights2[s], (Type *)aweights2[s]);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1)
    for_each (_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      trans_input(atinput3[s][ithr], input, _t2, Tz);
      gemm(atoutput3[s][ithr], atinput3[s][ithr], atweights2[s], _t2, Tz);
      trans_output(aoutput3[0][s], atoutput3[s][ithr], abias2[s], _t2, Tz);
    }
  }
  if (inference_acc_) is_first_run_ = false;
}



template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, V, I>::execute(
    Type *output, Type *input, Type *weights, Type *bias)
{
  (this->*execute_opt_)(output, input, weights, bias);
}

} // namespace euler
