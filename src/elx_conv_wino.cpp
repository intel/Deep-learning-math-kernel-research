#include <string.h>
#include <x86intrin.h>
#include "el_utils.hpp"
#include "elx_conv_wino.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elk_conv_wino.hpp"
#include "elx_conv.hpp"
#include "euler.hpp"

namespace euler {

//
// -------------+-------------------+--------------+---------------
//  execute-opt | thread-teaming-by | fusion-along | duplication
// -------------+-------------------+--------------+---------------
//     A040     |        _          |      t       |    _
// -------------+-------------------+--------------+---------------
//     A048*    |        _          |      t       |    W
// -------------+-------------------+--------------+---------------
//     A060*    |        _          |    t + o     |    _
// -------------+-------------------+--------------+---------------
//     A061     |        _          |    t + o     |    I
// -------------+-------------------+--------------+---------------
//     A069*    |        _          |    t + o     |  I + W
// -------------+-------------------+--------------+---------------
//     A0e1     |        _          |  t + o + wA  |    I
// -------------+-------------------+--------------+---------------
//     A0e0     |        _          |  t + o + wA  |    _
// -------------+-------------------+--------------+---------------
//     A073     |        _          |  t + o + i   |  I + O
// -------------+-------------------+--------------+---------------
//     A448     |        t          |      t       |    W
// -------------+-------------------+--------------+---------------
//     A241     |        o          |      t       |    I
// -------------+-------------------+--------------+---------------
//     A000     |        _          |      _       |    _
// -------------+-------------------+--------------+---------------
//     A201     |        o          |      _       |    I
// -------------+-------------------+--------------+---------------
//     A020*    |        _          |      o       |    _
// -------------+-------------------+--------------+---------------
//     A021*    |        _          |      o       |    I
// -------------+-------------------+--------------+---------------
//  *: TODO
//

const unsigned XOPT_MSK = 0xA000;

const unsigned TTM_MSK = 0xF00;
const unsigned TTM_I   = 0x100;
const unsigned TTM_O   = 0x200;
const unsigned TTM_T   = 0x400;

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;
const unsigned FUS_T   = 0x40;
const unsigned FUS_A   = 0x80;

const unsigned DUP_MSK = 0xF;
const unsigned DUP_I   = 0x1;
const unsigned DUP_O   = 0x2;
const unsigned DUP_W   = 0x8;

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_t<Type, A, K, V, I>::elx_conv_wino_t(
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

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n", this->V, this->Ir, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d, oc3=%d, oc4=%d, OC=%d\n", this->V, this->Or, this->O2, this->oc3, this->oc4, this->OC);
}

template <typename Type, const int A, const int K, const int V, const int I>
int  elx_conv_wino_t<Type, A, K, V, I>::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t routput_size = 0, toutputa_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t l1_usage = 0, l2_usage = 0;

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

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING)
      : false;

  if (!(xopt_ & TTM_MSK)) {
    this->nthreads = mthr_;
    this->nteams = 1;
  }
  if (xopt_ & TTM_T) {
    divide_tasks_ttm(this->t2);
  }
  if (xopt_ & TTM_O) {
    if (this->oc3 % this->nteams != 0) {
      // Force single nteams
      this->nthreads = mthr_;
      this->nteams = 1;
    } else {
      // ignore user --pat-o=oc4
      this->oc3 /= this->nteams;
      this->oc4 = this->nteams;
    }
  }
  if (xopt_ & FUS_O) {
    this->oc3 /= this->oc4;
    if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_error("Config error!");
      return -1;
    }
  }
  if (xopt_ & FUS_I) {
    this->ic3 /= this->ic4;
    if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
      el_error("Config error!");
      return -1;
    }
  }

  input_is_bfmt_ = this->input_fmt == nchw ? false : true;
  weights_is_bfmt_ = this->weights_fmt == oihw ? false : true;
  output_is_bfmt_ = this->output_fmt == nchw ? false : true;
  input_as_bfmt_ = !input_is_bfmt_ && this->input_as_blocked;
  weights_as_bfmt_ = !weights_is_bfmt_ && this->weights_as_blocked;
  output_as_bfmt_ = !output_is_bfmt_ && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->ic4 > 1 && this->Ir != V) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V
      && (!output_as_bfmt_ || !weights_as_bfmt_)) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw;
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC * this->kh * this->kw;
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow;

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  toutputa_ = nullptr;
  routput_ = nullptr;
  routput_cntr_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  l1_usage = sizeof(Type)
      * (this->O2 * this->I2 * V * V + this->T * V * (this->I2 + this->O2));
MEMALIGN64(&boutput_, this->n * this->OC * this->oh * this->ow);

  switch (xopt_) {
  case 0xa000:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->t;
    toutput_size = A * A * this->OC * this->t;
    l2_usage = this->IC * this->OC / this->oc3
        + this->T * (this->IC + this->OC / this->oc3);
    break;
  case 0xa040:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->T * mthr_;
    toutput_size = A * A * this->OC * this->T * mthr_;
    l2_usage = tweights_size + A * A * this->T * (this->IC + this->OC);
    break;
  case 0xa061:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->T * mthr_;
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + A * A * this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa073:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_;
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_;
    routput_size = this->n * this->OC * this->oh * this->ow * (this->ic4 - 1);
    routput_cntr_ = (unsigned char *)malloc(this->t2 * this->oc4);
    l2_usage = tweights_size / this->ic4 / this->oc4
        + A * A * this->T * (this->IC / this->ic4 + this->OC / this->oc4);
    break;
  case 0xa0e0:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->t;
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_;
    toutputa_size = A * (A - K + 1) * this->OC * this->t;
    l2_usage = tweights_size / this->oc4 / A
        + A * this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa0e1:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * this->IC * this->T * mthr_;
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_;
    toutputa_size = A * (A - K + 1) * this->OC * this->t;
    l2_usage = tweights_size / this->oc4 / A
        + A * this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa201:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->T * this->t2 * this->nteams;
    toutput_size = A * A * this->OC * this->T * this->t2;
    l2_usage = this->IC * this->OC / this->oc3 / this->oc4
        + this->T * (this->IC + this->OC / this->oc3 / this->oc4);
    break;
  case 0xa241:
    tweights_size = A * A * this->IC * this->OC;
    tinput_size = A * A * this->IC * this->T * mthr_;
    toutput_size = A * A * this->OC * this->T * mthr_;
    l2_usage = tweights_size / this->oc4
        + A * A * this->T * (this->IC + this->OC / this->oc4);
    break;
  case 0xa448:
    tweights_size = A * A * this->IC * this->OC * this->nteams;
    tinput_size = A * A * this->IC * this->T * mthr_;
    toutput_size = A * A * this->OC * this->T * mthr_;
    l2_usage = tweights_size / this->nteams
        + A * A * this->T * (this->IC + this->OC);
    break;
  default:
      el_error("Config error!");
      return -1;
    break;
  }

  l2_usage *= sizeof(Type);

  if (tweights_size > 0)
    MEMALIGN64(&tweights_, tweights_size * sizeof(Type));
  if (tinput_size > 0)
    MEMALIGN64(&tinput_, tinput_size * sizeof(Type));
  if (toutput_size > 0)
    MEMALIGN64(&toutput_, toutput_size * sizeof(Type));
  if (routput_size > 0)
    MEMALIGN64(&routput_, routput_size * sizeof(Type));
  if (toutputa_size > 0)
    MEMALIGN64(&toutputa_, toutputa_size * sizeof(Type));
  if (binput_size > 0)
    MEMALIGN64(&binput_, binput_size * sizeof(Type));
  if (bweights_size > 0)
    MEMALIGN64(&bweights_, bweights_size * sizeof(Type));
  if (boutput_size > 0)
    MEMALIGN64(&boutput_, boutput_size * sizeof(Type));

  // dbg
  printf("nteams=%d, nthreads=%d, mthr_=%d\n", this->nteams, this->nthreads, mthr_);
  printf("l2_usage=%ld, l1_usage=%ld\n", l2_usage, l1_usage);

  return 0;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::bind_execute_functions()
{
  ker_trans_input_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(false))>::trans_input;
  ker_trans_input0_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(true))>::trans_input;
  ker_trans_inputa_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(false))>::trans_inputa;
  ker_trans_inputa0_ = convolution_winograd_kernel<S_INPUT(
      Type, A, K, V, I, BORDER(true))>::trans_inputa;
  ker_trans_weights_ = convolution_winograd_kernel<S_WEIGHTS(
      Type, A, K, V, I)>::trans_weights;
  ker_trans_output_nobias_ = convolution_winograd_kernel<S_OUTPUT(
      Type, A, K, V, I, BORDER(false), BIAS(false))>::trans_output;
  ker_trans_output0_nobias_ = convolution_winograd_kernel<S_OUTPUT(
      Type, A, K, V, I, BORDER(true), BIAS(false))>::trans_output;
  if (this->with_bias) {
    ker_trans_output_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(true))>::trans_output;
    ker_trans_output0_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(true))>::trans_output;
    ker_trans_outputa_bh_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(true))>::trans_outputa_bh;
    ker_trans_outputa0_bh_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(true))>::trans_outputa_bh;
  } else {
    ker_trans_output_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(false))>::trans_output;
    ker_trans_output0_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(false))>::trans_output;
    ker_trans_outputa_bh_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(false), BIAS(false))>::trans_outputa_bh;
    ker_trans_outputa0_bh_ = convolution_winograd_kernel<S_OUTPUT(
        Type, A, K, V, I, BORDER(true), BIAS(false))>::trans_outputa_bh;
  }
  ker_trans_outputa_th_ = convolution_winograd_kernel<S_OUTPUT(
      Type, A, K, V, I, BORDER(false), BIAS(false))>::trans_outputa_th;

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

  if (this->Ir != V) {
#define GEMM_TAIL_CASE(z, n, data)                                           \
  case n:                                                                    \
    ker_gemm_tail_                                                           \
        = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm_tail;     \
    break;

    switch (this->T) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_TAIL_CASE, nil)
    default:
      el_error("Unimplemented");
      break;
    }

#define GEMM_CASE0_TAIL(z, n, data)                                          \
  case n:                                                                    \
    ker_gemm0_tail_                                                          \
        = convolution_winograd_kernel<S_GEMM(Type, n, V, I)>::gemm_tail;     \
    break;

    switch (this->Tr) {
      BOOST_PP_REPEAT_FROM_TO(1, MAX_FMA_PRL, GEMM_CASE0_TAIL, nil);
    default:
      el_error("Unimplemented");
      break;
    }
  } else {
    ker_gemm_tail_ = ker_gemm_;
    ker_gemm0_tail_ = ker_gemm0_;
  }

#define EXECUTE_CASE(n)                                                      \
  case 0x##n:                                                                \
    printf("execute_opt=" #n "\n");                                          \
    execute_opt_ = &elx_conv_wino_t<Type, A, K, V, I>::__execute_##n;        \
    break

  switch (xopt_) {
  EXECUTE_CASE(a241);
  EXECUTE_CASE(a201);
  EXECUTE_CASE(a448);
  EXECUTE_CASE(a040);
  EXECUTE_CASE(a061);
  EXECUTE_CASE(a0e1);
  EXECUTE_CASE(a0e0);
  EXECUTE_CASE(a073);
  EXECUTE_CASE(a000);
  default:
    el_error("Unimplemented");
    break;
  }
}

template <typename Type, const int A, const int K, const int V, const int I>
elx_conv_wino_t<Type, A, K, V, I>::~elx_conv_wino_t()
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
  if (routput_ != nullptr) {
    free(routput_);
    routput_ = nullptr;
  }
  if (routput_cntr_ != nullptr) {
    free(routput_cntr_);
    routput_cntr_ = nullptr;
  }
  if (toutputa_ != nullptr) {
    free(toutputa_);
    toutputa_ = nullptr;
  }
  if (binput_ != nullptr) {
    free(binput_);
    binput_ = nullptr;
  }
  if (bweights_ != nullptr) {
    free(bweights_);
    bweights_ = nullptr;
  }
  if (boutput_ != nullptr) {
    free(boutput_);
    boutput_ = nullptr;
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2 | A, A, ic3, I2, T, V
// toutput:  t2 | A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a040(
    Type *output, Type *input, Type *weights, Type *bias)
{
  MD2(Type, atinput2, tinput_, mthr_, A * A * this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, A * A * this->T * this->OC);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(tweights_, weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1)
    for_each (_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      trans_input(&md2(atinput2, ithr, 0), input, _t2, Tz);
      gemm(&md2(atoutput2, ithr, 0),
           &md2(atinput2, ithr, 0), tweights_, _t2, Tz);
      trans_output(output, &md2(atoutput2, ithr, 0), bias, _t2, Tz);
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2      | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a061(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD2(Type, atinput2, tinput_, mthr_, A * A * this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, A * A * this->T * this->oc3 * this->O2 * V);
  MD2(Type, atweights2, tweights_, this->oc4, A * A * this->IC * this->oc3 * this->O2 * V);

  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(tweights_, weights, this->oc4);
#pragma omp barrier
    }
#pragma omp for nowait collapse(2)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
        size_t ithr = omp_get_thread_num();

        trans_input(&md2(atinput2, ithr, 0), input, _t2, Tz);
        gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0),
            &md2(atweights2, _oc4, 0), _t2, Tz);
        trans_output(&md3(aoutput, 0, _oc4, 0), &md2(atoutput2, ithr, 0),
            &md2(abias, _oc4, 0), _t2, Tz);
      }
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, wA | hA, oc3, ic3, O2, I2, V, V
// tinputa:  t2,      wA | hA, ic3, I2, T, V
// toutput:  t2, oc4, wA | hA, oc3, O2, T, V
// toutputa: t2, oc4, oc3, O2, T, wA, hA, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a0e1(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD2(Type, atinputa2, tinput_, mthr_, A * this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, A * this->T * this->oc3 * this->O2 * V);
  MD2(Type, atoutputa2, toutputa_, this->t2, this->OC * A * (A - K + 1) * this->T);
  MD3(Type, atweights3, tweights_, this->oc4, A, A * this->IC * this->oc3 * this->O2 * V);
  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weightsa(tweights_, weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(3)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        for_each (_wA, A) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          size_t ithr = omp_get_thread_num();

          MD6(Type, atoutputa6, &md2(atoutputa2, _t2, 0), this->oc4, this->oc3, this->O2, Tz, A, (A - K + 1) * V);
          trans_inputa(&md2(atinputa2, ithr, 0), input, _t2, _wA, Tz);
          gemma(&md2(atoutput2, ithr, 0), &md2(atinputa2, ithr, 0),
              &md3(atweights3, _oc4, _wA, 0), _t2, Tz);
          trans_outputa_th(&md6(atoutputa6, _oc4, 0, 0, 0, _wA, 0),
              &md2(atoutput2, ithr, 0), Tz);
        }
      }
    }
#pragma omp barrier
    trans_outputa_bh(output, toutputa_, bias);
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, wA | hA, oc3, ic3, O2, I2, V, V
// tinputa:  t2,      wA | hA, ic3, I2, T, V
// toutput:  t2, oc4, wA | hA, oc3, O2, T, V
// toutputa: t2, oc4, oc3, O2, T, wA, hA, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a0e0(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD2(Type, atinput2, tinput_, this->t2, A * A * this->T * this->IC);
  MD2(Type, atoutput2, toutput_, mthr_, A * this->T * this->oc3 * this->O2 * V);
  MD2(Type, atoutputa2, toutputa_, this->t2, this->OC * A * (A - K + 1) * this->T);
  MD3(Type, atweights3, tweights_, this->oc4, A, A * this->IC * this->oc3 * this->O2 * V);
  MD3(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2 * V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weightsa(tweights_, weights);
    }
    trans_input(tinput_, input);
#pragma omp barrier

#pragma omp for nowait collapse(3)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        for_each (_wA, A) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          size_t ithr = omp_get_thread_num();

          MD6(Type, atoutputa6, &md2(atoutputa2, _t2, 0), this->oc4, this->oc3, this->O2, Tz, A, (A - K + 1) * V);
          MD2(Type, atinputa2, &md2(atinput2, _t2, 0), A, A * Tz * this->IC);
          gemma(&md2(atoutput2, ithr, 0), &md2(atinputa2, _wA, 0),
              &md3(atweights3, _oc4, _wA, 0), _t2, Tz);
          trans_outputa_th(&md6(atoutputa6, _oc4, 0, 0, 0, _wA, 0),
              &md2(atoutput2, ithr, 0), Tz);
        }
      }
    }
#pragma omp barrier
    trans_outputa_bh(output, toutputa_, bias);
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// tweights:     oc4, ic4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2,      ic4 | A, A, ic3, I2, T, V
// toutput:  t2, oc4      | A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a073(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD2(Type, atinput2, tinput_, mthr_, A * A * this->T * this->ic3 * this->I2 * V);
  MD2(Type, atoutput2, toutput_, mthr_, A * A * this->T * this->oc3 * this->O2 * V);
  MD3(Type, atweights3, tweights_, this->oc4, this->ic4, A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(Type, ainput, input, this->n, this->ic4, this->ih * this->iw * this->ic3 * this->I2 * V);
  MD5(Type, aroutput, routput_, this->ic4, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2, V);
  MD4(Type, aoutput, output, this->n, this->oc4, this->oh * this->ow * this->oc3 * this->O2, V);
  MD2(Type, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  bool inline_reduce = (output_is_bfmt_ || output_as_bfmt_) && (this->ic4 > 1)
      && ((this->O2 * this->oc3) % 2 == 0)
      && (mthr_ >= this->t2 * this->oc4 * this->ic4);

  memset(routput_cntr_, 0, this->t2 * this->oc4);

#pragma omp parallel num_threads(mthr_) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(tweights_, weights, this->oc4);
#pragma omp barrier
    }
#pragma omp for nowait collapse(3)
    for_each (_t2, this->t2) {
      for_each (_oc4, this->oc4) {
        for_each (_ic4, this->ic4) {
          int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
          size_t ithr = omp_get_thread_num();
          trans_input(&md2(atinput2, ithr, 0), &md3(ainput, 0, _ic4, 0), _t2, Tz);
          gemm(&md2(atoutput2, ithr, 0), &md2(atinput2, ithr, 0), &md3(atweights3, _oc4, _ic4, 0), _t2, Tz);
          if (_ic4 == 0)
            trans_output(output, output, &md2(atoutput2, ithr, 0), bias, _t2, Tz, _ic4,
                _oc4, inline_reduce);
          else
            trans_output(output, &md5(aroutput, _ic4 - 1, 0, 0, 0, 0), &md2(atoutput2, ithr, 0),
                nullptr, _t2, Tz, _ic4, _oc4, inline_reduce);
        }
      }
    }
    if (!inline_reduce) {
#pragma omp barrier
#pragma omp for nowait collapse(3)
      for (size_t _n = 0; _n < this->n; ++_n) {
        for (size_t _oc4 = 0; _oc4 < this->oc4; ++_oc4) {
          for (size_t _i = 0; _i < this->oc3 * this->O2 * this->oh * this->ow;
               ++_i) {
            for (size_t _ic4 = 1; _ic4 < this->ic4; ++_ic4) {
#pragma omp simd
              for (size_t _V = 0; _V < V; ++_V) {
                md4(aoutput, _n, _oc4, _i, _V)
                    += md5(aroutput, _ic4 - 1, _n, _oc4, _i, _V);
              }
            }
          }
        }
      }
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// Thread-teaming along 't' dimension.
// Fuse trans-input, gemm and trans-output along 't' dimension
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a448(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD3(Type, atinput3, tinput_, this->nteams, this->nthreads, A * A * this->T * this->IC);
  MD3(Type, atoutput3, toutput_, this->nteams, this->nthreads, A * A * this->T * this->OC);
  MD2(Type, atweights2, tweights_, this->nteams, this->OC * this->IC * A * A);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(&md2(atweights2, s, 0), weights);
#pragma omp barrier
    }
#pragma omp for nowait collapse(1) schedule(static)
    for (int _t2 = ttm_[s].start; _t2 <= ttm_[s].end; _t2++) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      trans_input(&md3(atinput3, s, ithr, 0), input, _t2, Tz);
      gemm(&md3(atoutput3, s, ithr, 0), &md3(atinput3, s, ithr, 0),
          &md2(atweights2, s, 0), _t2, Tz);
      trans_output(output, &md3(atoutput3, s, ithr, 0), bias, _t2, Tz);
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

// Flat mode
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a000(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
#pragma omp parallel num_threads(mthr_) proc_bind(close)
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
// tweights: nteams | oc3, ic3, A, A, O2, I2, V, V (oc3 /= nteams)
// tinputs:  nteams | t2, A, A, ic3, I2, T, V (dup)
// toutput:  nteams | t2, A, A, oc3, O2, T, V
template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a201(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD3(Type, aoutput3, output, this->n, this->nteams, this->OC * this->oh * this->ow / this->nteams);
  MD2(Type, aweights2, weights, this->nteams, this->OC * this->IC * K * K / this->nteams);
  MD2(Type, abias2, bias, this->nteams, this->OC / this->nteams);
  MD2(Type, atinput2, tinput_, this->nteams, this->t2 * A * A * this->T * this->IC);
  MD2(Type, atoutput2, toutput_, this->nteams, this->t2 * A * A * this->T * this->OC / this->nteams);
  MD2(Type, atweights2, tweights_, this->nteams, this->OC * this->IC * A * A / this->nteams);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_)
      trans_weights(&md2(atweights2, s, 0), &md2(aweights2, s, 0));
    trans_input(&md2(atinput2, s, 0), input);
#pragma omp barrier
    gemm(&md2(atoutput2, s, 0), &md2(atinput2, s, 0), &md2(atweights2, s, 0));
#pragma omp barrier
    trans_output(
        &md3(aoutput3, 0, s, 0), &md2(atoutput2, s, 0), &md2(abias2, s, 0));
  }
  if (inference_acc_)
    is_first_run_ = false;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::__execute_a241(
    Type * __restrict output, Type * __restrict input, Type * __restrict weights, Type * __restrict bias)
{
  MD3(Type, aoutput3, output, this->n, this->nteams, this->OC * this->oh * this->ow / this->nteams);
  MD2(Type, aweights2, weights, this->nteams, this->OC * this->IC * K * K / this->nteams);
  MD2(Type, abias2, bias, this->nteams, this->OC / this->nteams);
  MD3(Type, atinput3, tinput_, this->nteams, this->nthreads, A * A * this->T * this->IC);
  MD3(Type, atoutput3, toutput_, this->nteams, this->nthreads, A * A * this->T * this->OC / this->nteams);
  MD2(Type, atweights2, tweights_, this->nteams, this->OC * this->IC * A * A / this->nteams);

  omp_set_nested(1);
#pragma omp parallel num_threads(this->nteams) proc_bind(spread)
#pragma omp for nowait collapse(1) schedule(static)
  for (int s = 0; s < this->nteams; s++)
#pragma omp parallel num_threads(this->nthreads) proc_bind(close)
  {
    if (is_first_run_) {
      trans_weights(&md2(atweights2, s, 0), &md2(aweights2, s, 0));
#pragma omp barrier
    }
#pragma omp for nowait collapse(1)
    for_each (_t2, this->t2) {
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      size_t ithr = omp_get_thread_num();

      trans_input(&md3(atinput3, s, ithr, 0), input, _t2, Tz);
      gemm(&md3(atoutput3, s, ithr, 0), &md3(atinput3, s, ithr, 0),
          &md2(atweights2, s, 0), _t2, Tz);
      trans_output(&md3(aoutput3, 0, s, 0), &md3(atoutput3, s, ithr, 0),
          &md2(abias2, s, 0), _t2, Tz);
    }
  }
  if (inference_acc_)
    is_first_run_ = false;
}

template <typename Type, const int A, const int K, const int V, const int I>
void elx_conv_wino_t<Type, A, K, V, I>::execute(
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
