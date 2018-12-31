#include <string.h>
#include <float.h>
#include <math.h>
#include "el_intrin.hpp"
#include "el_utils.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elx_conv_wino.hpp"
#include "euler.hpp"

namespace euler {

const unsigned XOPT_MSK = 0xA000;

const unsigned FUS_MSK = 0xF0;
const unsigned FUS_I   = 0x10;
const unsigned FUS_O   = 0x20;
const unsigned FUS_T   = 0x40;
const unsigned FUS_A   = 0x80;

const unsigned DUP_MSK = 0xF;
const unsigned DUP_I   = 0x1;
const unsigned DUP_O   = 0x2;
const unsigned DUP_W   = 0x8;

const float INT8GEMM_TWT_QTSCALE = 127.0f;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE_T = 127.0f;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE_T4 = 157.0f;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE_T5 = 177.0f;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE_T6 = 197.0f;

Template_elx_conv_wino_t Instance_elx_conv_wino_t::elx_conv_wino_t(
    eld_conv_t<UserTypes> &dc)
    : elx_conv_t<UserTypes>(dc)
{
  if (std::isunordered(this->input_avg, 1.0f) ||
      std::isunordered(this->tinput_min, 1.0f) ||
      std::isunordered(this->tinput_max, 1.0f)) {
    global_minmax_ = false;
  }

  // TODO: error when V!=16 && fmt=OIhw16i16o
  xopt_ = this->execution_mode;

  this->Vx = ((xopt_ & 0xf00) == 0x100) ? 4 : 1;
  this->IC = ALIGNUP(this->ic, V * this->Vx);
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
  if (this->O == 0) this->O = 1; // TODO: O selection
  if (this->O1 == 0) this->O1 = 1; // TODO: O1 selection
  if (this->I2 == 0) this->I2 = 1; // TODO: I2 selection
  if (this->T == 0)  this->T = 1; // TODO: T selection
  this->O2 = this->O * this->O1;

  // Tailing
  this->Tr = this->t % this->T ? this->t % this->T : this->T;
  this->Ir = this->ic % (V * this->Vx)
      ? ALIGNUP(this->ic % (V * this->Vx), this->Vx) / this->Vx
      : (V * this->Vx);
  this->Or = this->oc % V ? this->oc % V : V;

  if (this->Vx == 4 && this->Ir != (V * this->Vx))
    el_error("ic / 64 != 0 is not implement while doing int8 gemm");

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = omp_get_max_threads();
  if (this->nthreads == 0 || this->nthreads > mthr_) {
    this->nthreads = mthr_;
  } else {
    mthr_ = this->nthreads;
  }
  inference_acc_ = this->prop_kind == forward_inference;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  // further divide packed oc/ic
  this->oc3 = this->oc2 / this->O2;
  this->ic3 = this->ic2 / this->I2 / this->Vx;

  this->t2 = (this->t + this->T - 1) / this->T;

  this->tweights_preprocessed_ = false;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("############################################################\n");
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, Vx=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      this->V, this->Ir, this->Vx, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, OC=%d\n",
      this->V, this->Or, this->O2, this->O, this->O1, this->oc3, this->oc4, this->OC);

#ifdef DEBUG
  if (this->Vx * this->V * this->I2 * this->ic3 * this->ic4 != this->IC) {
    el_warn("Vx * V * I2 * ic3 * ic4 != this->IC\n Force ic4 = IC / (Vx * V * I2 * ic3)");
    this->ic4 = this->IC / (this->Vx * this->V * this->I2 * this->ic3);
  }

  if (this->V * this->O2 * this->oc3 * this->oc4 != this->OC) {
    el_warn("V * O2 * oc3 * oc4 != this->OC\n Force oc4 = OC / (V * O2 * oc3)");
    this->oc4 = this->OC / (this->V * this->O2 * this->oc3);
  }
#else
  if ((xopt_ == 0xa073 || xopt_ == 0xa07b || this->with_ip_sum)
      && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (this->Vx * this->V * this->I2 * this->ic3 * this->ic4 != this->IC) {
    el_error("Vx * V * I2 * ic3 * ic4 != this->IC\n)");
  }

  if (this->V * this->O2 * this->oc3 * this->oc4 != this->OC) {
    el_error("V * O2 * oc3 * oc4 != this->OC\n)");
  }
#endif
}

Template_elx_conv_wino_t
int Instance_elx_conv_wino_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t toutputa_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t tinput_u8_size = 0, tinput_qt_scale_size = 0,
      tinput_qt_factor_size = 0, tinput_max_abs_size = 0, tweights_s8_size = 0,
      tweights_qt_scale_size = 0, tweights_qt_factor_size = 0, tweights_ci_size = 0,
      weights_shift_size = 0;

  stream_in_ = this->streaming_input
      ? (this->streaming_input == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_wei_ = this->streaming_weights
      ? (this->streaming_weights == STORE_STREAMING)
      : !(xopt_ & FUS_MSK) ? true : false;
  stream_out_ = this->streaming_output
      ? (this->streaming_output == STORE_STREAMING)
      : false;

  if (xopt_ & FUS_O) {
    this->oc3 /= this->oc4;
    if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
      el_error("Config error!");
      return -1;
    }
  }
  if (xopt_ & FUS_I) {
    this->ic3 /= this->ic4;
    if (V * this->Vx * this->I2 * this->ic3 * this->ic4 != this->IC) {
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

  if (this->ic4 > 1 && this->Ir != V * Vx) {
    el_error("Unimplemented: ic4 > 1 for IC % V != 0");
  }
  if (this->oc4 > 1 && this->Or != V
      && (!output_as_bfmt_ || !weights_as_bfmt_)) {
    el_error("Unimplemented: oc4 > 1 for OC % V != 0");
  }

  if (input_as_bfmt_)
    binput_size = this->n * this->IC * this->ih * this->iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = this->OC * this->IC * this->kh * this->kw * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = this->n * this->OC * this->oh * this->ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  toutputa_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;
  tinput_u8_ = nullptr;
  tinput_qt_scale_ = nullptr;
  tinput_qt_factor_ = nullptr;
  tinput_max_abs_ = nullptr;
  tweights_s8_ = nullptr;
  tweights_qt_scale_ = nullptr;
  tweights_qt_factor_ = nullptr;
  tweights_ci_ = nullptr;
  weights_shift_ = nullptr;

  switch (xopt_) {
  case 0xa000:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * this->IC * this->t * sizeof(TinputType);
    toutput_size = A * A * this->OC * this->t * sizeof(ToutputType);
    break;
  case 0xa033:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * this->ic3 * this->I2 * V * this->t * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->t * sizeof(ToutputType);
    break;
  case 0xa061:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * this->IC * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    weights_shift_size = this->OC * sizeof(TscaleType);
    break;
  case 0xa071:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * this->OC * this->t * sizeof(ToutputType);
    break;
  case 0xa073:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    break;
  case 0xa079:
    tweights_size = A * A * (this->IC / this->ic4) * (this->OC / this->oc4) * mthr_ * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * this->OC * this->t * sizeof(ToutputType);
    break;
  case 0xa07b:
    tweights_size = A * A * (this->IC / this->ic4) * (this->OC / this->oc4) * mthr_ * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    break;
  case 0xa0e0:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * this->IC * this->t * sizeof(TinputType);
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    toutputa_size = A * (A - K + 1) * this->OC * this->t * sizeof(TrOpType);
    break;
  case 0xa0e1:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * this->IC * this->T * mthr_ * sizeof(TinputType);
    toutput_size = A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    toutputa_size = A * (A - K + 1) * this->OC * this->t * sizeof(TrOpType);
    break;
  case 0xa133:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->t * sizeof(InputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->t * sizeof(ToutputType);
    tinput_u8_size = A * A * (this->IC / this->ic4) * this->t * sizeof(uint8_t);
    tinput_qt_scale_size = this->t * this->ic3 * 2 * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_qt_scale_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    tweights_qt_factor_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    break;
  case 0xa161:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * this->I2 * this->Vx * V * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    tinput_u8_size = A * A * this->IC * mthr_ * this->T * sizeof(uint8_t);
    tinput_qt_scale_size = mthr_ * 2 * this->ic3 * this->T * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_qt_scale_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    tweights_qt_factor_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType); // * this->ic4
    tweights_ci_size = this->OC * sizeof(TscaleType);
    weights_shift_size = this->OC * sizeof(TscaleType);
    break;
  case 0xa173:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    tinput_u8_size = A * A * (this->IC / this->ic4) * mthr_ * this->T * sizeof(uint8_t);
    tinput_qt_scale_size = mthr_ * 2 * this->ic3 * this->T * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_qt_scale_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    tweights_qt_factor_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    break;
  default:
      el_error("Config error!");
      return -1;
    break;
  }

  // TODO change align for different types
#define WEIGHTS_MAX_PRELOAD 4 * sizeof(TweightsType)
  const size_t align = PAGE_SIZE;
  if (tweights_size > 0)
    tweights_size += WEIGHTS_MAX_PRELOAD * V;

  tweights_size_ = tweights_size > 0 ? alignup(tweights_size, align) : 0;
  tinput_size_ = tinput_size > 0 ? alignup(tinput_size, align) : 0;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  toutputa_size_ = toutputa_size > 0 ? alignup(toutputa_size, align) : 0;
  binput_size_ = binput_size > 0 ? alignup(binput_size, align) : 0;
  bweights_size_ = bweights_size > 0 ? alignup(bweights_size, align) : 0;
  boutput_size_ = boutput_size > 0 ? alignup(boutput_size, align) : 0;
  tinput_u8_size_ = tinput_u8_size > 0 ? alignup(tinput_u8_size, align) : 0;
  tinput_qt_scale_size_ = tinput_qt_scale_size > 0 ? alignup(tinput_qt_scale_size, align) : 0;
  tinput_qt_factor_size_ = tinput_qt_factor_size > 0 ? alignup(tinput_qt_factor_size, align) : 0;
  tinput_max_abs_size_ = tinput_max_abs_size > 0 ? alignup(tinput_max_abs_size, align) : 0;
  tweights_s8_size_ = tweights_s8_size > 0 ? alignup(tweights_s8_size, align) : 0;
  tweights_qt_scale_size_ = tweights_qt_scale_size > 0 ? alignup(tweights_qt_scale_size, align) : 0;
  tweights_qt_factor_size_ = tweights_qt_factor_size > 0 ? alignup(tweights_qt_factor_size, align) : 0;
  tweights_ci_size_ = tweights_ci_size > 0 ? alignup(tweights_ci_size, align) : 0;
  weights_shift_size_ = weights_shift_size > 0 ? alignup(weights_shift_size, align) : 0;

  workspace_ = nullptr, scratch_ = nullptr;
  size_t workspace_size = tweights_size_ + tweights_s8_size_
      + tweights_qt_scale_size_ + tweights_qt_factor_size_ + tweights_ci_size_ + weights_shift_size;
  size_t scratch_size = tinput_size_ + toutput_size_ + toutputa_size_
      + binput_size_ + bweights_size_ + boutput_size_ + tinput_u8_size_
      + tinput_qt_scale_size_ + tinput_qt_factor_size_ + tinput_max_abs_size_;

  if (xopt_ == 0xa079 || xopt_ == 0xa07b) {
    scratch_size += tweights_size_;
    workspace_size = 0;
  }
  // TODO: user provided buffer
  if (scratch_size != 0)
    scratch_ = galloc::acquire(scratch_size);
  if (workspace_size != 0)
    MEMALIGN64(&workspace_, workspace_size);

  // dbg
  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  printf("gemmker_input_footprint = %ld\n", gemmker_input_footprint());
  printf("gemmker_weights_footprint = %ld\n", gemmker_weights_footprint());
  printf("gemmker_output_footprint = %ld\n", gemmker_output_footprint());
  printf("gemm_input_reuse_set = %ld\n", gemm_input_reuse_set());
  printf("gemm_output_reuse_set = %ld\n", gemm_output_reuse_set());

  auto plan = execute_plan(this->nthreads, 1, 1024 * 1024, 32 * 1024);
  plan.dump();

  return 0;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::set_trans_buffers()
{
  if (workspace_ != nullptr) {
    tweights_ = (TweightsType *)workspace_;
    tinput_ = (TinputType *)galloc::get();
    // int8gemm supported in weights reuse case only.
    tweights_qt_scale_ = (TscaleType *)((char *)tweights_ + tweights_size_);
    tweights_qt_factor_ = (TscaleType *)((char *)tweights_qt_scale_ + tweights_qt_scale_size_);
    tweights_ci_ = (TscaleType *)((char *)tweights_qt_factor_ + tweights_qt_factor_size_);
    tweights_s8_ = (int8_t *)((char *)tweights_ci_ + tweights_ci_size_);
    weights_shift_ = (TscaleType *)((char *)tweights_s8_ + tweights_s8_size_);
  } else {
    tweights_ = (TweightsType *)galloc::get();
    tinput_ = (TinputType *)((char *)tweights_ + tweights_size_);
  }
  toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
  toutputa_ = (TrOpType *)((char *)toutput_ + toutput_size_);
  binput_ = (InputType *)((char *)toutputa_ + toutputa_size_);
  bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
  boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
  tinput_qt_scale_ = (TscaleType *)((char *)boutput_ + boutput_size_);
  tinput_qt_factor_ = (TscaleType *)((char *)tinput_qt_scale_ + tinput_qt_scale_size_);
  tinput_max_abs_ = (TscaleType *)((char *)tinput_qt_factor_ + tinput_qt_factor_size_);
  tinput_u8_ = (uint8_t *)((char *)tinput_max_abs_ + tinput_max_abs_size_);
}

Template_elx_conv_wino_t
Instance_elx_conv_wino_t::~elx_conv_wino_t()
{
  if (workspace_ != nullptr)
    ::free(workspace_);

  galloc::release();
}

#define t2spato(__t2, __T, __n, __oh, __ow, __hOA_end, __wOA_end)            \
  do {                                                                       \
    int _t = __t2 * this->T + __T;                                           \
    int _nt = _t % this->nt;                                                 \
    int _ht = _nt / this->wt;                                                \
    int _wt = _nt % this->wt;                                                \
    __n = _t / this->nt;                                                     \
    __oh = _ht * (A - K + 1);                                                \
    __ow = _wt * (A - K + 1);                                                \
    __hOA_end = (_ht < this->ht - 1) ? A - K : hOA_end_;                     \
    __wOA_end = (_wt < this->wt - 1) ? A - K : wOA_end_;                     \
  } while (0)

#define t2spati(                                                             \
    __t2, __T, __n, __ih, __iw, __hA_start, __hA_end, __wA_start, __wA_end)  \
  do {                                                                       \
    int _t = __t2 * this->T + __T;                                           \
    int _nt = _t % this->nt;                                                 \
    int _ht = _nt / this->wt;                                                \
    int _wt = _nt % this->wt;                                                \
    __n = _t / this->nt;                                                     \
    __ih = _ht * (A - K + 1) - this->tp;                                     \
    __iw = _wt * (A - K + 1) - this->lp;                                     \
    __hA_start = (_ht > 0) ? 0 : this->tp;                                   \
    __wA_start = (_wt > 0) ? 0 : this->lp;                                   \
    __hA_end = (_ht < this->ht - 1) ? A - 1 : hA_end_;                       \
    __wA_end = (_wt < this->wt - 1) ? A - 1 : wA_end_;                       \
  } while (0)

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_plain(
    TweightsType * __restrict tweights, WeightsType * __restrict weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD11(WeightsType, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V,
      this->ic4, this->ic3, this->I2, V, K, K);
  MD11(TweightsType, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
        constexpr auto scale = sizeof(WeightsType);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, this->oc, this->ic, K, K);

    assert(this->ic4 == 1 && this->oc4 == 1);
    int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
    int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;
    int iV = is_Ir ? this->Ir : V;

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          constexpr auto scale = sizeof(WeightsType);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md4(awei, _oc2 * V, _ic2 * V + _iV, _hK, _wK), scale);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            ain[_hK][_wK][_iV][_oV]
                = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
          }
        }
      }}}
    }
  };

#pragma omp for nowait collapse(7) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1
        && _O == this->O - 1;

    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) TrOpType aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
        && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
       && std::is_same<TweightsType, float16>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&aout[_wA][_hA][_iV][0],
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::stream_si256((__m256i *)&md11(atweights, _oc4, _ic4, _oc3,
                               _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);

        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&aout[_wA][_hA][_iV][0],
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::store_si256((__m256i *)&md11(atweights, _oc4, _ic4, _oc3,
                               _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_blocked(
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD11(WeightsType, aweights, weights, oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2 * this->Vx, K, K, V, V);
  MD11(TweightsType, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2 * this->Vx, V, this->O, V);

#pragma omp for nowait collapse(7) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2 * this->Vx) {
  iter_each (_O, this->O) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
        && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
       && std::is_same<TweightsType, float16>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&aout[_wA][_hA][_iV][0],
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::stream_si256((__m256i *)&md11(atweights, _oc4, _ic4, _oc3,
                               _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);

        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&aout[_wA][_hA][_iV][0],
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::store_si256((__m256i *)&md11(atweights, _oc4, _ic4, _oc3,
                               _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V)
          md11(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
              = aout[_wA][_hA][_iV][_oV];
      }}}
    }
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_s8_blocked(
    TscaleType *tweights_qt_scale, TscaleType *tweights_qt_factor, int8_t *tweights_s8,
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
  MD12(WeightsType, aweights, weights, oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2, this->Vx, K, K, V, V);
  MD12(TweightsType, atweights, tweights, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2, this->Vx, V, this->O, V);
  MD12(int8_t, atweights_s8, tweights_s8, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2, V, this->O, V, this->Vx);
  MD9(TscaleType, atweights_qt_scale, tweights_qt_scale, oc4, this->ic4, this->oc3, this->ic3, A, A,
      this->O1, this->O, V);
  MD9(TscaleType, atweights_qt_factor, tweights_qt_factor, oc4, this->ic4, this->oc3, this->ic3, A, A,
      this->O1, this->O, this->V);

  __m<V> zero = _mm<V>::set1_ps(0.0);
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);

#pragma omp for nowait collapse(8) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
  iter_each (_iVx, this->Vx) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md12(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, _iVx, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iVx, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iVx, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V) {
          md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iVx, _iV, _O, _oV)
              = aout[_wA][_hA][_iV][_oV];
        }
      }}}
    }
   }}}}}}}}
#pragma omp barrier

  // MD5(TscaleType, atweights_ci, tweights_ci_, oc4, this->oc3, this->O1, this->O, V);
  if (this->tweights_preprocessed_) {
#if 0
#pragma omp for nowait collapse(4) schedule(static)
    iter_each (_oc4, oc4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_O1, this->O1) {
    iter_each (_O, this->O) {
      _mm512_store_ps(&md5(atweights_qt_scale, _oc4, _oc3, _O1, _O, 0),
          *(__m<V> *)&md5(atweights_ci, _oc4, _oc3, _O1, _O, 0));
    }}}}
#pragma omp barrier
#endif
  } else {
#pragma omp for nowait collapse(8) schedule(static)
    iter_each (_oc4, oc4) {
    iter_each (_ic4, this->ic4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
    iter_each (_O1, this->O1) {
    iter_each (_O, this->O) {
      __m<V> mmax_cur = _mm<V>::set1_ps(0.0);
      iter_each (_I2, this->I2) {
      iter_each (_iV, V) {
      iter_each (_iVx, this->Vx) {
        __m<V> mmax_abs;
        TweightsType *max_abs = (TweightsType *)&mmax_abs;
#pragma omp simd
        iter_each (_oV, V) {
          max_abs[_oV] =
              md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iVx, _iV, _O, _oV) >= 0.0 ?
              md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iVx, _iV, _O, _oV) :
              -md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iVx, _iV, _O, _oV);
        }
        mmax_cur = _mm<V>::max_ps(mmax_cur, mmax_abs);
        #if 0
        if (this->tweights_preprocessed_) {
          mmax_cur = _mm<V>::min_ps(
              mmax_cur, *(__m<V> *)&md5(atweights_ci, _oc4, _oc3, _O1, _O, 0));
        }
        #endif
      }}}

      _mm512_store_ps(&md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0), mmax_cur);
    }}}}}}}}
#pragma omp barrier
  }

#ifdef DEBUG
  if (omp_get_thread_num() == 0) {
    iter_each (_oc4, oc4) {
    iter_each (_ic4, this->ic4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
    iter_each (_O1, this->O1) {
    iter_each (_O, this->O) {
    iter_each (_oV, V) {
      printf("max abs: %f\n", md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, _oV));
    }}}}}}}}
  }
#endif

#pragma omp for nowait collapse(8) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_O1, this->O1) {
  iter_each (_O, this->O) {
    __m<V> mmfactor = _mm<V>::set1_ps(0.0);
    iter_each (_I2, this->I2) {
    iter_each (_iV, V) {
    iter_each (_iVx, this->Vx) {
      mmfactor = _mm<V>::add_ps(
          *(__m<V> *)&md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
           _O1, _I2, _iVx, _iV, _O, 0), mmfactor);
    }}}
    _mm512_store_ps(&md9(atweights_qt_factor, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0), mmfactor);
  }}}}}}}}

  // I2 Vx V => I2 V Vx
  MD12(TweightsType, _atweights, tweights, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2, V, this->Vx, this->O, V);
#pragma omp for nowait collapse(11) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_iV, V) {
  iter_each (_O, this->O) {
  iter_each (_iVx, this->Vx) {
    __m<V> t0;
    if (this->tweights_preprocessed_) {
    #if 0
      // saturate
      t0 = _mm<V>::min_ps(
          *(__m<V> *)&md12(_atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
          _O1, _I2, _iV, _iVx, _O, 0),
          *(__m<V> *)&md7(atweights_qt_scale, _oc4, _oc3, _wA, _hA, _O1, _O, 0));
      // multi scal
      t0 = _mm<V>::mul_ps(t0, mmscale);
    #endif
    } else {
      // multi scal
      t0 = _mm<V>::mul_ps(
          *(__m<V> *)&md12(_atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
          _O1, _I2, _iV, _iVx, _O, 0), mmscale);
    }

    t0 = _mm<V>::div_ps(
        t0, *(__m<V> *)&md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0));
    // rounding
    t0 = _mm<V>::roundscale_ps(t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // int8_t
    TweightsType *rounded = (TweightsType *)&t0;
#pragma omp simd
    iter_each (_oV, V) {
      md12(atweights_s8,
          _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV, _iVx) =
          (int8_t)rounded[_oV];
    }
  }}}}}}}}}}}
#pragma omp barrier

#ifdef DEBUG
  static bool debug = true;
#pragma omp for nowait collapse(10) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_iV, V) {
  iter_each (_O, this->O) {
  iter_each (_iVx, this->Vx) {
    if (omp_get_thread_num() == 24 && _iVx == 1 && debug) {
      iter_each (_oV, V) {
        printf("[%d][%d][%d][%d][%d][%d][%d][%d][%d][%d][%d][%d] %f * %f / %f = %d\n",
            _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _iVx, _oV,
            (float)md12(_atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _iVx, _O, _oV), *(float *)&mmscale,
            (float)md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, _oV),
            (int)md12(atweights_s8, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV, _iVx));
        debug = false;
      }
    }
  }}}}}}}}}}}
#endif

#pragma omp for nowait collapse(8) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_O1, this->O1) {
  iter_each (_O, this->O) {
    if (I == ISA_SKX_AVX512 && std::is_same<TscaleType, float>::value) {
      _mm512_store_ps(&md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0),
          _mm<V>::div_ps(
          *(__m<V> *)&md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0), mmscale));
    } else {
#pragma omp simd
      iter_each (_oV, V)
        md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, _oV) =
            md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, _oV) / INT8GEMM_TWT_QTSCALE;
    }
  }}}}}}}}

  // weights shift
  MD6(WeightsType, aweights6, weights, this->oc2, this->ic2, this->kh, this->kw, V, V);
  MD2(TscaleType, aweights_shift, weights_shift_, this->oc2, V);
#pragma omp for nowait schedule(static)
  iter_each (_oc2, this->oc2) {
    __m<V> s = _mm<V>::setzero_ps();
    __m<V> S = _mm<V>::set1_ps(this->input_avg);
    iter_each (_ic2, this->ic2) {
    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_V, V) {
      s += *(__m<V> *)&md6(aweights6, _oc2, _ic2, _hK, _wK, _V, 0);
    }}}}
    s *= S;
    _mm512_store_ps(&md2(aweights_shift, _oc2, 0), s);
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weights(
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights, oc4);
  else
    __trans_weights_plain(tweights, weights, oc4);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weights_s8(
    TscaleType *tweights_qt_scale, TscaleType *tweights_qt_factor, int8_t *tweights_s8,
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_s8_blocked(tweights_qt_scale, tweights_qt_factor,
        tweights_s8, tweights, weights, oc4);
  else
    el_error("Unimplemented: plain format weights for int8");
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsf_plain(
    TweightsType * __restrict tweights, WeightsType * __restrict weights, int _ic4, int _oc4)
{
  MD11(WeightsType, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V,
      this->ic4, this->ic3, this->I2, V, K, K);
  MD9(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
        constexpr auto scale = sizeof(WeightsType);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, this->oc, this->ic, K, K);

    assert(this->ic4 == 1 && this->oc4 == 1);
    int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
    int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;
    int iV = is_Ir ? this->Ir : V;

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          constexpr auto scale = sizeof(WeightsType);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md4(awei, _oc2 * V, _ic2 * V + _iV, _hK, _wK), scale);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            ain[_hK][_wK][_iV][_oV]
                = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
          }
        }
      }}}
    }
  };

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1
        && _O == this->O - 1;

    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) TrOpType aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights_v, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md9(atweights, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsf_blocked(
    TweightsType *tweights, WeightsType *weights, int _ic4, int _oc4)
{
  MD11(WeightsType, aweights, weights, oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2, K, K, V, V);
  MD9(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O1, this->I2, V, this->O, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V)
          md9(atweights, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
              = aout[_wA][_hA][_iV][_oV];
      }}}
    }
  }}}}};
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weightsf(
    TweightsType *tweights, WeightsType *weights, int _ic4, int _oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weightsf_blocked(tweights, weights, _ic4, _oc4);
  else
    __trans_weightsf_plain(tweights, weights, _ic4, _oc4);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsa_blocked(
    TweightsType *tweights, WeightsType *weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD11(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2, K, K, V, V);
  MD11(TweightsType, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3,
      this->ic3, this->O1, this->I2, V, this->O, V);

#pragma omp for nowait collapse(7) schedule(static)
  iter_each (_oc4, this->oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);

    if (I == ISA_SKX_AVX512 && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsa_plain(
    TweightsType * __restrict tweights, WeightsType * __restrict weights)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, wA, hA, oc3, ic3, O2, I2, V, V
  MD11(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O, V,
      this->ic4, this->ic3, this->I2, V, K, K);
  MD11(TweightsType, atweights, tweights, this->oc4, this->ic4, A, A, this->oc3,
      this->ic3, this->O1, this->I2, V, this->O, V);

  SET_EPI32(this->ic * this->kh * this->kw)

  auto readin_v = [&](WeightsType ain[K][K][V][V], WeightsType *wei) {
    MD5(WeightsType, awei, wei, V, this->ic2, V, K, K);

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, V) {
      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
        constexpr auto scale = sizeof(WeightsType);
        auto t = _mm<V>::i32gather_ps(vindex,
            &md5(awei, 0, 0, _iV, _hK, _wK), scale);
        _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
      } else {
        iter_each (_oV, V)
          ain[_hK][_wK][_iV][_oV] = md5(awei, _oV, 0, _iV, _hK, _wK);
      }
    }}}
  };

  auto readin_r = [&](WeightsType ain[K][K][V][V], int _oc4, int _oc3, int _O2,
                      int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, awei, weights, this->oc, this->ic, K, K);

    int _oc2 = _oc4 * this->oc3 * this->O2 + _oc3 * this->O2 + _O2;
    int _ic2 = _ic4 * this->ic3 * this->I2 + _ic3 * this->I2 + _I2;
    int iV = is_Ir ? this->Ir : V;

    if (is_Or) {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
#pragma omp simd
      iter_each (_oV, this->Or) {
        ain[_hK][_wK][_iV][_oV]
            = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
      }}}}
    } else {
      iter_each (_hK, K) {
      iter_each (_wK, K) {
      iter_each (_iV, iV) {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          constexpr auto scale = sizeof(WeightsType);
          auto t = _mm<V>::i32gather_ps(vindex,
              &md4(awei, _oc2 * V, _ic2 * V + _iV, _hK, _wK), scale);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V) {
            ain[_hK][_wK][_iV][_oV]
                = md4(awei, _oc2 * V + _oV, _ic2 * V + _iV, _hK, _wK);
          }
        }
      }}}
    }
  };

#pragma omp for nowait collapse(7) schedule(static)
  iter_each (_oc4, this->oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {

    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1
        && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
        && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1
        && _O == this->O - 1;

    alignas(64) WeightsType ain[K][K][V][V];
    alignas(64) TrOpType aout[A][A][V][V];

    if (this->Ir != V || is_Ir || is_Or)
      readin_r(ain, _oc4, _oc3, _O1 * this->O + _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    else
      readin_v(
          ain, &md11(aweights, _oc4, _oc3, _O1, _O, 0, _ic4, _ic3, _I2, 0, 0, 0));

    ker_trans_weights_(aout, ain);

    if (I == ISA_SKX_AVX512 && std::is_same<TweightsType, float>::value) {
      if (stream_wei_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm<V>::stream_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
        iter_each (_iV, V) {
          _mm512_store_ps(&md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3,
              _O1, _I2, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
        }}}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
      iter_each (_oV, V) {
        md11(atweights, _oc4, _ic4, _wA, _hA, _oc3, _ic3, _O1, _I2, _iV, _O, _oV)
            = aout[_wA][_hA][_iV][_oV];
      }}}}
    }
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weightsa(
    TweightsType *tweights, WeightsType *weights)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weightsa_blocked(tweights, weights);
  else
    __trans_weightsa_plain(tweights, weights);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_plain(
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int Tz)
{
  // n, IC, ih, iw => t2 | wA, hA, ic3, I2, T, V
  MD6(TinputType, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) TrOpType aout[A][A][V];
  alignas(64) InputType ain[A][A][V];
  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2, V,
        this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (_hA, A) {
    iter_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V)
          ain[_hA][_wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm<V>::store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD4(InputType, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    assert(this->ic4 == 1);
    if (is_Ir) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
#pragma omp simd
          iter_each (_v, this->Ir)
            ain[_hA][_wA][_v] = md4(ainput, _n,
                (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + _hA, _iw + _wA),
                scale);
            _mm<V>::store_ps(ain[_hA][_wA], t);
          } else {
#pragma omp simd
            iter_each (_v, V)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }}
    }
  };

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    if (this->Ir != V) {
      readin_r(_ic3, _I2, _T, ain);
    } else
      readin_v(_ic3, _I2, _T, ain);

    ker_trans_input_(*this, aout, (InputType *)ain, 0, 0, 0, -1);

    if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
      if (stream_in_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        md6(atinput, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
      }}}
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_blocked(
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int Tz)
{
  // n, ic2, ih, iw, V => t2 | wA, hA, ic3, I2, T, V
  MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2,
      this->ih, this->iw, V);
  MD6(TinputType, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);

  alignas(64) TrOpType aout[A][A][V];

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt,
      this->ih, this->iw, this->tp, this->lp);
  iter_each (_T, Tz) {
    auto _ih = t2spati_o.anchor_t_;
    auto _iw = t2spati_o.anchor_l_;

    InputType *in = &md7(ainput, t2spati_o.n_, 0, _ic3, _I2, _ih, _iw, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*this, aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

    ++ t2spati_o;

    if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
        && std::is_same<TinputType, float>::value) {
      if (stream_in_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::stream_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          _mm<V>::store_ps(&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_wA][_hA][0]));
        }}
      }
    } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
       && std::is_same<TinputType, float16>::value) {
      if (stream_in_) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&aout[_wA][_hA][0]),
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::stream_si256(
              (__m256i *)&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0), fp16v);
        }}
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&aout[_wA][_hA][0]),
              _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V/2>::store_si256(
              (__m256i *)&md6(atinput, _wA, _hA, _ic3, _I2, _T, 0), fp16v);
        }}
      }
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        md6(atinput, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
      }}}
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_u8_blocked(
    TscaleType * tinput_qt_scale, uint8_t * __restrict tinput_u8,
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int Tz)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

  MD8(InputType, ainput, input, this->n,
      this->ic4, this->ic3, this->I2, this->Vx, this->ih, this->iw, V);
  // 4i,V temporarily here for store AVX instruction
  MD5(TinputType, atinput, tinput, this->I2, this->Vx, A, A, V);
  MD7(uint8_t, atinput_u8, tinput_u8, A, A, this->ic3, this->I2, Tz, this->Vx, V);
  MD5(TscaleType, atinput_qt_scale, tinput_qt_scale, this->ic3, A, A, 2, Tz);

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  if (global_minmax_) {
    auto INT8GEMM_TIN_MIN_MAX_QTSCALE =
        A == 4 ? INT8GEMM_TIN_MIN_MAX_QTSCALE_T4
               : A == 5 ? INT8GEMM_TIN_MIN_MAX_QTSCALE_T5
                        : INT8GEMM_TIN_MIN_MAX_QTSCALE_T6;
printf("QT=%f\n", INT8GEMM_TIN_MIN_MAX_QTSCALE);
    float max = this->tinput_max, min = this->tinput_min;
    float fscale = (max - min) / INT8GEMM_TIN_MIN_MAX_QTSCALE;
    auto mmin = _mm<V>::set1_ps(min);
    auto mscale = _mm<V>::set1_ps(INT8GEMM_TIN_MIN_MAX_QTSCALE / (max - min));
    alignas(64) TrOpType aout[A][A][V];

    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_Vx, this->Vx) {
    input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt,
        this->ih, this->iw, this->tp, this->lp);
    iter_each (_T, Tz) {
      auto _ih = t2spati_o.anchor_t_;
      auto _iw = t2spati_o.anchor_l_;

      InputType *in = &md8(ainput, t2spati_o.n_, 0, _ic3, _I2, _Vx, _ih, _iw, 0);
      if (!t2spati_o.is_border())
        ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(*this, aout, in,
            t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

      ++ t2spati_o;

      if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
          && std::is_same<TinputType, float>::value) {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
          // quant
          auto f32 = *((__m<V> *)&aout[_wA][_hA][0]);
          f32 = (f32 - mmin) * mscale;
          // convert to u8
          __i<V> u32 = _mm<V>::cvt_roundps_epu32(f32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i u8 = _mm<V>::cvtusepi32_epi8(u32);
          // store
          _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), u8);
        }}
      }
    }}}}

    // TODO:
    iter_each (_ic3, this->ic3) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
    iter_each (_T, Tz) {
      md5(atinput_qt_scale, _ic3, _wA, _hA, 0, _T) = fscale;
      md5(atinput_qt_scale, _ic3, _wA, _hA, 1, _T) = min;
    }}}}

    return;
  }

  iter_each(_ic3, this->ic3) {
    input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt, this->ih,
                                    this->iw, this->tp, this->lp);
    iter_each (_T, Tz) {
      auto INT8GEMM_TIN_MIN_MAX_QTSCALE = INT8GEMM_TIN_MIN_MAX_QTSCALE_T;
      alignas(64) TinputType mmax[A][A][V];
      alignas(64) TinputType mmin[A][A][V];
      bool flush = true;
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
        auto _ih = t2spati_o.anchor_t_;
        auto _iw = t2spati_o.anchor_l_;

        MD3(TinputType, aout, &md5(atinput, _I2, _Vx, 0, 0, 0), A, A, V);
        using Array = TrOpType[A][A][V];
        InputType *in = &md8(ainput, t2spati_o.n_, 0, _ic3, _I2, _Vx, _ih, _iw, 0);
        if (!t2spati_o.is_border())
          ker_trans_input_(*this, *(Array *)&md3(aout, 0, 0, 0), in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(*this, *(Array *)&md3(aout, 0, 0, 0), in,
              t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

        if (flush) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            __m<V> &_mmax = *(__m<V> *)&mmax[_wA][_hA][0];
            _mmax = *(__m<V> *)&md3(aout, _wA, _hA, 0);
            __m<V> &_mmin = *(__m<V> *)&mmin[_wA][_hA][0];
            _mmin = *(__m<V> *)&md3(aout, _wA, _hA, 0);
          }}
          flush = false;
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            __m<V> &_mmax = *(__m<V> *)&mmax[_wA][_hA][0];
            _mmax = _mm<V>::max_ps(_mmax, *(__m<V> *)&md3(aout, _wA, _hA, 0));
            __m<V> &_mmin = *(__m<V> *)&mmin[_wA][_hA][0];
            _mmin = _mm<V>::min_ps(_mmin, *(__m<V> *)&md3(aout, _wA, _hA, 0));
          }}
        }
      }}

      iter_each (_wA, A) {
      iter_each (_hA, A) {
        if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
          mmax[_wA][_hA][0] = _mm<V>::reduce_max_ps(*(__m<V> *)&mmax[_wA][_hA][0]);
          mmin[_wA][_hA][0] = _mm<V>::reduce_min_ps(*(__m<V> *)&mmin[_wA][_hA][0]);
        } else {
          for (int _V = 1; _V < V; _V++) {
            mmax[_wA][_hA][0] =
                mmax[_wA][_hA][_V] > mmax[_wA][_hA][0] ?
                mmax[_wA][_hA][_V] : mmax[_wA][_hA][0];
            mmin[_wA][_hA][0] =
                mmin[_wA][_hA][_V] < mmin[_wA][_hA][0] ?
                mmin[_wA][_hA][_V] : mmin[_wA][_hA][0];
          }
        }
        mmax[_wA][_hA][0] = mmax[_wA][_hA][0] - mmin[_wA][_hA][0] + 0.000001;
        md5(atinput_qt_scale, _ic3, _wA, _hA, 0, _T) =
            mmax[_wA][_hA][0] * (1.0f / INT8GEMM_TIN_MIN_MAX_QTSCALE);
        mmax[_wA][_hA][0] = INT8GEMM_TIN_MIN_MAX_QTSCALE / mmax[_wA][_hA][0];
        md5(atinput_qt_scale, _ic3, _wA, _hA, 1, _T) = mmin[_wA][_hA][0];
      }}

      // quantization
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        // Min-Max quantization
        __m<V> mdifff32 = _mm<V>::set1_ps(mmax[_wA][_hA][0]);
        __m<V> mminf32 = _mm<V>::set1_ps(mmin[_wA][_hA][0]);
        __m<V> f = *(__m<V> *)&md5(atinput, _I2, _Vx, _wA, _hA, 0);
        __m<V> mmresf32 = (f - mminf32) * mdifff32;
        // convert to uint8
        __i<V> mmresu32 = _mm<V>::cvt_roundps_epu32(mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mmresu32);
        // store
        _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), mmresu8);
      }}}}
      ++ t2spati_o;
    }
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_input(
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input, _t2, Tz);
  else
    __trans_input_plain(tinput, input, _t2, Tz);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_input_u8(
    TscaleType * tinput_qt_scale, uint8_t * __restrict tinput_u8,
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int Tz)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_u8_blocked(
        tinput_qt_scale, tinput_u8, tinput, input, _t2, Tz);
  else
    el_error("Unimplemented: plain format input for int8");
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_blocked(
    TinputType * __restrict tinput, InputType * __restrict input)
{
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2 * this->Vx,
      this->ih, this->iw, V);
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * this->Vx * V);

  // ICC-19 bug, build crash in case of t2 first
#pragma omp for nowait collapse(3)
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2 * this->Vx) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2 * this->Vx, Tz, V);
    alignas(64) TrOpType aout[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      InputType *in = &md7(ainput, _n, 0, _ic3, _I2, _ih, _iw, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);

      if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
          && std::is_same<TinputType, float>::value) {
        if (stream_in_) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                           *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                          *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        }
      } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
         && std::is_same<TinputType, float16>::value) {
        if (stream_in_) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&aout[_wA][_hA][0]),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm<V/2>::stream_si256(
                (__m256i *)&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0), fp16v);
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&aout[_wA][_hA][0]),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm<V/2>::store_si256(
                (__m256i *)&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0), fp16v);
          }}
        }
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
#pragma omp simd
        iter_each (_V, V) {
          md6(atinput6, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
        }}}
      }
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_plain(
    TinputType * __restrict tinput, InputType * __restrict input)
{
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);

  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _t2, int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2, V,
        this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (_hA, A) {
    iter_each (_wA, A) {
      if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
          || _wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V) {
          ain[_hA][_wA][_V] = 0.0f;
        }
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
              scale);
          _mm<V>::store_ps(ain[_hA][_wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _t2, int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD4(InputType, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    assert(this->ic4 == 1);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    if (is_Ir) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
#pragma omp simd
          iter_each (_v, this->Ir)
            ain[_hA][_wA][_v] = md4(ainput, _n,
                (this->ic2 - 1) * V + _v, _ih + _hA, _iw + _wA);
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[_hA][_wA][_V] = 0.0f;
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + _hA, _iw + _wA),
                scale);
            _mm<V>::store_ps(ain[_hA][_wA], t);
          } else {
#pragma omp simd
            iter_each (_v, V)
              ain[_hA][_wA][_v] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _v, _ih + _hA, _iw + _wA);
          }
        }
      }}
    }
  };

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    alignas(64) TrOpType aout[A][A][V];
    alignas(64) InputType ain[A][A][V];

    iter_each (_T, Tz) {
      if (this->Ir != V)
        readin_r(_t2, _ic3, _I2, _T, ain);
      else
        readin_v(_t2, _ic3, _I2, _T, ain);
      ker_trans_input_(*this, aout, (InputType *)ain, 0, 0, 0, -1);

      if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
        if (stream_in_) {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        } else {
          iter_each (_wA, A) {
          iter_each (_hA, A) {
            _mm<V>::store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                *((__m<V> *)&aout[_wA][_hA][0]));
          }}
        }
      } else {
        iter_each (_wA, A) {
        iter_each (_hA, A) {
#pragma omp simd
        iter_each (_V, V) {
          md6(atinput6, _wA, _hA, _ic3, _I2, _T, _V) = aout[_wA][_hA][_V];
        }}}
      }
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_input(
    TinputType *tinput, InputType *input)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input);
  else
    __trans_input_plain(tinput, input);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_inputa_blocked(
    TinputType *tinput, InputType *input, int _t2, int _wA, int Tz)
{
  // n, ic2, ih, iw, V => t2, wA | hA, ic3, I2, T, V
  MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2,
      this->ih, this->iw, V);
  MD5(TinputType, atinput, tinput, A, this->ic3, this->I2, Tz, V);

  alignas(64) TrOpType aout[A][A][V];

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    InputType *in = &md7(ainput, _n, 0, _ic3, _I2, _ih, _iw, 0);
    if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
        && _wA_end == A - 1) {
      ker_trans_inputa_(*this, aout, in, _wA, 0, A - 1, 0, A - 1);
    } else {
      ker_trans_inputa0_(
          *this, aout, in, _wA, _hA_start, _hA_end, _wA_start, _wA_end);
    }

    if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
      if (stream_in_) {
        iter_each (_hA, A) {
          _mm<V>::stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
        }
      } else {
        iter_each (_hA, A) {
          _mm<V>::store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
        }
      }
    } else {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        md5(atinput, _hA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
      }}
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_inputa_plain(
    TinputType * __restrict tinput, InputType * __restrict input, int _t2, int _wA, int Tz)
{
  // n, ic2, ih, iw, V => t2, wA | hA, ic3, I2, T, V
  MD5(TinputType, atinput, tinput, A, this->ic3, this->I2, Tz, V);

  alignas(64) TrOpType aout[A][A][V];
  alignas(64) InputType ain[A][A][V];
  SET_EPI32(this->ih * this->iw)

  auto readin_v = [&](int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2, V,
        this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    iter_each (__wA, A) {
    iter_each (__hA, A) {
      if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
          || __wA > _wA_end) {
#pragma omp simd
        iter_each (_V, V)
          ain[__hA][__wA][_V] = 0.0f;
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
          constexpr int scale = sizeof(InputType);
          __m<V> t = _mm<V>::i32gather_ps(vindex,
              &md7(ainput, _n, 0, _ic3, _I2, 0, _ih + __hA, _iw + __wA),
              scale);
          _mm<V>::store_ps(ain[__hA][__wA], t);
        } else {
#pragma omp simd
          iter_each (_V, V)
            ain[__hA][__wA][_V]
                = md7(ainput, _n, 0, _ic3, _I2, _V, _ih + __hA, _iw + __wA);
        }
      }
    }}
  };

  auto readin_r = [&](int _ic3, int _I2, int _T, InputType ain[A][A][V]) {
    MD4(InputType, ainput, input, this->n, this->ic, this->ih, this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

    assert(this->ic4 == 1);
    bool is_Ir = _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    if (is_Ir) {
      iter_each (__wA, A) {
      iter_each (__hA, A) {
        if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
            || __wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[__hA][__wA][_V] = 0.0f;
        } else {
#pragma omp simd
          iter_each (_V, this->Ir)
            ain[__hA][__wA][_V] = md4(ainput, _n,
                (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
        }
      }}
    } else {
      iter_each (__wA, A) {
      iter_each (__hA, A) {
        if (__hA < _hA_start || __hA > _hA_end || __wA < _wA_start
            || __wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V)
            ain[__hA][__wA][_V] = 0.0f;
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md4(ainput, _n, (_ic3 * this->I2 + _I2) * V, _ih + __hA, _iw + __wA),
                scale);
            _mm<V>::store_ps(ain[__hA][__wA], t);
          } else {
#pragma omp simd
            iter_each (_V, V)
              ain[__hA][__wA][_V] = md4(ainput, _n,
                  (_ic3 * this->I2 + _I2) * V + _V, _ih + __hA, _iw + __wA);
          }
        }
      }}
    }
  };

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_T, Tz) {
    if (this->Ir != V)
      readin_r(_ic3, _I2, _T, ain);
    else
      readin_v(_ic3, _I2, _T, ain);
    ker_trans_inputa_(*this, aout, (InputType *)ain, _wA, 0, A - 1, 0, -1);

    if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
      if (stream_in_) {
        iter_each (_hA, A)
          _mm<V>::stream_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
      } else {
        iter_each (_hA, A)
          _mm<V>::store_ps(&md5(atinput, _hA, _ic3, _I2, _T, 0),
              *((__m<V> *)&aout[_hA][_wA][0]));
      }
    } else {
      iter_each (_hA, A) {
#pragma omp simd
      iter_each (_V, V) {
        md5(atinput, _hA, _ic3, _I2, _T, _V) = aout[_hA][_wA][_V];
      }}
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_inputa(
    TinputType *tinput, InputType *input, int _t2, int _wA, int Tz)
{
  if(input_is_bfmt_ || input_as_bfmt_)
    __trans_inputa_blocked(tinput, input, _t2, _wA, Tz);
  else
    __trans_inputa_plain(tinput, input, _t2, _wA, Tz);
}

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD6(TinputType, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

  bool scramble = (this->T == this->Tr) || (this->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _hA = n % A;
      int _wA = n / A;
      iter_each(_oc3, this->oc3) {
        bool last_ic4 = _ic4 == this->ic4 - 1;
        int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
        iter_each(_ic3, ic3) {
          int attr = _ic3 == 0 && _ic4 == 0
              ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*(elx_conv_params_t *)this,
                   &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                   &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                   &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
                   nullptr, attr, 0, nullptr, nullptr, nullptr);
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 && this->ic4 == 1
                          ? set_attr(attr_, r_output_idx)
                          : attr_;
          ker_gemm_tail(*(elx_conv_params_t *)this,
                        &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                        &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                        &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                        nullptr, attr, 0, nullptr, nullptr, nullptr);
        }
      }
    }
  } else {
    iter_each(_wA, A) {
    iter_each(_hA, A) {
    iter_each(_oc3, this->oc3) {
      bool last_ic4 = _ic4 == this->ic4 - 1;
      int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
      iter_each(_ic3, ic3) {
        int attr =
            _ic3 == 0 && _ic4 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm(*(elx_conv_params_t *)this,
                 &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                 &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                 &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
                 nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
      if (last_ic4) {
        auto attr = this->ic3 == 1 && this->ic4 == 1
                        ? set_attr(attr_, r_output_idx)
                        : attr_;
        ker_gemm_tail(*(elx_conv_params_t *)this,
                      &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                      &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                      &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                      nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
    }}}
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_input_quantization(
    uint8_t *tinput_u8, TscaleType *tinput_qt_scale, TscaleType *tinput_qt_factor,
    TscaleType *tinput_max_abs, TinputType *tinput) {
  MD2(uint8_t, atinput2_u8, tinput_u8, this->t2, A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  MD2(TscaleType, atinput_qt_scale2, tinput_qt_scale, this->t2, A * A * this->ic3 * 2 * this->T);
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->ic3 * this->I2 * this->Vx * this->T * V);

  auto INT8GEMM_TIN_MIN_MAX_QTSCALE = INT8GEMM_TIN_MIN_MAX_QTSCALE_T;
#pragma omp for nowait collapse(4)
  iter_each (_t2, this->t2) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_ic3, this->ic3) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD7(TinputType, atinput7, &md2(atinput2, _t2, 0),
        A, A, this->ic3, this->I2, this->Vx, Tz, V);
    MD5(TscaleType, atinput_qt_scale, &md2(atinput_qt_scale2, _t2, 0),
        A, A, this->ic3, 2, Tz);
    MD7(uint8_t, atinput_u8, &md2(atinput2_u8, _t2, 0),
        A, A, this->ic3, this->I2, Tz, this->Vx, V);
    iter_each (_T, Tz) {
      __m<V> mmax, mmin;
      bool flush = true;
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
        __m<V> mcur = *(__m<V> *)&md7(atinput7, _wA, _hA, _ic3, _I2, _Vx, _T, 0);
        if (flush) {
          mmax = mcur;
          mmin = mcur;
          flush = false;
        } else {
          mmax = _mm<V>::max_ps(mcur, mmax);
          mmin = _mm<V>::min_ps(mcur, mmin);
        }
      }}

      TinputType max, min;
      if (I == ISA_SKX_AVX512 && std::is_same<TinputType, float>::value) {
        max = _mm<V>::reduce_max_ps(mmax);
        min = _mm<V>::reduce_min_ps(mmin);
      } else {
        TinputType *_mmax = (TinputType *)&mmax;
        TinputType *_mmin = (TinputType *)&mmin;
        max = _mmax[0];
        min = _mmin[0];
        for (int _V = 1; _V < V; _V++) {
          max = _mmax[_V] > max ? _mmax[_V] : max;
          min = _mmin[_V] < min ? _mmin[_V] : min;
        }
      }

      auto diff = max - min + 0.000001;
      md5(atinput_qt_scale, _wA, _hA, _ic3, 0, _T) =
          diff * (1.0f / INT8GEMM_TIN_MIN_MAX_QTSCALE);
      diff = INT8GEMM_TIN_MIN_MAX_QTSCALE / diff;
      md5(atinput_qt_scale, _wA, _hA, _ic3, 1, _T) = min;

      __m<V> mdifff32 = _mm<V>::set1_ps(diff);
      __m<V> mminf32 = _mm<V>::set1_ps(min);
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
        __m<V> f = *(__m<V> *)&md7(atinput7, _wA, _hA, _ic3, _I2, _Vx, _T, 0);
        __m<V> mmresf32 = (f - mminf32) * mdifff32;
        __i<V> mmresu32 = _mm<V>::cvt_roundps_epu32(mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mmresu32);
        _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), mmresu8);
      }}
    }
  }}}}
#pragma omp barrier
}

// tweights:      oc4 | oc3, ic3, A, A, O2, I2, V, V, Vx
// tinputs:        t2 | A, A, ic3, I2, T, V, Vx
// toutput:   t2, oc4 | A, A, oc3, O2, T, V
// weights_scale  oc4 | oc3, O2, V
// facotr:        oc4 | oc3, A, A, O2, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights, int _t2, int Tz,
    TscaleType *src_scale, TscaleType *weights_scale, TscaleType *weights_factor, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_i8_gemm0_ : ker_i8_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_i8_gemm0_tail_ : ker_i8_gemm_tail_;

  MD6(uint8_t, atinput, tinput, A, A, this->ic3, this->I2, Tz, V * this->Vx);
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(int8_t, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O2 * this->I2 * V * V * this->Vx);
  MD6(TscaleType, aweights_scale, weights_scale, this->oc3, this->ic3, A, A, this->O2, V);
  MD6(TscaleType, aweights_factor, weights_factor, this->oc3, this->ic3, A, A, this->O2, V);
  MD5(TscaleType, asrc_scale, src_scale, this->ic3,  A, A, 2, Tz);

  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_oc3, this->oc3) {
    bool last_ic4 = _ic4 == this->ic4 - 1;
    int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
    iter_each (_ic3, ic3) {
      auto attr = _ic3 == 0 && _ic4 == 0 ?
          set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      ker_gemm(*(elx_conv_params_t *)this,
          &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
          nullptr, attr,
          &md5(asrc_scale, _ic3, _wA, _hA, 0, 0),
          &md5(asrc_scale, _ic3, _wA, _hA, 1, 0),
          &md6(aweights_scale, _oc3, _ic3, _wA, _hA, 0, 0),
          &md6(aweights_factor, _oc3, _ic3, _wA, _hA, 0, 0));
    }
    if (last_ic4) {
      auto attr = this->ic3 == 1 && this->ic4 == 1 ?
          set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      ker_gemm_tail(*(elx_conv_params_t *)this,
          &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
          nullptr, attr,
          &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 0, 0),
          &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 1, 0),
          &md6(aweights_scale, _oc3, this->ic3 - 1, _wA, _hA, 0, 0),
          &md6(aweights_factor, _oc3, this->ic3 - 1, _wA, _hA, 0, 0));
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm_non_acc(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD6(TinputType, atinput, tinput, A, A, this->ic3, this->I2, Tz, V);
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O2 * this->I2 * V * V);

  bool scramble = (this->T == this->Tr) || (this->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _hA = n % A;
      int _wA = n / A;
      iter_each(_oc3, this->oc3) {
        bool last_ic4 = _ic4 == this->ic4 - 1;
        int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
        iter_each(_ic3, ic3) {
          int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm(*(elx_conv_params_t *)this,
                   &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                   &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                   &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
                   nullptr, attr, 0, nullptr, nullptr, nullptr);
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
          ker_gemm_tail(*(elx_conv_params_t *)this,
                        &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                        &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                        &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                        nullptr, attr, 0, nullptr, nullptr, nullptr);
        }
      }
    }
  } else {
    iter_each(_wA, A) {
    iter_each(_hA, A) {
    iter_each(_oc3, this->oc3) {
      bool last_ic4 = _ic4 == this->ic4 - 1;
      int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
      iter_each(_ic3, ic3) {
        int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm(*(elx_conv_params_t *)this,
                 &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                 &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
                 &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
                 nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
      if (last_ic4) {
        auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
        ker_gemm_tail(*(elx_conv_params_t *)this,
                      &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
                      &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
                      &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
                      nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
    }}}
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm_non_acc(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights, int _t2, int Tz,
    TscaleType *src_scale, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_i8_gemm0_ : ker_i8_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_i8_gemm0_tail_ : ker_i8_gemm_tail_;

  MD6(uint8_t, atinput, tinput, A, A, this->ic3, this->I2, Tz, V * this->Vx);
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD5(int8_t, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O2 * this->I2 * V * V * this->Vx);
  MD6(TscaleType, aweights_scale, weights_scale, this->oc3, this->ic3, A, A, this->O2, V);
  MD6(TscaleType, aweights_factor, weights_factor, this->oc3, this->ic3, A, A, this->O2, V);
  MD5(TscaleType, asrc_scale, src_scale, this->ic3, A, A, 2, Tz);

  bool scramble = (this->T == this->Tr) || (this->t2 >= 2 * mthr_);
  if (scramble) {
    int it_start = omp_get_thread_num();
    iter_each(i, A * A) {
      int n = (it_start + i) % (A * A);
      int _hA = n % A;
      int _wA = n / A;
      iter_each(_oc3, this->oc3) {
        bool last_ic4 = _ic4 == this->ic4 - 1;
        int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
        iter_each(_ic3, ic3) {
          auto attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
          attr = set_attr(attr, l_output_idx);
          attr = set_attr(attr, c_output_idx);
          ker_gemm(*(elx_conv_params_t *)this,
              &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
              &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
              &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
              nullptr, attr,
              &md5(asrc_scale, _ic3, _wA, _hA, 0, 0),
              &md5(asrc_scale, _ic3, _wA, _hA, 1, 0),
              &md6(aweights_scale, _oc3, _ic3, _wA, _hA, 0, 0),
              &md6(aweights_factor, _oc3, _ic3, _wA, _hA, 0, 0));
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
          attr = set_attr(attr, l_output_idx);
          attr = set_attr(attr, c_output_idx);
          ker_gemm_tail(*(elx_conv_params_t *)this,
              &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
              &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
              &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
              nullptr, attr,
              &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 0, 0),
              &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 1, 0),
              &md6(aweights_scale, _oc3, this->ic3 - 1, _wA, _hA, 0, 0),
              &md6(aweights_factor, _oc3, this->ic3 - 1, _wA, _hA, 0, 0));
        }
      }
    }
  } else {
    iter_each(_wA, A) {
    iter_each(_hA, A) {
    iter_each(_oc3, this->oc3) {
      bool last_ic4 = _ic4 == this->ic4 - 1;
      int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;
      iter_each(_ic3, ic3) {
        auto attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
        attr = set_attr(attr, l_output_idx);
        attr = set_attr(attr, c_output_idx);
        ker_gemm(*(elx_conv_params_t *)this,
            &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
            &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
            &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
            nullptr, attr,
            &md5(asrc_scale, _ic3, _wA, _hA, 0, 0),
            &md5(asrc_scale, _ic3, _wA, _hA, 1, 0),
            &md6(aweights_scale, _oc3, _ic3, _wA, _hA, 0, 0),
            &md6(aweights_factor, _oc3, _ic3, _wA, _hA, 0, 0));
      }
      if (last_ic4) {
        auto attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
        attr = set_attr(attr, l_output_idx);
        attr = set_attr(attr, c_output_idx);
        ker_gemm_tail(*(elx_conv_params_t *)this,
            &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
            &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
            &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
            nullptr, attr,
            &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 0, 0),
            &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 1, 0),
            &md6(aweights_scale, _oc3, this->ic3 - 1, _wA, _hA, 0, 0),
            &md6(aweights_factor, _oc3, this->ic3 - 1, _wA, _hA, 0, 0));
      }
    }}}
  }
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm(
    ToutputType * __restrict toutput, TinputType * __restrict tinput,
    TweightsType * __restrict tweights, int _ic4)
{
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD5(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

#pragma omp for nowait collapse(4)
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_oc3, this->oc3) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    auto ker_gemm_tail
        = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    bool last_ic4 = _ic4 == this->ic4 - 1;
    int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;

    iter_each (_ic3, ic3) {
      int attr = _ic3 == 0 && _ic4 == 0 ?
          set_attr(attr_, r_output_idx) : attr_;
      ker_gemm(*(elx_conv_params_t *)this,
          &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
          nullptr, attr, 0, nullptr, nullptr, nullptr);
    }
    if (last_ic4) {
      int attr = this->ic3 == 1 && this->ic4 == 1 ?
          set_attr(attr_, r_output_idx) : attr_;
      ker_gemm_tail(*(elx_conv_params_t *)this,
          &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
          nullptr, attr, 0, nullptr, nullptr, nullptr);
    }
  }}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm_non_acc(
    ToutputType * __restrict toutput, TinputType * __restrict tinput,
    TweightsType * __restrict tweights, int _ic4)
{
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * V);
  MD2(ToutputType, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD5(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A, this->O2 * this->I2 * V * V);

#pragma omp for nowait collapse(4)
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_oc3, this->oc3) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
    auto ker_gemm_tail
        = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;
    MD6(TinputType, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    bool last_ic4 = _ic4 == this->ic4 - 1;
    int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;

    iter_each (_ic3, ic3) {
      int attr = _ic3 == 0 ?
          set_attr(attr_, r_output_idx) : attr_;
      ker_gemm(*(elx_conv_params_t *)this,
          &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
          nullptr, attr, 0, nullptr, nullptr, nullptr);
    }
    if (last_ic4) {
      int attr = this->ic3 == 1 ?
          set_attr(attr_, r_output_idx) : attr_;
      ker_gemm_tail(*(elx_conv_params_t *)this,
          &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
          nullptr, attr, 0, nullptr, nullptr, nullptr);
    }
  }}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm_non_acc(
    ToutputType *toutput, uint8_t *tinput, int8_t *tweights,
    TscaleType *src_scale, TscaleType *src_factor, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ic4) {
  MD2(uint8_t, atinput2, tinput, this->t2, A * A * this->ic3 * this->I2 * this->T * V * this->Vx);
  MD2(ToutputType, atoutput2, toutput, this->t2, A * A * this->oc3 * this->O2 * this->T * V);
  MD5(int8_t, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O2 * this->I2 * V * V * this->Vx);
  MD6(TscaleType, aweights_scale, weights_scale,
      this->oc3, this->ic3, A, A, this->O2, V);
  MD6(TscaleType, aweights_factor, weights_factor,
      this->oc3, this->ic3, A, A, this->O2, V);
  MD2(TscaleType, asrc_scale2, src_scale, this->t2, A * A * this->ic3 * 2 * this->T);

#pragma omp for nowait collapse(4)
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_oc3, this->oc3) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    auto ker_gemm = (_t2 == this->t2 - 1) ? ker_i8_gemm0_ : ker_i8_gemm_;
    auto ker_gemm_tail
        = (_t2 == this->t2 - 1) ? ker_i8_gemm0_tail_ : ker_i8_gemm_tail_;
    MD6(uint8_t, atinput6, &md2(atinput2, _t2, 0), A, A, this->ic3, this->I2, Tz, V * this->Vx);
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    MD5(TscaleType, asrc_scale5, &md2(asrc_scale2, _t2, 0), A, A, this->ic3, 2, Tz);
    bool last_ic4 = _ic4 == this->ic4 - 1;
    int ic3 = last_ic4 ? this->ic3 - 1 : this->ic3;

    iter_each (_ic3, ic3) {
      int attr = _ic3 == 0 ?
          set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      ker_gemm(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
          nullptr, attr,
          &md5(asrc_scale5, _wA, _hA, _ic3, 0, 0),
          &md5(asrc_scale5, _wA, _hA, _ic3, 1, 0),
          &md6(aweights_scale, _oc3, _ic3, _wA, _hA, 0, 0),
          &md6(aweights_factor, _oc3, _ic3, _wA, _hA, 0, 0));
    }
    if (last_ic4) {
      int attr = this->ic3 == 1 ?
          set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      ker_gemm_tail(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput6, _wA, _hA, this->ic3 - 1, 0, 0, 0),
          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
          nullptr, attr,
          &md5(asrc_scale5, _wA, _hA, this->ic3 - 1, 0, 0),
          &md5(asrc_scale5, _wA, _hA, this->ic3 - 1, 1, 0),
          &md6(aweights_scale, _oc3, this->ic3 - 1, _wA, _hA, 0, 0),
          &md6(aweights_factor, _oc3, this->ic3 - 1, _wA, _hA, 0, 0));
    }
  }}}}
}

// tweights:    oc4, A | A, oc3, ic3, O2, I2, V, V
// tinputs:      t2, A | A, ic3, I2, T, V
// toutput: t2, oc4, A | A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemma(
    ToutputType * __restrict toutput, TinputType * __restrict tinput,
    TweightsType *tweights, int _t2, int Tz)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;
  auto ker_gemm_tail = (_t2 == this->t2 - 1) ? ker_gemm0_tail_ : ker_gemm_tail_;

  MD5(TinputType, atinput, tinput,  A, this->ic3, this->I2, Tz, V);
  MD5(ToutputType, atoutput, toutput, A, this->oc3, this->O2, Tz, V);
  MD4(TweightsType, atweights, tweights, A, this->oc3, this->ic3,
      this->O2 * this->I2 * V * V);

  iter_each (_hA, A) {
  iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3 - 1) {
      int attr = _ic3 == 0 ? set_attr(attr_, r_output_idx) : attr_;
      ker_gemm(*(elx_conv_params_t *)this,
          &md5(atoutput, _hA, _oc3, 0, 0, 0),
          &md5(atinput, _hA, _ic3, 0, 0, 0),
          &md4(atweights, _hA, _oc3, _ic3, 0),
          nullptr, attr, 0, nullptr, nullptr, nullptr);
    }
    int attr = this->ic3 == 1 ? set_attr(attr_, r_output_idx) : attr_;
    ker_gemm_tail(*(elx_conv_params_t *)this,
        &md5(atoutput, _hA, _oc3, 0, 0, 0),
        &md5(atinput, _hA, this->ic3 - 1, 0, 0, 0),
        &md4(atweights, _hA, _oc3, this->ic3 - 1, 0),
        nullptr, attr, 0, nullptr, nullptr, nullptr);
  }}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_plain(
    OutputType * __restrict output, ToutputType * __restrict toutput,
    BiasType * __restrict bias, TscaleType *shift, int _t2, int Tz, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  alignas(64) OutputType aout[A - K + 1][A - K + 1][V];
  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _oc3, int _O2, int _T,
                      OutputType aout[A - K + 1][A - K + 1][V]) {
    MD7(OutputType, aoutput, output, this->n, this->oc4, this->oc3, this->O2, V,
        this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr int scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _oc3, int _O2, int _T,
                        OutputType aout[A - K + 1][A - K + 1][V]) {
    MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);

    assert(this->oc4 == 1);
    int is_Or = _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr int scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  union {__m<V> vin; TrOpType ain[V];} In[A][A];
  using Array = TrOpType[A][A][V];

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
  iter_each (_T, Tz) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
      if (std::is_same<ToutputType, float>::value) {
        In[_wA][_hA].vin = _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
      } else {
        auto fp16v = _mm<V/2>::load_si256(
            (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
      }
    }}

    ker_trans_output_(*this, (OutputType *)aout, *(Array *)&In,
        (_ic4 == this->ic4 - 1 || _ic4 == -1)
        ? &md3(abias, _oc3, _O2, 0) : nullptr, nullptr, 0, -1);

    if (this->Or != V)
      writeout_r(_oc3, _O2, _T, aout);
    else
      writeout_v(_oc3, _O2, _T, aout);
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_blocked(
    OutputType *output, ToutputType *toutput, BiasType *bias, TscaleType *shift, int _t2, int Tz, int _ic4)
{
  auto ker_trans_output = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output_acc_ : ker_trans_output_;
  auto ker_trans_output_tail = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output0_acc_ : ker_trans_output0_;

  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD7(OutputType, aoutput, output, this->n, this->oc4, this->oc3, this->O2,
      this->oh, this->ow, V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);
  MD3(TscaleType, ashift, shift, this->oc3, this->O2, V);

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n_off = res.quot;
  auto _t_off = res.rem;

  union {__m<V> vin; TrOpType ain[V];} In[A][A];
  using Array = TrOpType[A][A][V];

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    output_tile_iter<A, K> t2spato_o(_n_off, _t_off, this->ht, this->wt,
        this->oh, this->ow);
  iter_each (_T, Tz) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
      if (std::is_same<ToutputType, float>::value) {
        In[_wA][_hA].vin = _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
      } else {
        auto fp16v = _mm<V/2>::load_si256(
            (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
      }
    }}

    auto _n = t2spato_o.n_;
    auto _oh = t2spato_o.t_;
    auto _ow = t2spato_o.l_;
    OutputType *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

    if (t2spato_o.is_border())
      ker_trans_output_tail(*this, out, *(Array *)&In,
          (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, &md3(ashift, _oc3, _O2, 0), t2spato_o.d_, t2spato_o.r_);
    else
      ker_trans_output(*this, out, *(Array *)&In,
          (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, &md3(ashift, _oc3, _O2, 0), A - K, A - K);

    ++ t2spato_o;
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_output(
    OutputType *output, ToutputType *toutput, BiasType *bias, TscaleType *shift, int _t2, int Tz, int _ic4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, shift, _t2, Tz, _ic4);
  else
    __trans_output_plain(output, toutput, bias, shift, _t2, Tz, _ic4);
}

// toutput:  mthr | hA/A, oc3, O2, T, V
// toutputa: t2, oc4 | oc3, O2, T, wA/A | hA/A-K+1, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_outputa_th(
    TrOpType *toutputa, void *toutput, int Tz)
{
  MD4(TrOpType, atoutput, toutput, A, this->oc3 * this->O2, Tz, V);
  MD4(TrOpType, atoutputa, toutputa, this->oc3 * this->O2, Tz, A, (A - K + 1) * V);

  iter_each (_oc, this->oc3 * this->O2) {
  iter_each (_T, Tz) {
    ker_trans_outputa_th_(*this, &md4(atoutputa, _oc, _T, 0, 0),
      &md4(atoutput, 0, _oc, _T, 0), Tz, stream_out_);
  }}
}

// output: n, oc2, h, w, V
// toutputa: t2, oc2, T, wA/A | hA/A-K+1, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_outputa_bh_blocked(
    OutputType *output, TrOpType *toutputa, BiasType *bias)
{
  MD5(OutputType, aoutput, output, this->n, this->oc2, this->oh, this->ow, V);
  MD2(BiasType, abias, bias, this->oc2, V);
  MD2(TrOpType, atoutputa2, toutputa, this->t2, A * (A - K + 1) * this->T * this->OC);

#pragma omp for nowait collapse(2)
  iter_each (_t2, this->t2) {
  iter_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(TrOpType, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz,
        A * (A - K + 1) * V);

    iter_each (_T, Tz) {
      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      OutputType *out = &md5(aoutput, _n, _oc2, _oh, _ow, 0);
      using Array1 = TrOpType[A][A - K + 1][V];
      Array1 *in = (Array1 *)&md3(atoutputa3, _oc2, _T, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_outputa0_bh_(
            *this, out, *in, &md2(abias, _oc2, 0), _hOA_end, _wOA_end);
      else
        ker_trans_outputa_bh_(
            *this, out, *in, &md2(abias, _oc2, 0), A - K, A - K);
    }
  }}
}

// output: n, OC, h, w
// toutputa: t2, oc2, T, wA/A | hA/A-K+1, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_outputa_bh_plain(
    OutputType * __restrict output, TrOpType * __restrict toutputa, BiasType *bias)
{
  MD2(BiasType, abias, bias, this->oc2, V);
  MD2(TrOpType, atoutputa2, toutputa, this->t2, A * (A - K + 1) * this->T * this->OC);

  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _t2, int _oc2, int _T,
                      OutputType aout[A - K + 1][A - K + 1][V]) {
    MD5(OutputType, aoutput, output, this->n, this->oc2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
          iter_each (_V, V)
            md5(aoutput, _n, _oc2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(
              &md5(aoutput, _n, _oc2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
            md5(aoutput, _n, _oc2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _t2, int _oc2, int _T,
                      OutputType aout[A - K + 1][A - K + 1][V]) {
    MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    assert(this->oc4 == 1);
    bool is_Or = _oc2 == this->oc2 - 1;
    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
          if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if (this->with_ip_sum && !output_as_bfmt_) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, _oc2 * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, _oc2 * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

#pragma omp for nowait collapse(2)
  iter_each (_t2, this->t2) {
  iter_each (_oc2, this->oc2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD3(TrOpType, atoutputa3, &md2(atoutputa2, _t2, 0), this->oc2, Tz,
        A * (A - K + 1) * V);
    alignas(64) OutputType aout[A - K + 1][A - K + 1][V];

    iter_each (_T, Tz) {
      using Array1 = TrOpType[A][A - K + 1][V];
      Array1 *in = (Array1 *)&md3(atoutputa3, _oc2, _T, 0);

      ker_trans_outputa_bh_(
          *this, (OutputType *)aout, *in, &md2(abias, _oc2, 0), 0, -1);

      if (this->Or != V)
        writeout_r(_t2, _oc2, _T, aout);
      else
        writeout_v(_t2, _oc2, _T, aout);
    }
  }}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_outputa_bh(
    OutputType *output, TrOpType *toutputa, BiasType *bias)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_outputa_bh_blocked(output, toutputa, bias);
  else
    __trans_outputa_bh_plain(output, toutputa, bias);

}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_blocked(
    OutputType *output, ToutputType *toutput, BiasType *bias, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD7(OutputType, aoutput, output, this->n, this->oc4, this->oc3, this->O2,
      this->oh, this->ow, V);
  MD2(ToutputType, atoutput2, toutput, this->t2, A * A * this->T * this->oc3 * this->O2 * V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  auto ker_trans_output = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output_acc_ : ker_trans_output_;
  auto ker_trans_output_tail = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output0_acc_ : ker_trans_output0_;

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(ToutputType, atoutput, &md2(atoutput2, _t2, 0), A, A, this->oc3, this->O2, Tz, V);
    union {__m<V> vin; TrOpType ain[V];} In[A][A];
    using Array = TrOpType[A][A][V];

    iter_each (_T, Tz) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        if (std::is_same<ToutputType, float>::value) {
          In[_wA][_hA].vin = _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        } else {
          auto fp16v = _mm<V/2>::load_si256(
              (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
          In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
        }
      }}

      int _n, _oh, _ow, _hOA_end, _wOA_end;
      t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
      OutputType *out = &md7(aoutput, _n, 0, _oc3, _O2, _oh, _ow, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_output_tail(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, nullptr, _hOA_end, _wOA_end);
      else
        ker_trans_output(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, nullptr, A - K, A - K);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_plain(
    OutputType * __restrict output, ToutputType * __restrict toutput, BiasType *bias, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);
  MD2(ToutputType, atoutput2, toutput, this->t2,
      A * A * this->T * this->oc3 * this->O2 * V);

  SET_EPI32(this->oh * this->ow)

  auto writeout_v = [&](int _t2, int _oc3, int _O2, int _T,
                      OutputType aout[A - K + 1][A - K + 1][V]) {
    MD7(OutputType, aoutput, output, this->n, this->oc4, this->oc3,
        this->O2, V, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
        } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
          __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
          constexpr auto scale = sizeof(OutputType);
          _mm<V>::i32scatter_ps(
              &md7(aoutput, _n, 0, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
              vindex, t, scale);
        } else {
#pragma omp simd
          iter_each (_V, V)
            md7(aoutput, _n, 0, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
        }
      }
    }
  };

  auto writeout_r = [&](int _t2, int _oc3, int _O2, int _T,
                        OutputType aout[A - K + 1][A - K + 1][V]) {
    MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);

    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);

    assert(this->oc4 == 1);
    bool is_Or = _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md4(aoutput, _n, (this->oc2 - 1) * V + _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md4(aoutput, _n, (_oc3 * this->O2 + _O2) * V + _V, _oh + _hA,
                  _ow + _wA) = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  // ICC-19 bug, build crash in case of t2 first
#pragma omp for nowait collapse(3)
  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD6(ToutputType, atoutput6, &md2(atoutput2, _t2, 0), A, A, this->oc3,
        this->O2, Tz, V);
    alignas(64) OutputType aout[A - K + 1][A - K + 1][V];
    union {__m<V> vin; TrOpType ain[V];} In[A][A];
    using Array = TrOpType[A][A][V];

    iter_each (_T, Tz) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        if (std::is_same<ToutputType, float>::value) {
          In[_wA][_hA].vin = _mm<V>::load_ps(&md6(atoutput6, _wA, _hA, _oc3, _O2, _T, 0));
        } else {
          auto fp16v = _mm<V/2>::load_si256(
              (__m256i *)&md6(atoutput6, _wA, _hA, _oc3, _O2, _T, 0));
          In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
        }
      }}

      ker_trans_output_(*this, (OutputType *)aout, *(Array *)&In,
          (_ic4 == -1 || _ic4 == this->ic4 - 1)
          ? &md3(abias, _oc3, _O2, 0) : nullptr, nullptr, 0, -1);

      if (this->Or != V)
        writeout_r(_t2, _oc3, _O2, _T, aout);
      else
        writeout_v(_t2, _oc3, _O2, _T, aout);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_output(
    OutputType *output, ToutputType *toutput, BiasType *bias, int _ic4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _ic4);
  else
    __trans_output_plain(output, toutput, bias, _ic4);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::prepare_tweights(
    WeightsType * __restrict weights) {
  trans_weights(tweights_, weights, this->oc4);
  MD5(ToutputType, atweights_ci, tweights_ci_,
      this->oc4, this->oc3, this->O1, this->O, V);

  // confident interval
  if (weights_is_bfmt_ || weights_as_bfmt_) {
    MD12(ToutputType, atweights, tweights_, this->oc4, this->ic4, this->oc3, this->ic3,
        A, A, this->O1, this->I2, this->Vx, V, this->O, V);
    __m<V> mmblk
        = _mm<V>::set1_ps(this->ic4 * this->ic3 * A * A * this->I2 * this->Vx * V);

#pragma omp for nowait collapse(4) schedule(static)
    iter_each (_oc4, this->oc4) {
    iter_each (_oc3, this->oc3) {
    iter_each (_O1, this->O1) {
    iter_each (_O, this->O) {
      __m<V> mmsum = _mm<V>::set1_ps(0.0);
      iter_each (_ic4, this->ic4) {
      iter_each (_ic3, this->ic3) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_I2, this->I2) {
      iter_each (_iVx, this->Vx) {
      iter_each (_iV, V) {
        mmsum = _mm<V>::add_ps(
            *(__m<V> *)&md12(atweights, _oc4, _ic4, _oc3, _ic3,
                             _wA, _hA, _O1, _I2, _iVx, _iV, _O, 0), mmsum);
      }}}}}}}
      // avarage
      __m<V> mmavg = _mm<V>::div_ps(mmsum, mmblk);
      // standard deviation
      __m<V> mmsd = _mm<V>::set1_ps(0.0);
      iter_each (_ic4, this->ic4) {
      iter_each (_ic3, this->ic3) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_I2, this->I2) {
      iter_each (_iVx, this->Vx) {
      iter_each (_iV, V) {
        __m<V> mmdiff;
        mmdiff = _mm<V>::sub_ps(
            *(__m<V> *)&md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
                             _O1, _I2, _iVx, _iV, _O, 0), mmavg);
        mmsd = _mm<V>::add_ps(_mm<V>::mul_ps(mmdiff, mmdiff), mmsd);
      }}}}}}}
      mmsd = _mm<V>::div_ps(mmsd, mmblk);
      mmsd = _mm<V>::sqrt_ps(mmsd);

      // upper
      // 90% => 1.645
      // 95% => 1.960
      // 99% => 2.576
      __m<V> ci_coef = _mm<V>::set1_ps(2.576);
      __m<V> mmupper = _mm<V>::add_ps(mmavg, _mm<V>::mul_ps(mmsd, ci_coef));
      _mm<V>::store_ps(&md5(atweights_ci, _oc4, _oc3, _O1, _O, 0), mmupper);
    }}}}
  } else {
    // TODO:
  }

#ifdef DEBUG
  printf("Confident Interval upper +++++\n");
  iter_each (_oc4, this->oc4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_O1, this->O1) {
  iter_each (_O, this->O) {
  iter_each (_oV, V) {
    printf("upper %f\n", md5(atweights_ci, _oc4, _oc3, _O1, _O, _oV));
  }}}}}
  printf("Confident Interval upper -----\n\n");
#endif

  this->tweights_preprocessed_ = true;
  return;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::preprocess(WeightsType * __restrict weights) {
  if (this->execution_mode  == 0xa161)
    prepare_tweights(weights);
}

} // namespace euler
