#include <string.h>
#include <float.h>
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

const float INT8GEMM_TWT_QTSCALE = 127.0;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE = 255.0;

Template_elx_conv_wino_t Instance_elx_conv_wino_t::elx_conv_wino_t(
    eld_conv_t<UserTypes> &dc)
    : elx_conv_t<UserTypes>(dc)
{
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
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t tinput_u8_size = 0, tinput_qt_scale_size = 0,
      tinput_qt_factor_size = 0, tinput_max_abs_size = 0, tweights_s8_size = 0,
      tweights_qt_scale_size = 0, tweights_qt_factor_size = 0, tweights_ci_size = 0;

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

  prepare_wino_tinput_quant_cali();

  input_is_bfmt_ = this->input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = this->weights_fmt == OIhw16i16o;
  output_is_bfmt_ = this->output_fmt == nChw16c;
  input_as_bfmt_ = this->input_fmt == nchw && this->input_as_blocked;
  weights_as_bfmt_ = this->input_fmt == oihw && this->weights_as_blocked;
  output_as_bfmt_ = this->output_fmt == nchw && this->output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (this->Or != V && this->output_fmt == nhwc) {
    el_error("Unimplemented: nhwc output with Or");
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
    if (!this->online_sampling_hp)
      tinput_size = this->IC * A * A * this->T * mthr_ * sizeof(TinputType);
    else
      tinput_size = A * A * this->I2 * this->Vx * V * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    tinput_u8_size = A * A * this->IC * mthr_ * this->T * sizeof(uint8_t);
    tinput_qt_scale_size = mthr_ * 2 * this->ic3 * this->T * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_qt_scale_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType);
    tweights_qt_factor_size = this->ic4 * this->ic3 * this->OC * A * A * sizeof(TscaleType); // * this->ic4
    tweights_ci_size = this->OC * sizeof(TscaleType);
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

  workspace_ = nullptr, scratch_ = nullptr;
  size_t workspace_size = tweights_size_ + tweights_s8_size_
      + tweights_qt_scale_size_ + tweights_qt_factor_size_ + tweights_ci_size_;
  size_t scratch_size = tinput_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_ + tinput_u8_size_
      + tinput_qt_scale_size_ + tinput_qt_factor_size_ + tinput_max_abs_size_;

  if (this->wino_tinput_qt_cali) {
    workspace_size += tinput_qt_scale_size_;
    scratch_size -= tinput_qt_scale_size_;
  }

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
    if (this->wino_tinput_qt_cali) {
      tinput_qt_scale_ = (TscaleType *)((char *)tweights_ci_ + tweights_ci_size_);
      tweights_s8_ = (int8_t *)((char *)tinput_qt_scale_ + tinput_qt_scale_size_);
    } else {
      tweights_s8_ = (int8_t *)((char *)tweights_ci_ + tweights_ci_size_);
    }
  } else {
    tweights_ = (TweightsType *)galloc::get();
    tinput_ = (TinputType *)((char *)tweights_ + tweights_size_);
  }
  toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
  binput_ = (InputType *)((char *)toutput_ + toutput_size_);
  bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
  boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
  tinput_qt_factor_ = (TscaleType *)((char *)boutput_ + boutput_size_);
  tinput_max_abs_ = (TscaleType *)((char *)tinput_qt_factor_ + tinput_qt_factor_size_);
  if (this->wino_tinput_qt_cali) {
    tinput_u8_ = (uint8_t *)((char *)tinput_max_abs_ + tinput_max_abs_size_);
  } else {
    tinput_qt_scale_ = (TscaleType *)((char *)tinput_max_abs_ + tinput_max_abs_size_);
    tinput_u8_ = (uint8_t *)((char *)tinput_qt_scale_ + tinput_qt_scale_size_);
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::prepare_wino_tinput_quant_cali()
{
  if (this->wino_tinput_qt_S != EL_NO_CALI &&
      this->wino_tinput_qt_z != EL_NO_CALI) {
    this->wino_tinput_qt_cali = true;
    printf("wino_tinput_qt_S %f wino_tinput_qt_z %d\n",
        this->wino_tinput_qt_S, (int)this->wino_tinput_qt_z);
  } else {
    this->wino_tinput_qt_cali = false;
  }
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

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_weights_post(
    TweightsType *tweights, TrOpType at[A][A][V][V], const int _oc4,
    const int _ic4, const int _oc3, const int _ic3, const int _O1,
    const int _I2, const int _O) {
  MD9(TweightsType, atweights, tweights, this->oc3, this->ic3, A, A,
      this->O1, this->I2, V, this->O, V);
  if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
      && std::is_same<TweightsType, float>::value) {
    if (stream_wei_) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
        _mm<V>::stream_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
            _O1, _I2, _iV, _O, 0), *((__m512 *)&at[_wA][_hA][_iV][0]));
      }}}
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
        _mm512_store_ps(&md9(atweights, _oc3, _ic3, _wA, _hA,
            _O1, _I2, _iV, _O, 0), *((__m512 *)&at[_wA][_hA][_iV][0]));
      }}}
    }
  } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
     && std::is_same<TweightsType, float16>::value) {
    if (stream_wei_) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&at[_wA][_hA][_iV][0],
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::stream_si256((__m256i *)&md9(atweights, _oc3,
                             _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);

      }}}
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)&at[_wA][_hA][_iV][0],
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256((__m256i *)&md9(atweights, _oc3,
                             _ic3, _wA, _hA, _O1, _I2, _iV, _O, 0), fp16v);
      }}}
    }
  } else {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
    iter_each (_iV, V) {
#pragma omp simd
    iter_each (_oV, V) {
      md9(atweights, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV)
          = at[_wA][_hA][_iV][_oV];
    }}}}
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_oihw(
    TweightsType * __restrict tweights, WeightsType * __restrict weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD11(WeightsType, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V,
      this->ic4, this->ic3, this->I2, V, K, K);
  MD3(TweightsType, atweights, tweights, this->oc4, this->ic4,
      this->oc3 * this->ic3 * A * A * this->O2 * this->I2 * V * V);

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
    __trans_weights_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                         _ic3, _O1, _I2, _O);
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_blocked(
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  // oc2, ic2, hK, wK, V, V => oc4, ic4, oc3, ic3, wA, hA, O2, I2, V, V
  MD8(WeightsType, aweights, weights, oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2 * this->Vx, K * K * V * V);
  MD3(TweightsType, atweights, tweights, this->oc4, this->ic4,
      this->oc3 * this->ic3 * A * A * this->O2 * this->I2 * V * V);

#pragma omp for nowait collapse(7) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2 * this->Vx) {
  iter_each (_O, this->O) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md8(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);
    __trans_weights_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                         _ic3, _O1, _I2, _O);
  }}}}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weights_hwio(
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _oc4, int _oc3, int _O1, int _O,
                    int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, this->ic, this->oc);
    int iV = is_Ir ? this->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), this->ic4,
          this->ic3, this->I2, V, this->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _ic4, _ic3, _I2, _iV, 0),
          this->oc4, this->oc3, this->O1, this->O, V);
      if (is_Or) {
        iter_each (_oV, this->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _oc4, _oc3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
        }
      }
    }}}
  };

  MD3(TweightsType, atweights, tweights, this->oc4, this->ic4,
      this->oc3 * this->ic3 * A * A * this->O2 * this->I2 * V * V);
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

    readin(ain, weights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    ker_trans_weights_(aout, ain);
    __trans_weights_post(&md3(atweights, _oc4, _ic4, 0), aout, _oc4, _ic4, _oc3,
                         _ic3, _O1, _I2, _O);
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

  // trans-weights
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
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
        _mm512_store_ps(&md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA,
            _O1, _I2, _iVx, _iV, _O, 0), *((__m512 *)&aout[_wA][_hA][_iV][0]));
      }}}
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
      iter_each (_iV, V) {
#pragma omp simd
        iter_each (_oV, V) {
          md12(atweights, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iVx, _iV,
              _O, _oV) = aout[_wA][_hA][_iV][_oV];
        }
      }}}
    }
   }}}}}}}}
#pragma omp barrier

  // abs-max
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
    }}}
    _mm512_store_ps(&md9(atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0), mmax_cur);
  }}}}}}}}
#pragma omp barrier

  // I2 Vx V => I2 V Vx
  MD12(TweightsType, _atweights, tweights, oc4, this->ic4, this->oc3, this->ic3,
      A, A, this->O1, this->I2, V, this->Vx, this->O, V);
  // quantization
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
    // multi scal
    t0 = _mm<V>::mul_ps(*(__m<V> *)&md12(_atweights, _oc4, _ic4, _oc3, _ic3,
                            _wA, _hA, _O1, _I2, _iV, _iVx, _O, 0), mmscale);
    t0 = _mm<V>::div_ps(t0,
        *(__m<V> *)&md9(
            atweights_qt_scale, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, 0));
    // rounding
    t0 = _mm<V>::roundscale_ps(
        t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

  // weights-acc
#pragma omp for nowait collapse(9) schedule(static)
  iter_each (_oc4, oc4) {
  iter_each (_ic4, this->ic4) {
  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_O1, this->O1) {
  iter_each (_O, this->O) {
  iter_each (_oV, V) {
    int acc = 0;
    iter_each (_I2, this->I2) {
    iter_each (_iV, V) {
    iter_each (_iVx, this->Vx) {
      acc += md12(atweights_s8, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _I2, _iV, _O, _oV, _iVx);
    }}}
    md9(atweights_qt_factor, _oc4, _ic4, _oc3, _ic3, _wA, _hA, _O1, _O, _oV) = acc;
  }}}}}}}}}

  // weights-scale
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
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weights(
    TweightsType *tweights, WeightsType *weights, int oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weights_blocked(tweights, weights, oc4);
  else if (this->weights_fmt == hwio)
    __trans_weights_hwio(tweights, weights, oc4);
  else
    __trans_weights_oihw(tweights, weights, oc4);
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
void Instance_elx_conv_wino_t::__trans_weightsf_oihw(
    TweightsType * __restrict tweights, WeightsType * __restrict weights, int _ic4, int _oc4)
{
  MD11(WeightsType, aweights_v, weights, oc4, this->oc3, this->O1, this->O, V,
      this->ic4, this->ic3, this->I2, V, K, K);

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
    __trans_weights_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsf_hwio(
    TweightsType * __restrict tweights, WeightsType * __restrict weights, int _ic4, int _oc4)
{
  auto readin = [&](WeightsType ain[K][K][V][V], WeightsType *wei,
                    int _oc4, int _oc3, int _O1, int _O,
                    int _ic4, int _ic3, int _I2, bool is_Ir, bool is_Or) {
    MD4(WeightsType, aweights0, wei, K, K, this->ic, this->oc);
    int iV = is_Ir ? this->Ir : V;

    iter_each (_hK, K) {
    iter_each (_wK, K) {
    iter_each (_iV, iV) {
      MD5(WeightsType, aweights1, &md4(aweights0, _hK, _wK, 0, 0), this->ic4,
          this->ic3, this->I2, V, this->oc);
      MD5(WeightsType, aweights2, &md5(aweights1, _ic4, _ic3, _I2, _iV, 0),
          this->oc4, this->oc3, this->O1, this->O, V);
      if (is_Or) {
        iter_each (_oV, this->Or)
          ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
      } else {
        if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
          auto t = *(__m<V>*)&md5(aweights2, _oc4, _oc3, _O1, _O, 0);
          _mm<V>::store_ps(ain[_hK][_wK][_iV], t);
        } else {
#pragma omp simd
          iter_each (_oV, V)
            ain[_hK][_wK][_iV][_oV] = md5(aweights2, _oc4, _oc3, _O1, _O, _oV);
        }
      }
    }}}
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

    readin(ain, weights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, is_Ir, is_Or);
    ker_trans_weights_(aout, ain);
    __trans_weights_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_weightsf_blocked(
    TweightsType *tweights, WeightsType *weights, int _ic4, int _oc4)
{
  MD11(WeightsType, aweights, weights, oc4, this->oc3, this->O1, this->O,
      this->ic4, this->ic3, this->I2, K, K, V, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
  iter_each (_O1, this->O1) {
  iter_each (_I2, this->I2) {
  iter_each (_O, this->O) {
    alignas(64) TrOpType aout[A][A][V][V];
    WeightsType *in = &md11(aweights, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, 0, 0, 0, 0);
    using Array = WeightsType[K][K][V][V];
    ker_trans_weights_(aout, *(Array *)in);
    __trans_weights_post(tweights, aout, _oc4, _ic4, _oc3, _ic3, _O1, _I2, _O);
  }}}}};
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_weightsf(
    TweightsType *tweights, WeightsType *weights, int _ic4, int _oc4)
{
  if (weights_is_bfmt_ || weights_as_bfmt_)
    __trans_weightsf_blocked(tweights, weights, _ic4, _oc4);
  else if (this->weights_fmt == hwio)
    __trans_weightsf_hwio(tweights, weights, _ic4, _oc4);
  else
    __trans_weightsf_oihw(tweights, weights, _ic4, _oc4);
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_input_nchw(
    TinputType *__restrict tinput, InputType *__restrict input, int Tz, int _t2,
    int _ic4) {
  // n, IC, ih, iw => t2 | wA, hA, ic3, I2, T, V
  alignas(64) TrOpType aout[A][A][V];
  alignas(64) InputType ain[A][A][V];
  SET_EPI32(this->ih * this->iw);

  auto readin = [&](InputType ain[A][A][V], int _ic3, int _I2, int _T, bool is_Ir) {
    MD2(InputType, ainput0, input, this->n, this->ic * this->ih * this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD6(InputType, ainput1, &md2(ainput0, _n, 0), this->ic4, this->ic3,
        this->I2, V, this->ih, this->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0.0f;
        } else {
          iter_each(_V, this->Ir) {
            ain[_hA][_wA][_V] =
                md6(ainput1, _ic4, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
          }
        }
      }}
    } else {
      iter_each (_hA, A) {
      iter_each (_wA, A) {
        if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start
            || _wA > _wA_end) {
#pragma omp simd
          iter_each (_V, V) ain[_hA][_wA][_V] = 0.0f;
        } else {
          if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
            constexpr int scale = sizeof(InputType);
            __m<V> t = _mm<V>::i32gather_ps(vindex,
                &md6(ainput1, _ic4, _ic3, _I2, 0, _ih + _hA, _iw + _wA),
                scale);
            _mm<V>::store_ps(ain[_hA][_wA], t);
          } else {
#pragma omp simd
            iter_each (_V, V)
              ain[_hA][_wA][_V] = md6(ainput1, _ic4, _ic3, _I2, _V,
                                      _ih + _hA, _iw + _wA);
          }
        }
      }}
    }
  };

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1 &&
                 _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    iter_each (_T, Tz) {
      readin(ain, _ic3, _I2, _T, is_Ir);
      ker_trans_input_(*this, aout, (InputType *)ain, 0, 0, 0, -1);
      __trans_input_post(tinput, aout, Tz, _ic3, _I2, _T);
    }
  }}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_blocked(
    TinputType * __restrict tinput, InputType * __restrict input, int Tz, int _t2, int _ic4)
{
  // n, ic2, ih, iw, V => t2 | wA, hA, ic3, I2, T, V
  MD7(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2, this->ih, this->iw, V);
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

    InputType *in = &md7(ainput, t2spati_o.n_, _ic4, _ic3, _I2, _ih, _iw, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*this, aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);
    __trans_input_post(tinput, aout, Tz, _ic3, _I2, _T);

    ++ t2spati_o;
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_nhwc(
    TinputType * __restrict tinput, InputType * __restrict input, int Tz, int _t2, int _ic4)
{
  // n, ih, iw, ic2, V => t2 | wA, hA, ic3, I2, T, V
  MD4(InputType, ainput0, input, this->n, this->ih, this->iw, this->ic);
  alignas(64) TrOpType aout[A][A][V];

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt,
      this->ih, this->iw, this->tp, this->lp);
  iter_each (_T, Tz) {
    auto _n = t2spati_o.n_;
    auto _ih = t2spati_o.anchor_t_;
    auto _iw = t2spati_o.anchor_l_;

    MD4(InputType, ainput1, &md4(ainput0, _n, _ih, _iw, 0), this->ic4, this->ic3,
        this->I2, V);
    InputType *in = &md4(ainput1, _ic4, _ic3, _I2, 0);
    if (!t2spati_o.is_border())
      ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
    else
      ker_trans_input0_(*this, aout, in,
          t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);
    __trans_input_post(tinput, aout, Tz, _ic3, _I2, _T);

    ++ t2spati_o;
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
  MD7(uint8_t, atinput_u8, tinput_u8, A, A, this->ic3, this->I2, Tz, this->Vx, V);
  MD5(TscaleType, atinput_qt_scale, tinput_qt_scale, this->ic3, A, A, 2, Tz);

  auto res = std::div(_t2 * this->T, this->nt);
  auto _n = res.quot;
  auto _t_off = res.rem;

  if (this->wino_tinput_qt_cali) {
    __m<V> mrepS = _mm<V>::set1_ps(1 / this->wino_tinput_qt_S);
    __m<V> mz = _mm<V>::set1_ps(this->wino_tinput_qt_z);
    alignas(64) TrOpType aout[A][A][V];

    iter_each(_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_Vx, this->Vx) {
      input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt, this->ih,
                                      this->iw, this->tp, this->lp);
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

        iter_each (_wA, A) {
        iter_each (_hA, A) {
          // Min-Max quantization
          __m<V> a = *(__m<V> *)&aout[_wA][_hA][0];
          __m<V> mmresf32 = a * mrepS + mz;
          // convert to uint8
          __i<V> mmresu32 = _mm<V>::cvt_roundps_epu32(
              mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mmresu32);
          // store
          _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), mmresu8);

          md5(atinput_qt_scale, _ic3, _wA, _hA, 0, _T) = this->wino_tinput_qt_S;
          md5(atinput_qt_scale, _ic3, _wA, _hA, 1, _T) = this->wino_tinput_qt_z;
        }}
      }
    }}}
    return;
  }

  if (!this->online_sampling_hp) {
    MD7(TinputType, atinput, tinput, this->ic3, this->I2, this->Vx, Tz, A, A, V);
    auto mmin = _mm<V>::set1_ps(FLT_MAX);
    auto mmax = _mm<V>::set1_ps(-FLT_MAX);

    iter_each(_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_Vx, this->Vx) {
      input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt, this->ih,
                                      this->iw, this->tp, this->lp);
      iter_each (_T, Tz) {
        auto _ih = t2spati_o.anchor_t_;
        auto _iw = t2spati_o.anchor_l_;

        MD3(TinputType, aout, &md7(atinput, _ic3, _I2, _Vx, _T, 0, 0, 0), A, A, V);
        using Array = TrOpType[A][A][V];
        InputType *in = &md8(ainput, t2spati_o.n_, 0, _ic3, _I2, _Vx, _ih, _iw, 0);
        if (!t2spati_o.is_border())
          ker_trans_input_(*this, *(Array *)&md3(aout, 0, 0, 0), in, 0, A - 1, 0, A - 1);
        else
          ker_trans_input0_(*this, *(Array *)&md3(aout, 0, 0, 0), in,
              t2spati_o.t_, t2spati_o.d_, t2spati_o.l_, t2spati_o.r_);

        ++ t2spati_o;

        iter_each (_wA, A) {
        iter_each (_hA, A) {
          mmin = _mm<V>::min_ps(mmin, *(__m<V> *)&md3(aout, _wA, _hA, 0));
          mmax = _mm<V>::max_ps(mmax, *(__m<V> *)&md3(aout, _wA, _hA, 0));
        }}
      }
    }}}

    TinputType min = _mm<V>::reduce_min_ps(mmin);
    TinputType max = _mm<V>::reduce_max_ps(mmax);

    TinputType delta = max - min + 0.000001;
    TinputType S = delta / INT8GEMM_TIN_MIN_MAX_QTSCALE;
    TinputType repS = INT8GEMM_TIN_MIN_MAX_QTSCALE / delta;
    TinputType z = std::ceil(-min * repS);

    iter_each(_T, Tz) {
      md5(atinput_qt_scale, 0, 0, 0, 0, _T) = S;
      md5(atinput_qt_scale, 0, 0, 0, 1, _T) = z;
    }

    iter_each (_ic3, this->ic3) {
    iter_each (_I2, this->I2) {
    iter_each (_Vx, this->Vx) {
    iter_each (_T, Tz) {
    iter_each (_wA, A) {
    iter_each (_hA, A) {
      // Min-Max quantization
      __m<V> mrepS = _mm<V>::set1_ps(repS);
      __m<V> mz = _mm<V>::set1_ps(z);
      __m<V> a = *(__m<V> *)&md7(atinput, _ic3, _I2, _Vx, _T, _wA, _hA, 0);
      __m<V> mmresf32 = a * mrepS + mz;
      // convert to uint8
      __i<V> mmresu32 = _mm<V>::cvt_roundps_epu32(mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
      __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mmresu32);
      // store
      _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), mmresu8);
    }}}}}}
    return;
  }

  MD5(TinputType, atinput, tinput, this->I2, this->Vx, A, A, V);
  iter_each(_ic3, this->ic3) {
    input_tile_iter<A, K> t2spati_o(_n, _t_off, this->ht, this->wt, this->ih,
                                    this->iw, this->tp, this->lp);
    iter_each (_T, Tz) {
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
        float delta = mmax[_wA][_hA][0] - mmin[_wA][_hA][0] + 0.000001;
        float S = delta / INT8GEMM_TIN_MIN_MAX_QTSCALE;
        float repS = INT8GEMM_TIN_MIN_MAX_QTSCALE / delta;
        float z = std::ceil(- mmin[_wA][_hA][0] * repS);
        mmax[_wA][_hA][0] = repS;
        mmin[_wA][_hA][0] = z;

        md5(atinput_qt_scale, _ic3, _wA, _hA, 0, _T) = S;
        md5(atinput_qt_scale, _ic3, _wA, _hA, 1, _T) = z;
      }}

      // quantization
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        // Min-Max quantization
        __m<V> mrepS = _mm<V>::set1_ps(mmax[_wA][_hA][0]);
        __m<V> mz = _mm<V>::set1_ps(mmin[_wA][_hA][0]);
        __m<V> f = *(__m<V> *)&md5(atinput, _I2, _Vx, _wA, _hA, 0);
        __m<V> mmresf32 = f * mrepS + mz;
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

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::trans_input(
    TinputType *__restrict tinput, InputType *__restrict input, int Tz, int _t2,
    int _ic4) {
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input, Tz, _t2, _ic4);
  else if (this->input_fmt == nhwc)
    __trans_input_nhwc(tinput, input, Tz, _t2, _ic4);
  else
    __trans_input_nchw(tinput, input, Tz, _t2, _ic4);
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

Template_elx_conv_wino_t void
Instance_elx_conv_wino_t::__trans_input_post(TinputType *__restrict tinput,
  TrOpType at[A][A][V], const int Tz, const int _ic3, const int _I2, const int _T) {
  MD6(TinputType, atinput6, tinput, A, A, this->ic3, this->I2 * this->Vx, Tz, V);

  if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
      && std::is_same<TinputType, float>::value) {
    if (stream_in_) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        _mm<V>::stream_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                       *((__m<V> *)&at[_wA][_hA][0]));
      }}
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        _mm<V>::store_ps(&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0),
                      *((__m<V> *)&at[_wA][_hA][0]));
      }}
    }
  } else if (I == ISA_SKX_AVX512 && std::is_same<TrOpType, float>::value
     && std::is_same<TinputType, float16>::value) {
    if (stream_in_) {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&at[_wA][_hA][0]),
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::stream_si256(
            (__m256i *)&md6(atinput6, _wA, _hA, _ic3, _I2, _T, 0), fp16v);
      }}
    } else {
      iter_each (_wA, A) {
      iter_each (_hA, A) {
        auto fp16v = _mm<V>::cvtps_ph(*((__m<V> *)&at[_wA][_hA][0]),
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
      md6(atinput6, _wA, _hA, _ic3, _I2, _T, _V) = at[_wA][_hA][_V];
    }}}
  }
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_input_blocked(
    TinputType *__restrict tinput, InputType *__restrict input, int _ic4) {
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD2(TinputType, atinput, tinput, this->t2,
      A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  MD7(InputType, ainput, input, this->n, this->ic4, this->ic3,
      this->I2 * this->Vx, this->ih, this->iw, V);

  // ICC-19 bug, build crash in case of t2 first
#pragma omp for nowait collapse(3)
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2 * this->Vx) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    alignas(64) TrOpType aout[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      InputType *in = &md7(ainput, _n, _ic4, _ic3, _I2, _ih, _iw, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
      __trans_input_post(&md2(atinput, _t2, 0), aout, Tz, _ic3, _I2, _T);
    }
  }}}
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_input_nhwc(
    TinputType *__restrict tinput, InputType *__restrict input, int _ic4) {
  // n, ih, iw, ic => t2, wA, hA, ic3, I2, T, V
  MD2(TinputType, atinput, tinput, this->t2,
      A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  MD4(InputType, ainput0, input, this->n, this->ih, this->iw, this->ic);

  // ICC-19 bug, build crash in case of t2 first
#pragma omp for nowait collapse(3)
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2 * this->Vx) {
  iter_each (_t2, this->t2) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    alignas(64) TrOpType aout[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
      MD4(InputType, ainput1, &md4(ainput0, _n, _ih, _iw, 0), this->ic4,
          this->ic3, this->I2, V);
      InputType *in = &md4(ainput1, _ic4, _ic3, _I2, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*this, aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *this, aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
      __trans_input_post(&md2(atinput, _t2, 0), aout, Tz, _ic3, _I2, _T);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_input_nchw(
    TinputType * __restrict tinput, InputType * __restrict input, int _ic4)
{
  // n, ic2, ih, iw, V => t2, wA, hA, ic3, I2, T, V
  MD2(TinputType, atinput, tinput, this->t2, A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  SET_EPI32(this->ih * this->iw)

  auto readin = [&](InputType ain[A][A][V], int _t2, int _ic3, int _I2, int _T,
                    bool is_Ir) {
    MD2(InputType, ainput0, input, this->n, this->ic * this->ih * this->iw);
    int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
    t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);
    MD6(InputType, ainput1, &md2(ainput0, _n, 0), this->ic4, this->ic3,
        this->I2, V, this->ih, this->iw);

    if (is_Ir) {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
#pragma omp simd
            iter_each(_V, this->Ir) ain[_hA][_wA][_V] =
              md6(ainput1, _ic4, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
          }
        }
      }
    } else {
      iter_each (_hA, A) {
        iter_each(_wA, A) {
          if (_hA < _hA_start || _hA > _hA_end || _wA < _wA_start ||
              _wA > _wA_end) {
#pragma omp simd
            iter_each(_V, V) ain[_hA][_wA][_V] = 0.0f;
          } else {
            if (I == ISA_SKX_AVX512 && std::is_same<InputType, float>::value) {
              constexpr int scale = sizeof(InputType);
              __m<V> t = _mm<V>::i32gather_ps(vindex,
                  &md6(ainput1, _ic4, _ic3, _I2, 0, _ih + _hA, _iw + _wA), scale);
              _mm<V>::store_ps(ain[_hA][_wA], t);
            } else {
#pragma omp simd
              iter_each(_V, V) ain[_hA][_wA][_V] =
                  md6(ainput1, _ic4, _ic3, _I2, _V, _ih + _hA, _iw + _wA);
            }
          }
        }
      }
    }
  };

#pragma omp for nowait collapse(3)
  iter_each (_t2, this->t2) {
  iter_each (_ic3, this->ic3) {
  iter_each(_I2, this->I2) {
    bool is_Ir = this->Ir != V && _ic4 == this->ic4 - 1 &&
         _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    alignas(64) TrOpType aout[A][A][V];
    alignas(64) InputType ain[A][A][V];
    iter_each(_T, Tz) {
      readin(ain, _t2, _ic3, _I2, _T, is_Ir);
      ker_trans_input_(*this, aout, (InputType *)ain, 0, 0, 0, -1);
      __trans_input_post(&md2(atinput, _t2, 0), aout, Tz, _ic3, _I2, _T);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_input(
    TinputType *tinput, InputType *input, int _ic4)
{
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_blocked(tinput, input, _ic4);
  else if (this->input_fmt == nhwc)
    __trans_input_nhwc(tinput, input, _ic4);
  else
    __trans_input_nchw(tinput, input, _ic4);
}

// tweights:     oc4 | oc3, ic3, A, A, O2, I2, V, V
// tinputs:       t2 | A, A, ic3, I2, T, V
// toutput:  t2, oc4 | A, A, oc3, O2, T, V
Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::gemm(
    ToutputType *toutput, TinputType *tinput, TweightsType *tweights, int _t2, int Tz, int _ic4)
{
  auto ker_gemm = (_t2 == this->t2 - 1) ? ker_gemm0_ : ker_gemm_;

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
          ker_gemm(*this,
              &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
              &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
              &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
              nullptr, attr, 0, nullptr, nullptr, nullptr);
        }
        if (last_ic4) {
          auto attr = this->ic3 == 1 && this->ic4 == 1
                          ? set_attr(attr_, r_output_idx)
                          : attr_;
         if (this->Ir != V * this->Vx)
           attr = set_attr(attr, has_Ir_idx);
          ker_gemm(*this,
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
        ker_gemm(*this,
            &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
            &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
            &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
            nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
      if (last_ic4) {
        auto attr = this->ic3 == 1 && this->ic4 == 1
                        ? set_attr(attr_, r_output_idx)
                        : attr_;
        if (this->Ir != V * this->Vx)
          attr = set_attr(attr, has_Ir_idx);
        ker_gemm(*this,
            &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
            &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
            &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
            nullptr, attr, 0, nullptr, nullptr, nullptr);
      }
    }}}
  }
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::trans_input_u8(
    TscaleType *tinput_qt_scale, uint8_t *tinput_u8, TinputType *tinput,
    InputType *input) {
  if (input_is_bfmt_ || input_as_bfmt_)
    __trans_input_u8_blocked(tinput_qt_scale, tinput_u8, tinput, input);
  else
    el_error("Unimplemented: plain format input for int8");
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_input_u8_blocked(
    TscaleType *tinput_qt_scale, uint8_t *tinput_u8, TinputType *tinput,
    InputType *input) {
  MD2(uint8_t, atinput2_u8, tinput_u8, this->t2, A * A * this->T * this->ic3 * this->I2 * this->Vx * V);
  MD2(TscaleType, atinput_qt_scale2, tinput_qt_scale, this->t2, A * A * this->ic3 * 2 * this->T);
  MD2(TinputType, atinput2, tinput, this->t2, A * A * this->ic3 * this->I2 * this->Vx * this->T * V);
  MD8(InputType, ainput, input, this->n, this->ic4, this->ic3, this->I2, this->Vx, this->ih, this->iw, V);

#pragma omp for nowait collapse(4)
  iter_each (_t2, this->t2) {
  iter_each (_ic3, this->ic3) {
  iter_each (_I2, this->I2) {
  iter_each (_Vx, this->Vx) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD5(TscaleType, atinput_qt_scale, &md2(atinput_qt_scale2, _t2, 0), A, A, this->ic3, 2, Tz);
    MD5(TinputType, atinput5, &md2(atinput2, _t2, 0), this->ic3, this->I2, this->Vx, Tz, A * A * V);
    using Array = TrOpType[A][A][V];

    iter_each (_T, Tz) {
      int _n, _ih, _iw, _hA_start, _wA_start, _hA_end, _wA_end;
      t2spati(_t2, _T, _n, _ih, _iw, _hA_start, _hA_end, _wA_start, _wA_end);

      InputType *in = &md8(ainput, _n, 0, _ic3, _I2, _Vx, _ih, _iw, 0);
      auto aout = &md5(atinput5, _ic3, _I2, _Vx, _T, 0);
      if (_hA_start == 0 && _wA_start == 0 && _hA_end == A - 1
          && _wA_end == A - 1)
        ker_trans_input_(*this, *(Array *)aout, in, 0, A - 1, 0, A - 1);
      else
        ker_trans_input0_(
            *this, *(Array *)aout, in, _hA_start, _hA_end, _wA_start, _wA_end);
    }
  }}}}

#pragma omp barrier
#pragma omp for nowait collapse(4)
  iter_each (_t2, this->t2) {
  iter_each (_wA, A) {
  iter_each (_hA, A) {
  iter_each (_ic3, this->ic3) {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    MD7(TinputType, atinput7, &md2(atinput2, _t2, 0), this->ic3, this->I2, this->Vx, Tz, A, A, V);
    MD5(TscaleType, atinput_qt_scale, &md2(atinput_qt_scale2, _t2, 0),
        A, A, this->ic3, 2, Tz);
    MD7(uint8_t, atinput_u8, &md2(atinput2_u8, _t2, 0),
        A, A, this->ic3, this->I2, Tz, this->Vx, V);
    iter_each (_T, Tz) {
      __m<V> mmax, mmin;
      bool flush = true;
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
        __m<V> mcur = *(__m<V> *)&md7(atinput7, _ic3, _I2, _Vx, _T, _wA, _hA, 0);
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

      auto delta = max - min + 0.000001;
      float S = delta / INT8GEMM_TIN_MIN_MAX_QTSCALE;
      float repS = INT8GEMM_TIN_MIN_MAX_QTSCALE / delta;
      float z = std::ceil(- min * repS);
      md5(atinput_qt_scale, _wA, _hA, _ic3, 0, _T) = S;
      md5(atinput_qt_scale, _wA, _hA, _ic3, 1, _T) = z;

      __m<V> mrepS = _mm<V>::set1_ps(repS);
      __m<V> mz = _mm<V>::set1_ps(z);
      iter_each (_I2, this->I2) {
      iter_each (_Vx, this->Vx) {
        __m<V> f = *(__m<V> *)&md7(atinput7, _ic3, _I2, _Vx, _T, _wA, _hA, 0);
        __m<V> mmresf32 = f * mrepS + mz;
        __i<V> mmresu32 = _mm<V>::cvt_roundps_epu32(mmresf32, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
        __m128i mmresu8 = _mm<V>::cvtusepi32_epi8(mmresu32);
        _mm_store_si128((__m128i *)&md7(atinput_u8, _wA, _hA, _ic3, _I2, _T, _Vx, 0), mmresu8);
      }}
    }
  }}}}
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
      TscaleType *asrc_s, *asrc_z;
      if (!this->online_sampling_hp) {
        asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
        asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
      } else {
        asrc_s = &md5(asrc_scale, this->ic3, _wA, _hA, 0, 0);
        asrc_z = &md5(asrc_scale, this->ic3, _wA, _hA, 1, 0);
      }
      ker_gemm(*(elx_conv_params_t *)this,
          &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput, _wA, _hA, _ic3, 0, 0, 0),
          &md5(atweights, _oc3, _ic3, _wA, _hA, 0),
          nullptr, attr, asrc_s, asrc_z,
          &md6(aweights_scale, _oc3, _ic3, _wA, _hA, 0, 0),
          &md6(aweights_factor, _oc3, _ic3, _wA, _hA, 0, 0));
    }
    if (last_ic4) {
      auto attr = this->ic3 == 1 && this->ic4 == 1 ?
          set_attr(attr_, r_output_idx) : attr_;
      attr = set_attr(attr, l_output_idx);
      attr = set_attr(attr, c_output_idx);
      if (this->Ir != V * this->Vx)
        attr = set_attr(attr, has_Ir_idx);
      TscaleType *asrc_s, *asrc_z;
      if (!this->online_sampling_hp) {
        asrc_s = &md5(asrc_scale, 0, 0, 0, 0, 0);
        asrc_z = &md5(asrc_scale, 0, 0, 0, 1, 0);
      } else {
        asrc_s = &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 0, 0);
        asrc_z = &md5(asrc_scale, this->ic3 - 1, _wA, _hA, 1, 0);
      }
      ker_gemm(*(elx_conv_params_t *)this,
          &md6(atoutput, _wA, _hA, _oc3, 0, 0, 0),
          &md6(atinput, _wA, _hA, this->ic3 - 1, 0, 0, 0),
          &md5(atweights, _oc3, this->ic3 - 1, _wA, _hA, 0),
          nullptr, attr, asrc_s, asrc_z,
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
          if (this->Ir != V * this->Vx)
            attr = set_attr(attr, has_Ir_idx);
          ker_gemm(*(elx_conv_params_t *)this,
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
        if (this->Ir != V * this->Vx)
          attr = set_attr(attr, has_Ir_idx);
        ker_gemm(*(elx_conv_params_t *)this,
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
          if (this->Ir != V * this->Vx)
            attr = set_attr(attr, has_Ir_idx);
          ker_gemm(*(elx_conv_params_t *)this,
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
        if (this->Ir != V * this->Vx)
          attr = set_attr(attr, has_Ir_idx);
        ker_gemm(*(elx_conv_params_t *)this,
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
      if (this->Ir != V * this->Vx)
        attr = set_attr(attr, has_Ir_idx);
      ker_gemm(*(elx_conv_params_t *)this,
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
      if (this->Ir != V * this->Vx)
        attr = set_attr(attr, has_Ir_idx);
      ker_gemm(*(elx_conv_params_t *)this,
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
      if (this->Ir != V * this->Vx)
        attr = set_attr(attr, has_Ir_idx);
      ker_gemm(*this, &md6(atoutput6, _wA, _hA, _oc3, 0, 0, 0),
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

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_nchw(
    OutputType * __restrict output, ToutputType * __restrict toutput,
    BiasType * __restrict bias, int Tz, int _t2, int _oc4, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  alignas(64) OutputType aout[A - K + 1][A - K + 1][V];
  SET_EPI32(this->oh * this->ow)

  auto writeout = [&](OutputType aout[A - K + 1][A - K + 1][V],
                      int _oc3, int _O2, int _T, bool is_Or) {
    MD2(OutputType, aoutput0, output, this->n, this->oc * this->oh * this->ow);
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
    MD6(OutputType, aoutput1, &md2(aoutput0, _n, 0), this->oc4, this->oc3,
        this->O2, this->V, this->oh, this->ow);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
            iter_each (_V, this->Or)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
            iter_each (_V, this->Or)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || _ic4 > 0) {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr int scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md6(aoutput1, _oc4, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
          }
        }
      }
    }
  };

  union {__m<V> vin; TrOpType ain[V];} In[A][A];
  using Array = TrOpType[A][A][V];

  iter_each (_oc3, this->oc3) {
  iter_each (_O2, this->O2) {
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1 &&
                 _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;
    iter_each(_T, Tz) {
      iter_each(_wA, A) {
      iter_each(_hA, A) {
        if (std::is_same<ToutputType, float>::value) {
          In[_wA][_hA].vin =
              _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        } else {
          auto fp16v = _mm<V / 2>::load_si256(
              (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
          In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
        }
      }}
      ker_trans_output_(*this, (OutputType *)aout, *(Array *)&In,
                        (_ic4 == this->ic4 - 1 || _ic4 == -1)
                            ? &md3(abias, _oc3, _O2, 0)
                            : nullptr,
                        0, -1);

      writeout(aout, _oc3, _O2, _T, is_Or);
    }
  }}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_blocked(
    OutputType *output, ToutputType *toutput, BiasType *bias,
    int Tz, int _t2, int _oc4, int _ic4)
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
          In[_wA][_hA].vin =
              _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        } else {
          auto fp16v = _mm<V / 2>::load_si256(
              (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
          In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
        }
      }}

      auto _n = t2spato_o.n_;
      auto _oh = t2spato_o.t_;
      auto _ow = t2spato_o.l_;
      OutputType *out = &md7(aoutput, _n, _oc4, _oc3, _O2, _oh, _ow, 0);

      if (t2spato_o.is_border())
        ker_trans_output_tail(*this, out, *(Array *)&In,
            (_ic4 == -1 || _ic4 == this->ic4 - 1)
            ? &md3(abias, _oc3, _O2, 0) : nullptr, t2spato_o.d_, t2spato_o.r_);
      else
        ker_trans_output(*this, out, *(Array *)&In,
            (_ic4 == -1 || _ic4 == this->ic4 - 1)
            ? &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);

      ++ t2spato_o;
    }
  }}
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_output_nhwc(
    OutputType *output, ToutputType *toutput, BiasType *bias,
    int Tz, int _t2, int _oc4, int _ic4)
{
  auto ker_trans_output = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output_acc_ : ker_trans_output_;
  auto ker_trans_output_tail = (this->with_ip_sum || _ic4 > 0) ?
      ker_trans_output0_acc_ : ker_trans_output0_;

  // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
  MD6(ToutputType, atoutput, toutput, A, A, this->oc3, this->O2, Tz, V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);
  MD4(OutputType, aoutput0, output, this->n, this->oh, this->ow, this->oc);

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
          In[_wA][_hA].vin =
              _mm<V>::load_ps(&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
        } else {
          auto fp16v = _mm<V / 2>::load_si256(
              (__m256i *)&md6(atoutput, _wA, _hA, _oc3, _O2, _T, 0));
          In[_wA][_hA].vin = _mm<V>::cvtph_ps(fp16v);
        }
      }}
      auto _n = t2spato_o.n_;
      auto _oh = t2spato_o.t_;
      auto _ow = t2spato_o.l_;
      MD4(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), this->oc4,
          this->oc3, this->O2, V);
      OutputType *out = &md4(aoutput1, _oc4, _oc3, _O2, 0);
      if (t2spato_o.is_border())
        ker_trans_output_tail(*this, out, *(Array *)&In,
            (_ic4 == -1 || _ic4 == this->ic4 - 1)
            ? &md3(abias, _oc3, _O2, 0) : nullptr, t2spato_o.d_, t2spato_o.r_);
      else
        ker_trans_output(*this, out, *(Array *)&In,
            (_ic4 == -1 || _ic4 == this->ic4 - 1)
            ? &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);

      ++ t2spato_o;
    }
  }}
}

Template_elx_conv_wino_t void Instance_elx_conv_wino_t::trans_output(
    OutputType *output, ToutputType *toutput, BiasType *bias, int Tz, int _t2,
    int _oc4, int _ic4) {
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, Tz, _t2, _oc4, _ic4);
  else if (this->output_fmt == nhwc)
    __trans_output_nhwc(output, toutput, bias, Tz, _t2, _oc4, _ic4);
  else
    __trans_output_nchw(output, toutput, bias, Tz, _t2, _oc4, _ic4);
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_blocked(
    OutputType *output, ToutputType *toutput, BiasType *bias, int _oc4, int _ic4)
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
      OutputType *out = &md7(aoutput, _n, _oc4, _oc3, _O2, _oh, _ow, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_output_tail(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, _hOA_end, _wOA_end);
      else
        ker_trans_output(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::__trans_output_nhwc(
    OutputType *output, ToutputType *toutput, BiasType *bias, int _oc4, int _ic4)
{
  // A, A, oc3, O2, T, V -> n, oh, ow, oc
  MD4(OutputType, aoutput0, output, this->n, this->oh, this->ow, this->oc);
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
      MD4(OutputType, aoutput1, &md4(aoutput0, _n, _oh, _ow, 0), this->oc4,
          this->oc3, this->O2, V);
      OutputType *out = &md4(aoutput1, _oc4, _oc3, _O2, 0);

      if (_hOA_end < A - K || _wOA_end < A - K)
        ker_trans_output_tail(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, _hOA_end, _wOA_end);
      else
        ker_trans_output(
            *this, out, *(Array *)&In, (_ic4 == -1 || _ic4 == this->ic4 - 1) ?
            &md3(abias, _oc3, _O2, 0) : nullptr, A - K, A - K);
    }
  }}}
}
Template_elx_conv_wino_t void Instance_elx_conv_wino_t::__trans_output_nchw(
    OutputType *__restrict output, ToutputType *__restrict toutput,
    BiasType *bias, int _oc4, int _ic4) {
  // A, A, oc3, O2, T, V -> n, OC, oh, ow
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);
  MD2(ToutputType, atoutput2, toutput, this->t2,
      A * A * this->T * this->oc3 * this->O2 * V);

  SET_EPI32(this->oh * this->ow)

  auto writeout = [&](OutputType aout[A - K + 1][A - K + 1][V],
                      int _t2, int _oc3, int _O2, int _T, bool is_Or) {
    MD2(OutputType, aoutput0, output, this->n, this->oc * this->oh * this->ow);
    int _n, _oh, _ow, _hOA_end, _wOA_end;
    t2spato(_t2, _T, _n, _oh, _ow, _hOA_end, _wOA_end);
    MD6(OutputType, aoutput1, &md2(aoutput0, _n, 0),
        this->oc4, this->oc3, this->O2, V, this->oh, this->ow);

    for (int _wA = 0; _wA <= _wOA_end; ++_wA) {
      for (int _hA = 0; _hA <= _hOA_end; ++_hA) {
        if (is_Or) {
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, this->Or)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                  += aout[_hA][_wA][_V];
          } else {
#pragma omp simd
            iter_each (_V, this->Or)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                  = aout[_hA][_wA][_V];
          }
        } else {
          if ((this->with_ip_sum && !output_as_bfmt_) || (_ic4 > 0)) {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                 += aout[_hA][_wA][_V];
          } else if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __m<V> t = _mm<V>::load_ps(aout[_hA][_wA]);
            constexpr auto scale = sizeof(OutputType);
            _mm<V>::i32scatter_ps(
                &md6(aoutput1, _oc4, _oc3, _O2, 0, _oh + _hA, _ow + _wA),
                vindex, t, scale);
          } else {
#pragma omp simd
            iter_each (_V, V)
              md6(aoutput1, _oc4, _oc3, _O2, _V, _oh + _hA, _ow + _wA)
                = aout[_hA][_wA][_V];
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
    bool is_Or = this->Or != V && _oc4 == this->oc4 - 1 &&
                 _oc3 == this->oc3 - 1 && _O2 == this->O2 - 1;
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
          ? &md3(abias, _oc3, _O2, 0) : nullptr, 0, -1);
      writeout(aout, _t2, _oc3, _O2, _T, is_Or);
    }
  }}}
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::trans_output(
    OutputType *output, ToutputType *toutput, BiasType *bias, int _oc4, int _ic4)
{
  if (output_is_bfmt_ || output_as_bfmt_)
    __trans_output_blocked(output, toutput, bias, _oc4, _ic4);
  else if (this->output_fmt == nhwc)
    __trans_output_nhwc(output, toutput, bias, _oc4, _ic4);
  else
    __trans_output_nchw(output, toutput, bias, _oc4, _ic4);
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
