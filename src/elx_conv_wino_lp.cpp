#include "el_parallel.hpp"
#include "elx_conv_wino_lp.hpp"

namespace euler {

const float INT8GEMM_TWT_QTSCALE = 127.0;
const float INT8GEMM_TIN_MIN_MAX_QTSCALE = 255.0;

Template_elx_conv_wino_lp_t Instance_elx_conv_wino_lp_t::elx_conv_wino_lp_t(
    eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // TODO: error when V!=16 && fmt=OIhw16i16o
  xopt_ = this->execution_mode;

  this->Vx = 4;
  this->V1 = V / this->Vx;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  this->ht = (this->oh + A - K) / (A - K + 1);
  this->wt = (this->ow + A - K) / (A - K + 1);
  this->nt = this->ht * this->wt;
  this->t = this->nt * this->n;

  // TODO: santize user settings
  if (this->O == 0) this->O = 1; // TODO: O selection
  if (this->O1 == 0) this->O1 = 1; // TODO: O1 selection
  if (this->I2 == 0) this->I2 = 1; // TODO: I2 selection
  if (this->T == 0)  this->T = 1; // TODO: T selection
  this->O2 = this->O * this->O1;

  // Tailing
  this->Tr = this->t % this->T ? this->t % this->T : this->T;
  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  is_first_run_ = true;
  inference_acc_ = false;
  mthr_ = el_get_max_threads();
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
  this->ic3 = this->ic2 / this->I2;

  this->t2 = (this->t + this->T - 1) / this->T;

  if (this->sampling_kind != CALIBRATED) {
    el_error("Winograd: to enable sampling from elk_u8s8_gemm_otj");
  }
  prepare_quant_calibration(dc);

  prepare_execute_opt();
  bind_execute_functions();
  trans_input_u8.setup(this);
  trans_weights_s8.setup(this);
  u8s8_gemm.setup(this);
  trans_output.setup(this);

  // dbg
  printf("############################################################\n");
  printf("T=%d, Tr=%d, t2=%d, t=%d\n", this->T, this->Tr, this->t2, this->t);
  printf("V=%d, Ir=%d, Vx=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      V, this->Ir, this->Vx, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, OC=%d\n",
      V, this->Or, this->O2, this->O, this->O1, this->oc3, this->oc4, this->OC);

#ifdef DEBUG
  if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
    el_warn("V * I2 * ic3 * ic4 != this->IC\n Force ic4 = IC / (V * I2 * ic3)");
    this->ic4 = this->IC / (V * this->I2 * this->ic3);
  }

  if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
    el_warn("V * O2 * oc3 * oc4 != this->OC\n Force oc4 = OC / (V * O2 * oc3)");
    this->oc4 = this->OC / (V * this->O2 * this->oc3);
  }
#else
  if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
    el_error("V * I2 * ic3 * ic4 != this->IC\n)");
  }

  if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
    el_error("V * O2 * oc3 * oc4 != this->OC\n)");
  }
#endif
}

Template_elx_conv_wino_lp_t
int Instance_elx_conv_wino_lp_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;
  size_t tinput_u8_size = 0, tinput_quant_scale_size = 0,
      tweights_s8_size = 0,
      tweights_quant_scale_size = 0, tweights_quant_factor_size = 0;

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
  if (this->Ir != V && this->input_fmt == nhwc) {
    el_error("Unimplemented: nhwc input with Ir");
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
  tinput_quant_scale_ = nullptr;
  tweights_s8_ = nullptr;
  tweights_quant_scale_ = nullptr;
  tweights_quant_factor_ = nullptr;

  switch (xopt_) {
  case 0xa133:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * this->t * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->t * sizeof(ToutputType);
    tinput_u8_size = A * A * (this->IC / this->ic4) * this->t * sizeof(uint8_t);
    tinput_quant_scale_size = this->t * this->ic3 * 2 * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_quant_scale_size = this->ic4 * this->OC * A * A * sizeof(TscaleType);
    tweights_quant_factor_size = this->ic4 * this->OC * A * A * sizeof(TscaleType);
    break;
  case 0xa161:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    if (this->sampling_kind == COARSE)
      tinput_size = this->IC * A * A * this->T * mthr_ * sizeof(TinputType);
    else
      tinput_size = A * A * this->I2 * V * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    tinput_u8_size = A * A * this->IC * mthr_ * this->T * sizeof(uint8_t);
    tinput_quant_scale_size = mthr_ * 2 * this->ic3 * this->T * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);

    // FIXME: To implement OC sampling for weights transformation.
    // Current weights transformation includes a sampling involving OC and ic4.
    // As to a161, sampling scope should be only in OC. However, ic4 must be 1
    // in current execution mode. So far, we temporarily borrow OC and ic4
    // sampling for weights transformation, where ic4 is 1.
    tweights_quant_scale_size = this->OC * A * A * sizeof(TscaleType);
    tweights_quant_factor_size = this->OC * A * A * sizeof(TscaleType); // * this->ic4
    break;
  case 0xa173:
    tweights_size = A * A * this->IC * this->OC * sizeof(TweightsType);
    tinput_size = A * A * (this->IC / this->ic4) * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (this->OC / this->oc4) * this->T * mthr_ * sizeof(ToutputType);
    tinput_u8_size = A * A * (this->IC / this->ic4) * mthr_ * this->T * sizeof(uint8_t);
    tinput_quant_scale_size = mthr_ * 2 * this->ic3 * this->T * A * A * sizeof(TscaleType);
    tweights_s8_size = tweights_size / sizeof(TweightsType);
    tweights_quant_scale_size = this->ic4 * this->OC * A * A * sizeof(TscaleType);
    tweights_quant_factor_size = this->ic4 * this->OC * A * A * sizeof(TscaleType);
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
  tinput_quant_scale_size_ = tinput_quant_scale_size > 0 ? alignup(tinput_quant_scale_size, align) : 0;
  tweights_s8_size_ = tweights_s8_size > 0 ? alignup(tweights_s8_size, align) : 0;
  tweights_quant_scale_size_ = tweights_quant_scale_size > 0 ? alignup(tweights_quant_scale_size, align) : 0;
  tweights_quant_factor_size_ = tweights_quant_factor_size > 0 ? alignup(tweights_quant_factor_size, align) : 0;

  workspace_size_ = tweights_size_ + tweights_s8_size_
      + tweights_quant_scale_size_ + tweights_quant_factor_size_;
  scratch_size_ = estl::max(tinput_size_, toutput_size_)
      + binput_size_ + bweights_size_ + boutput_size_ + tinput_u8_size_;

  if (this->sampling_kind == CALIBRATED)
    workspace_size_ += tinput_quant_scale_size_;
  else
    scratch_size_ += tinput_quant_scale_size_;

  return 0;
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    tweights_ = (TweightsType *)base;
    // int8gemm supported in weights reuse case only.
    tweights_quant_scale_ = (TscaleType *)((char *)tweights_ + tweights_size_);
    tweights_quant_factor_ = (TscaleType *)((char *)tweights_quant_scale_ + tweights_quant_scale_size_);
    if (this->sampling_kind == CALIBRATED) {
      tinput_quant_scale_ = (TscaleType *)((char *)tweights_quant_factor_ + tweights_quant_factor_size_);
      tweights_s8_ = (int8_t *)((char *)tinput_quant_scale_ + tinput_quant_scale_size_);
    } else {
      tweights_s8_ = (int8_t *)((char *)tweights_quant_factor_ + tweights_quant_factor_size_);
    }
  }
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::set_scratch_buffers(void *base)
{
  if (base != nullptr) {
    tinput_ = (TinputType *)base;
    toutput_ = (ToutputType *)tinput_;
    binput_ = (InputType *)((char *)toutput_ + estl::max(tinput_size_, toutput_size_));
    bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
    boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
    if (this->sampling_kind == CALIBRATED) {
      tinput_u8_ = (uint8_t *)((char *)boutput_ + boutput_size_);
    } else {
      tinput_quant_scale_ = (TscaleType *)((char *)boutput_ + boutput_size_);
      tinput_u8_ = (uint8_t *)((char *)tinput_quant_scale_ + tinput_quant_scale_size_);
    }
  }
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::prepare_quant_calibration(eld_conv_t &dc)
{
  this->tinput_quant_S = dc.wino_tinput_quant.scale;
  this->tinput_quant_z = dc.wino_tinput_quant.z;

  if (this->sampling_kind == CALIBRATED) {
    if (this->input_quant_S == EL_NO_CALI ||
        this->input_quant_z == EL_NO_CALI) {
      this->sampling_kind = FINE;
      return;
    }
    this->input_quant_repS = 1 / this->input_quant_S;
    this->input_quant_z = (float)std::ceil(this->input_quant_z);
    this->tinput_quant_repS = 1 / this->tinput_quant_S;
    this->tinput_quant_z = (float)std::ceil(this->tinput_quant_z);
    this->output_quant_repS = 1 / this->output_quant_S;
    this->output_quant_z = (float)std::ceil(this->output_quant_z);
  }
}

Template_elx_conv_wino_lp_t
Instance_elx_conv_wino_lp_t::~elx_conv_wino_lp_t()
{
}

} // namespace euler
