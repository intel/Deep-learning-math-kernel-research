#include "el_parallel.hpp"
#include "elx_conv_wino.hpp"

namespace euler {

Template_elx_conv_wino_t Instance_elx_conv_wino_t::elx_conv_wino_t(
    eld_conv_t &dc)
    : elx_conv_t(dc) {
  // TODO: error when V!=16 && fmt=OIhw16i16o
  xopt_ = this->execution_mode;

  this->Vx = 1;
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

  prepare_execute_opt();
  bind_execute_functions();
  trans_input.setup(this);
  trans_weights.setup(this);
  gemm.setup(this);
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
  if ((xopt_ == 0xa073 || xopt_ == 0xa07b || this->with_ip_sum)
      && this->with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  if (V * this->I2 * this->ic3 * this->ic4 != this->IC) {
    el_error("V * I2 * ic3 * ic4 != this->IC\n)");
  }

  if (V * this->O2 * this->oc3 * this->oc4 != this->OC) {
    el_error("V * O2 * oc3 * oc4 != this->OC\n)");
  }
#endif
}

Template_elx_conv_wino_t
int Instance_elx_conv_wino_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

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

  workspace_size_ = tweights_size_;
  scratch_size_ = tinput_size_ + toutput_size_
      + binput_size_ + bweights_size_ + boutput_size_;

  if (xopt_ == 0xa079 || xopt_ == 0xa07b) {
    scratch_size_ += tweights_size_;
    workspace_size_ = 0;
  }

  return 0;
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    tweights_ = (TweightsType *)base;
  }
}

Template_elx_conv_wino_t
void Instance_elx_conv_wino_t::set_scratch_buffers(void *base)
{
  if (base != nullptr) {
    tinput_ = (TinputType *)base;
    toutput_ = (ToutputType *)((char *)tinput_ + tinput_size_);
    binput_ = (InputType *)((char *)toutput_ + toutput_size_);
    bweights_ = (WeightsType *)((char *)binput_ + binput_size_);
    boutput_ = (OutputType *)((char *)bweights_ + bweights_size_);
  }
}

Template_elx_conv_wino_t
Instance_elx_conv_wino_t::~elx_conv_wino_t()
{
}

} // namespace euler
