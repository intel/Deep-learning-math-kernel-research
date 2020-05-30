#include "el_parallel.hpp"
#include "elx_conv_wino.hpp"
#include "elx_conv_wino_bind.hpp"
#include "elx_conv_wino_xopt.hpp"

namespace euler {

Template_elx_conv_wino_t Instance_elx_conv_wino_t::elx_conv_wino_t(
    eld_conv_t &dc)
    : elx_conv_t(dc) {
  // TODO: error when V!=16 && fmt=OIhw16i16o
  xopt_ = ep.execution_mode;
  mthr_ = ep.nthreads;

  ep.Vx = 1;
  ep.V1 = V / ep.Vx;
  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;

  ep.ht = (ep.oh + A - K) / (A - K + 1);
  ep.wt = (ep.ow + A - K) / (A - K + 1);
  ep.nt = ep.ht * ep.wt;
  ep.t = ep.nt * ep.n;

  // TODO: santize user settings
  if (ep.O == 0) ep.O = 1; // TODO: O selection
  if (ep.O1 == 0) ep.O1 = 1; // TODO: O1 selection
  if (ep.I2 == 0) ep.I2 = 1; // TODO: I2 selection
  if (ep.T == 0)  ep.T = 1; // TODO: T selection
  ep.O2 = ep.O * ep.O1;

  // Tailing
  ep.Tr = ep.t % ep.T ? ep.t % ep.T : ep.T;
  ep.Ir = ep.ic % V ? ep.ic % V : V;
  ep.Or = ep.oc % V ? ep.oc % V : V;

  is_first_run_ = true;
  inference_acc_ = false;
  inference_acc_ = ep.prop_kind == forward_inference;

  ep.O4 = ep.O4 == 0 ? 1 : ep.O4;
  ep.I4 = ep.I4 == 0 ? 1 : ep.I4;

  // further divide packed oc/ic
  ep.O3 = ep.oc2 / ep.O2;
  ep.I3 = ep.ic2 / ep.I2;

  ep.t2 = (ep.t + ep.T - 1) / ep.T;

  if (xopt_ == 0) {
    auto t2_th = ep.t2 / mthr_;
    xopt_ = t2_th > 1 ? 0xa061 : 0xa033;
  }

  if ((xopt_ == 0xa073 || ep.with_ip_sum)
      && ep.with_relu && !output_is_bfmt_) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  prepare_execute_opt();
  bind_execute_functions();
  trans_input.setup(&ep);
  trans_weights.setup(&ep);
  gemm.setup(&ep);
  trans_output.setup(&ep);

  if (V * ep.I2 * ep.I3 * ep.I4 != ep.IC) {
    el_error("V * I2 * I3 * I4 != ep.IC\n)");
  }

  if (V * ep.O2 * ep.O3 * ep.O4 != ep.OC) {
    el_error("V * O2 * O3 * O4 != ep.OC\n)");
  }

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, t=%d", ep.T, ep.Tr, ep.t2, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, Vx=%d, I2=%d, I3=%d, I4=%d, IC=%d",
         V, ep.Ir, ep.Vx, ep.I2, ep.I3, ep.I4, ep.IC);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, OC=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1, ep.O3, ep.O4, ep.OC);
}

Template_elx_conv_wino_t
int Instance_elx_conv_wino_t::prepare_execute_opt()
{
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t binput_size = 0, bweights_size = 0, boutput_size = 0;

  if (xopt_ & FUS_O) {
    ep.O3 /= ep.O4;
    if (V * ep.O2 * ep.O3 * ep.O4 != ep.OC) {
      el_error("Config error!");
      return -1;
    }
  }
  if (xopt_ & FUS_I) {
    ep.I3 /= ep.I4;
    if (V * ep.I2 * ep.I3 * ep.I4 != ep.IC) {
      el_error("Config error!");
      return -1;
    }
  }

  input_is_bfmt_ = ep.input_fmt == nChw16c; // nChw8c
  weights_is_bfmt_ = ep.weights_fmt == OIhw16i16o;
  output_is_bfmt_ = ep.output_fmt == nChw16c;
  input_as_bfmt_ = ep.input_fmt == nchw && ep.input_as_blocked;
  weights_as_bfmt_ = ep.input_fmt == oihw && ep.weights_as_blocked;
  output_as_bfmt_ = ep.output_fmt == nchw && ep.output_as_blocked;
  is_bfmt_ = input_is_bfmt_ && weights_is_bfmt_ && output_is_bfmt_;

  if (ep.Or != V && ep.output_fmt == nhwc) {
    el_error("Unimplemented: nhwc output with Or");
  }

  if (input_as_bfmt_)
    binput_size = ep.n * ep.IC * ep.ih * ep.iw * sizeof(InputType);
  if (weights_as_bfmt_)
    bweights_size = ep.OC * ep.IC * ep.kh * ep.kw * sizeof(WeightsType);
  if (output_as_bfmt_)
    boutput_size = ep.n * ep.OC * ep.oh * ep.ow * sizeof(OutputType);

  tweights_ = nullptr;
  tinput_ = nullptr;
  toutput_ = nullptr;
  binput_ = nullptr;
  bweights_ = nullptr;
  boutput_ = nullptr;

  switch (xopt_) {
  case 0xa000:
    tweights_size = A * A * ep.IC * ep.OC * sizeof(TweightsType);
    tinput_size = A * A * ep.IC * ep.t * sizeof(TinputType);
    toutput_size = A * A * ep.OC * ep.t * sizeof(ToutputType);
    break;
  case 0xa033:
    tweights_size = A * A * ep.IC * ep.OC * sizeof(TweightsType);
    tinput_size = A * A * ep.I3 * ep.I2 * V * ep.t * sizeof(TinputType);
    toutput_size = A * A * (ep.OC / ep.O4) * ep.t * sizeof(ToutputType);
    break;
  case 0xa061:
    tweights_size = A * A * ep.IC * ep.OC * sizeof(TweightsType);
    tinput_size = A * A * ep.IC * ep.T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (ep.OC / ep.O4) * ep.T * mthr_ * sizeof(ToutputType);
    break;
  case 0xa071:
    tweights_size = A * A * ep.IC * ep.OC * sizeof(TweightsType);
    tinput_size = A * A * (ep.IC / ep.I4) * ep.T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * ep.OC * ep.t * sizeof(ToutputType);
    break;
  case 0xa073:
    tweights_size = A * A * ep.IC * ep.OC * sizeof(TweightsType);
    tinput_size = A * A * (ep.IC / ep.I4) * ep.T * mthr_ * sizeof(TinputType);
    toutput_size = A * A * (ep.OC / ep.O4) * ep.T * mthr_ * sizeof(ToutputType);
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

// fp32-f32f32f32
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 4, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 5, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 6, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32, 7, 3, 16, ISA_AVX512>;

// fp32-f16f16f16
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 4, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 5, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 6, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP32, conv_impl::FP32_F16iwo, 7, 3, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// fp16-f32f16f16
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 4, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 5, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 6, 3, 16, ISA_AVX512>;
template class elx_conv_wino_t<conv::FP16, conv_impl::FP32_F16wob, 7, 3, 16, ISA_AVX512>;
#endif

} // namespace euler
