#include <cfenv>
#include <cmath>
#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_int8_conv_direct_depthwise.hpp"
#include "elx_int8_conv_direct_depthwise_bind.hpp"
#include "elx_int8_conv_direct_depthwise_xopt.hpp"

namespace euler {

// depth-wise
Template_elx_int8_conv_direct_depthwise_t
Instance_elx_int8_conv_direct_depthwise_t::elx_int8_conv_direct_depthwise_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = ep.execution_mode;
  mthr_ = estl::max_concurrency();

  ep.grp = ep.g;
  ep.Vx = 1;
  ep.V1 = V / ep.Vx;
  ep.ocg = ep.oc / ep.g;
  ep.icg = ep.ic / ep.g;

  bool shape_ok = ep.icg == 1 && ep.ocg == 1 &&
                  estl::any_of(ep.kh, 3) &&
                  estl::any_of(ep.kw, 3) &&
                  estl::any_of(ep.ws, 1, 2) &&
                  estl::any_of(ep.hs, 1, 2) &&
                  estl::any_of(ep.lp, 0, 1) &&
                  estl::any_of(ep.rp, 0, 1) &&
                  estl::any_of(ep.tp, 0, 1) &&
                  estl::any_of(ep.bp, 0, 1);
  if (!shape_ok) {
    el_error("direct_depthwise: int8: shape not supported");
  }

  // compute multiple groups in one FMA
  // vector multi-group number
  ep.G = ALIGNUP(ep.g, V);
  ep.vmg = V / ep.ocg;
  ep.G2 = ep.G / ep.G3 / V;
  ep.g23 = ep.G2 * ep.G3;
  if (V * ep.G2 * ep.G3 != ep.G) {
    el_error("conv: g blocking error for depthwise conv");
  }
  if (ep.O != 1) {
    ep.O = 1;
    el_warn("conv: group: O!=1 found for vector multi-group");
  }
  ep.ic /= ep.g;
  ep.oc /= ep.g;

  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  if (ep.T == 0) ep.T = 1;
  ep.I2 = 1;

  ep.O4 = 1;
  ep.O3 = 1;
  ep.O = 1;
  ep.O1 = 1;
  ep.O2 = ep.O * ep.O1;
  
  ep.I4 = 1;
  ep.I3 = 1;
  ep.I2 = 1;

  ep.ic2 = ep.G / V;
  ep.oc2 = ep.G / V;

  xopt_ = 0xc160;

  // n, t2, (T, Tr)
  ep.ht = ep.oh;
  ep.wt = (ep.ow + ep.T - 1) / ep.T;
  ep.Tr = ep.ow % ep.T ? ep.ow % ep.T : ep.T;
  ep.nt = ep.oh * ep.ow;
  ep.t2 = ep.nt / ep.T;
  ep.t  = ep.nt * ep.n;

  if (ep.T < ep.lp || ep.Tr < ep.rp) {
    el_error("direct_depthwise: (T,Tr) < (lp,rp)");
  }
  bool format_ok = estl::any_of(ep.weights_fmt, ghwio, goihw) &&
                   estl::any_of(ep.input_fmt, nchw, nChw16c) &&
                   estl::any_of(ep.output_fmt, nchw, nChw16c);
  if (!format_ok) {
    el_error("direct_depthwise: format not supported");
  }

  // No Ir/Or
  ep.Ir = 0;
  ep.Or = 0;
  ep.ormask = 0;

  // IC = V = G * C; ic_orig = g * G * C
  if (ep.I4 * ep.I3 * ep.I2 * V != ep.IC) {
    el_error("IC blocking error");
  }
  // OC = V = G * C; oc_orig = g * G * C
  if (ep.O4 * ep.O3 * ep.O2 * V != ep.OC) {
    el_error("OC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  inference_acc_ = ep.prop_kind == forward_inference;

  attr_ = ep.with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  attr_ = ep.with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;
  attr_ = ep.with_relu ? set_bit(attr_, AT_RELU_MASK) : attr_;

  prepare_quant_calibration(dc);
  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d, G2=%d, G3=%d, g=%d, G=%d",
         V, ep.Ir, ep.I2, ep.I3, ep.I4, ep.IC, ep.G2, ep.G3, ep.g, ep.G);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC);
}

Template_elx_int8_conv_direct_depthwise_t void
Instance_elx_int8_conv_direct_depthwise_t::prepare_quant_calibration(
    eld_conv_t &dc) {
  ep.input_quant_S = dc.input_quant.scale;
  ep.input_quant_repS = 1.0f / dc.input_quant.scale;
  ep.input_quant_z = dc.input_quant.z;
  ep.output_quant_S = dc.output_quant.scale;
  ep.output_quant_repS = 1.0f / dc.output_quant.scale;
  ep.output_quant_z = dc.output_quant.z;

  if (ep.sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_int8_conv_direct_depthwise_t
int Instance_elx_int8_conv_direct_depthwise_t::prepare_execute_opt()
{
  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;

  switch (xopt_) {
  case 0xc160:
    tweights_size_ = ep.G * ep.kh * KW * sizeof(TweightsType);
    input_scale_size_ = 2 * ep.T * sizeof(float);
    weights_scale_size_ = ep.G * sizeof(float);
    weights_shift_size_ = ep.G * sizeof(float);
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size_ > 0)
    tweights_size_ += WEIGHTS_MAX_PRELOAD * V;

  workspace_size_ = tweights_size_ + input_scale_size_ +
    weights_scale_size_ + weights_shift_size_;
  scratch_size_ = 0;

  return 0;
}

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    weights_scale_ = (float *)base;
    weights_shift_ = (float *)((char *)weights_scale_ + weights_scale_size_);
    input_scale_ = (float *)((char *)weights_shift_ + weights_shift_size_);
    tweights_s8_ = (int8_t *)((char *)input_scale_ + input_scale_size_);
  }
}

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::set_scratch_buffers(void *base)
{
}

Template_elx_int8_conv_direct_depthwise_t
Instance_elx_int8_conv_direct_depthwise_t::~elx_int8_conv_direct_depthwise_t()
{
}

// weights: g23, V, kh, kw
// tweights: g23, kh, V, KW // kw-padded goih16g4w
Template_elx_int8_conv_direct_depthwise_t void
Instance_elx_int8_conv_direct_depthwise_t::trans_weights_3x3(
    float *weights_scale, float *weights_shift, int8_t *tweights_s8,
    WeightsType *weights, BiasType *bias)
{
  // absmax
  estl::parallel_for<2>([&](int _g23, int _V) {
    MD4(WeightsType, aweights, weights, ep.g23, V, ep.kh, ep.kw);
    MD2(float, aweights_scale, weights_scale, ep.g23, V);
    float absmax = 0.0;
    iter_each (_kh, ep.kh) {
      iter_each (_kw, ep.kw) {
        float val = md4(aweights, _g23, _V, _kh, _kw);
        absmax = estl::max(std::abs(val), absmax);
      }
    }
    md2(aweights_scale, _g23, _V) = absmax;
  }, ep.g23, V);

  // quantization
  std::fesetround(FE_TONEAREST);
  estl::parallel_for<4>([&](int _g23, int _kh, int _V, int _kw) {
    MD4(int8_t, atweights_s8, tweights_s8, ep.g23, ep.kh, V, KW);
    MD4(WeightsType, aweights, weights, ep.g23, V, ep.kh, ep.kw);
    MD2(float, aweights_scale, weights_scale, ep.g23, V);
    if (_kw < ep.kw) {
      // scale
      auto t0 = md4(aweights, _g23, _V, _kh, _kw);
      auto absmax = md2(aweights_scale, _g23, _V);
      t0 = t0 * EL_INT8_MAX / absmax;
      // round & store
      md4(atweights_s8, _g23, _kh, _V, _kw) = (int8_t)std::rint(t0);
    } else {
      md4(atweights_s8, _g23, _kh, _V, _kw) = 0;
    }
  }, ep.g23, ep.kh, V, KW);

  // weights-scale
  __m<V> mmscale = _mm<V>::set1_ps(EL_INT8_MAX);
  estl::parallel_for<1>([&](int _g23) {
    MD2(float, aweights_scale, weights_scale, ep.g23, V);
    auto t0 = *(__m<V> *)&md2(aweights_scale, _g23, 0);
    *(__m<V> *)&md2(aweights_scale, _g23, 0) = t0 / mmscale;
  }, ep.g23);

  // weights-acc
  estl::parallel_for<2>([&](int _g23, int _V) {
    MD4(int8_t, atweights_s8, tweights_s8, ep.g23, ep.kh, V, KW);
    MD2(float, aweights_shift, weights_shift, ep.g23, V);
    int acc = 0;
    iter_each(_kh, ep.kh) {
      iter_each(_kw, ep.kw) {
        acc += md4(atweights_s8, _g23, _kh, _V, _kw);
      }
    }
    md2(aweights_shift, _g23, _V) = acc;
  }, ep.g23, V);

  // combine with output restore
  auto out_repS = _mm<V>::set1_ps(ep.output_quant_repS);
  auto out_z = _mm<V>::set1_ps(ep.output_quant_z);
  auto input_S = _mm<V>::set1_ps(ep.input_quant_S);
  auto input_z = _mm<V>::set1_ps(ep.input_quant_z);

  estl::parallel_for<1>([&](int _g23) {
    MD2(float, aweights_scale, weights_scale, ep.g23, V);
    __m<V> &qs = *(__m<V> *)&md2(aweights_scale, _g23, 0);
    if (std::is_same<OutputType, float>::value) {
      qs = input_S * qs;
    } else {
      qs = input_S * qs * out_repS;
    }
  }, ep.g23);

  estl::parallel_for<1>([&](int _g23) {
    MD2(BiasType, abias, bias, ep.g23, V);
    MD2(float, aweights_scale, weights_scale, ep.g23, V);
    MD2(float, aweights_shift, weights_shift, ep.g23, V);

    __m<V> qs = *(__m<V> *)&md2(aweights_scale, _g23, 0);
    __m<V> b = ep.with_bias ? *(__m<V> *)&md2(abias, _g23, 0) : _mm<V>::setzero_ps();
    __m<V> &qf = *(__m<V> *)&md2(aweights_shift, _g23, 0);

    if (std::is_same<OutputType, float>::value) {
      qf = b - input_z * qf * qs;
    } else {
      qf = b * out_repS + out_z - input_z * qf * qs;
    }

  }, ep.g23);
}

Template_elx_int8_conv_direct_depthwise_t
void Instance_elx_int8_conv_direct_depthwise_t::conv_c160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, float *src_scale, float *weights_scale,
    float *weights_shift, int _ht, int _wt)
{
  auto ker_conv = _wt == ep.wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  int kws = _wt == 0 ? ep.lp : 0;
  int kwe = _wt == ep.wt - 1 ? ep.kw - ep.rp : ep.kw;

  auto _ih = _ht * ep.hs + (ep.kh / 2) - ep.tp;
  auto _iw = _wt * ep.T * ep.ws + (ep.kw / 2) - ep.lp;
  int pad_l = (_wt == 0) && (ep.lp > 0);
  int pad_r = (_wt == ep.wt - 1) && (ep.rp > 0);

  MD2(float, asrc_scale, src_scale, 2, ep.T);
  MD3(InputType, ainput_blocked, input_u8, ep.ih, ep.iw, V);

  auto ainput = &md3(ainput_blocked, _ih, _iw, 0);
  ker_conv(ep, toutput, output, ainput, weights_s8, bias,
           &md2(asrc_scale, 0, 0), &md2(asrc_scale, 1, 0), weights_scale,
           weights_shift, khs, khe, kws, kwe, pad_l, pad_r, attr_);
}

template class elx_int8_conv_direct_depthwise_t<conv::U8F32U8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;
template class elx_int8_conv_direct_depthwise_t<conv::U8F32S8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;

} // namespace euler
