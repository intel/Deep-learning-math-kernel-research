#include <cfenv>
#include <cmath>
#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct_depthwise_lp.hpp"

namespace euler {

static constexpr float INT8GEMM_TWT_QTSCALE = 127.0;

// depth-wise
Template_elx_conv_direct_depthwise_lp_t
Instance_elx_conv_direct_depthwise_lp_t::elx_conv_direct_depthwise_lp_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = this->execution_mode;
  mthr_ = el_get_max_threads();

  this->grp = this->g;
  this->Vx = 1;
  this->V1 = V / this->Vx;
  this->ocg = this->oc / this->g;
  this->icg = this->ic / this->g;

  bool shape_ok = this->icg == 1 && this->ocg == 1 &&
                  estl::any_of(this->kh, 3) &&
                  estl::any_of(this->kw, 3) &&
                  estl::any_of(this->ws, 1, 2) &&
                  estl::any_of(this->hs, 1, 2) &&
                  estl::any_of(this->lp, 0, 1) &&
                  estl::any_of(this->rp, 0, 1) &&
                  estl::any_of(this->tp, 0, 1) &&
                  estl::any_of(this->bp, 0, 1);
  if (!shape_ok) {
    el_error("direct_depthwise_lp: shape not supported");
  }

  // compute multiple groups in one FMA
  // vector multi-group number
  this->G = ALIGNUP(this->g, V);
  this->vmg = V / this->ocg;
  this->g2 = this->G / this->g3 / V;
  this->G2 = this->g2 * this->g3;
  if (V * this->g2 * this->g3 != this->G) {
    el_error("conv: g blocking error for depthwise conv");
  }
  if (this->O != 1) {
    this->O = 1;
    el_warn("conv: group: O!=1 found for vector multi-group");
  }
  this->ic /= this->g;
  this->oc /= this->g;

  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->T == 0) this->T = 1;
  this->I2 = 1;

  this->oc4 = 1;
  this->oc3 = 1;
  this->O = 1;
  this->O1 = 1;
  this->O2 = this->O * this->O1;
  
  this->ic4 = 1;
  this->ic3 = 1;
  this->I2 = 1;

  this->ic2 = this->G / V;
  this->oc2 = this->G / V;

  xopt_ = 0xa160;

  // t3, t2, (T, Tr)
  this->t3 = this->n;
  this->ht = this->oh;
  this->wt = (this->ow + this->T - 1) / this->T;
  this->Tr = this->ow % this->T ? this->ow % this->T : this->T;
  this->nt = this->oh * this->ow;
  this->t2 = this->nt / this->T;
  this->t  = this->nt * this->n;

  if (this->T <= this->lp || this->Tr <= this->rp) {
    el_error("direct_depthwise: (T,Tr) must greater than (lp,rp)");
  }
  bool format_ok = estl::any_of(this->weights_fmt, ghwio, goihw) &&
                   estl::any_of(this->input_fmt, nchw, nChw16c) &&
                   estl::any_of(this->output_fmt, nchw, nChw16c);
  if (!format_ok) {
    el_error("direct_depthwise: format not supported");
  }

  // No Ir/Or
  this->Ir = 0;
  this->Or = 0;
  this->ormask = 0;

  // IC = V = G * C; ic_orig = g * G * C
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC) {
    el_error("IC blocking error");
  }
  // OC = V = G * C; oc_orig = g * G * C
  if (this->oc4 * this->oc3 * this->O2 * V != this->OC) {
    el_error("OC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
  attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;
  attr_ = this->with_relu ? set_attr(attr_, relu_idx) : attr_;

  prepare_quant_calibration(dc);
  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d, g2=%d, g3=%d, g=%d, G=%d\n",
      V, this->Ir, this->I2, this->ic3, this->ic4, this->IC, this->g2, this->g3, this->g, this->G);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}

Template_elx_conv_direct_depthwise_lp_t void
Instance_elx_conv_direct_depthwise_lp_t::prepare_quant_calibration(
    eld_conv_t &dc) {
  this->input_quant_S = dc.input_quant.scale;
  this->input_quant_repS = 1.0f / dc.input_quant.scale;
  this->input_quant_z = dc.input_quant.z;
  this->output_quant_S = dc.output_quant.scale;
  this->output_quant_repS = 1.0f / dc.output_quant.scale;
  this->output_quant_z = dc.output_quant.z;

  if (this->sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_conv_direct_depthwise_lp_t
int Instance_elx_conv_direct_depthwise_lp_t::prepare_execute_opt()
{
  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;

  switch (xopt_) {
  case 0xa160:
    tweights_size_ = this->G * this->kh * this->KW * sizeof(TweightsType);
    input_scale_size_ = 2 * this->T * sizeof(TscaleType);
    weights_scale_size_ = this->G * sizeof(TscaleType);
    weights_factor_size_ = this->G * sizeof(TscaleType);
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
    weights_scale_size_ + weights_factor_size_;
  scratch_size_ = 0;

  return 0;
}

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    weights_scale_ = (TscaleType *)base;
    weights_factor_ = (TscaleType *)((char *)weights_scale_ + weights_scale_size_);
    input_scale_ = (TscaleType *)((char *)weights_factor_ + weights_factor_size_);
    tweights_s8_ = (int8_t *)((char *)input_scale_ + input_scale_size_);
  }
}

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::set_scratch_buffers(void *base)
{
}

Template_elx_conv_direct_depthwise_lp_t
Instance_elx_conv_direct_depthwise_lp_t::~elx_conv_direct_depthwise_lp_t()
{
}

// weights: G2, V, kh, kw
// tweights: G2, kh, V, KW // kw-padded goih16g4w
Template_elx_conv_direct_depthwise_lp_t void
Instance_elx_conv_direct_depthwise_lp_t::trans_weights_3x3(
    TscaleType *weights_scale, TscaleType *weights_factor, int8_t *tweights_s8,
    WeightsType *weights, BiasType *bias)
{
  // absmax
  parallel_for<2>(mthr_, [&](int _G2, int _V) {
    MD4(WeightsType, aweights, weights, this->G2, V, this->kh, this->kw);
    MD2(TscaleType, aweights_scale, weights_scale, this->G2, V);
    float absmax = 0.0;
    iter_each (_kh, this->kh) {
      iter_each (_kw, this->kw) {
        float val = md4(aweights, _G2, _V, _kh, _kw);
        absmax = estl::max(std::abs(val), absmax);
      }
    }
    md2(aweights_scale, _G2, _V) = absmax;
  }, this->G2, V);

  // quantization
  std::fesetround(FE_TONEAREST);
  parallel_for<4>(mthr_, [&](int _G2, int _kh, int _V, int _kw) {
    MD4(int8_t, atweights_s8, tweights_s8, this->G2, this->kh, V, this->KW);
    MD4(WeightsType, aweights, weights, this->G2, V, this->kh, this->kw);
    MD2(TscaleType, aweights_scale, weights_scale, this->G2, V);
    if (_kw < this->kw) {
      // scale
      auto t0 = md4(aweights, _G2, _V, _kh, _kw);
      auto absmax = md2(aweights_scale, _G2, _V);
      t0 = t0 * INT8GEMM_TWT_QTSCALE / absmax;
      // round & store
      md4(atweights_s8, _G2, _kh, _V, _kw) = (int8_t)std::rint(t0);
    } else {
      md4(atweights_s8, _G2, _kh, _V, _kw) = 0;
    }
  }, this->G2, this->kh, V, this->KW);

  // weights-scale
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);
  parallel_for<1>(mthr_, [&](int _G2) {
    MD2(TscaleType, aweights_scale, weights_scale, this->G2, V);
    auto t0 = *(__m<V> *)&md2(aweights_scale, _G2, 0);
    *(__m<V> *)&md2(aweights_scale, _G2, 0) = t0 / mmscale;
  }, this->G2);

  // weights-acc
  parallel_for<2>(mthr_, [&](int _G2, int _V) {
    MD4(int8_t, atweights_s8, tweights_s8, this->G2, this->kh, V, this->KW);
    MD2(TscaleType, aweights_factor, weights_factor, this->G2, V);
    int acc = 0;
    iter_each(_kh, this->kh) {
      iter_each(_kw, this->kw) {
        acc += md4(atweights_s8, _G2, _kh, _V, _kw);
      }
    }
    md2(aweights_factor, _G2, _V) = acc;
  }, this->G2, V);

  // combine with output restore
  auto out_repS = _mm<V>::set1_ps(this->output_quant_repS);
  auto out_z = _mm<V>::set1_ps(this->output_quant_z);
  auto input_S = _mm<V>::set1_ps(this->input_quant_S);
  auto input_z = _mm<V>::set1_ps(this->input_quant_z);

  parallel_for<1>(mthr_, [&](int _G2) {
    MD2(TscaleType, aweights_scale, weights_scale, this->G2, V);
    __m<V> &qs = *(__m<V> *)&md2(aweights_scale, _G2, 0);
    if (std::is_same<OutputType, float>::value) {
      qs = input_S * qs;
    } else {
      qs = input_S * qs * out_repS;
    }
  }, this->G2);

  parallel_for<1>(mthr_, [&](int _G2) {
    MD2(BiasType, abias, bias, this->G2, V);
    MD2(TscaleType, aweights_scale, weights_scale, this->G2, V);
    MD2(TscaleType, aweights_factor, weights_factor, this->G2, V);

    __m<V> qs = *(__m<V> *)&md2(aweights_scale, _G2, 0);
    __m<V> b = this->with_bias ? *(__m<V> *)&md2(abias, _G2, 0) : _mm<V>::setzero_ps();
    __m<V> &qf = *(__m<V> *)&md2(aweights_factor, _G2, 0);

    if (std::is_same<OutputType, float>::value) {
      qf = b - input_z * qf * qs;
    } else {
      qf = b * out_repS + out_z - input_z * qf * qs;
    }

  }, this->G2);
}

Template_elx_conv_direct_depthwise_lp_t
void Instance_elx_conv_direct_depthwise_lp_t::conv_a160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, TscaleType *src_scale, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ht, int _wt)
{
  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->rp : this->kw;

  auto _ih = _ht * this->hs + (this->kh / 2) - this->tp;
  auto _iw = _wt * this->T * this->ws + (this->kw / 2) - this->lp;
  int pad_l = (_wt == 0) && (this->lp > 0);
  int pad_r = (_wt == this->wt - 1) && (this->rp > 0);

  MD2(TscaleType, asrc_scale, src_scale, 2, T);
  MD3(InputType, ainput_blocked, input_u8, this->ih, this->iw, V);

  auto ainput = &md3(ainput_blocked, _ih, _iw, 0);
  ker_conv(*this, toutput, output, ainput, weights_s8, bias,
           &md2(asrc_scale, 0, 0), &md2(asrc_scale, 1, 0), weights_scale,
           weights_factor, khs, khe, kws, kwe, pad_l, pad_r, attr_);
}

} // namespace euler
