#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct_lp.hpp"

namespace euler {

static constexpr float INT8GEMM_TWT_QTSCALE = 127.0;

Template_elx_conv_direct_lp_t
Instance_elx_conv_direct_lp_t::elx_conv_direct_lp_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  xopt_ = this->execution_mode;
  mthr_ = el_get_max_threads();

  this->Vx = 4;
  this->V1 = V / this->Vx;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->T > this->ow) this->T = this->ow;
  if (this->I2 == 0) this->I2 = 1;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  if (this->oc4 == 0) this->oc4 = 1;
  if (this->ic4 == 0) this->ic4 = 1;

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;
  attr_ = 0x0;

  if(this->sampling_kind != CALIBRATED) {
      el_error("int8 direct: non-calibrated sampling_kind not supported");
  }

  // t3, t2, (T, Tr)
  if (xopt_ == 0xa160 || xopt_ == 0xd160 /*|| xopt_ == 0xb160*/) {
    this->t3 = this->n;
    this->ht = this->oh;
    this->wt = (this->ow + this->T - 1)/ this->T;
    this->Tr = this->ow % this->T ? this->ow % this->T : this->T;
    this->nt = this->oh * this->ow;
    this->t2 = this->nt / this->T;
    this->t = this->nt * this->n;

    if (this->T <= this->lp || this->Tr <= this->rp) {
      el_error("Unimplemented T: (T,Tr) must greater than (lp,rp)");
    }

    bool format_ok = (this->weights_fmt == OIhw16i16o) &&
                     estl::any_of(this->input_fmt, nChw16c, nhwc) &&
                     estl::any_of(this->output_fmt, nChw16c, nhwc);
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xa160 /*|| xopt_ == 0xb160*/) {
      bool shape_ok = estl::any_of(this->kh, 3, 5, 7)
          && estl::any_of(this->kw, 3, 5, 7)
          && (this->ws == 1 || this->ws == 2)
          && estl::any_of(this->lp, 0, this->kw / 2)
          && estl::any_of(this->tp, 0, this->kh / 2);
      if (!shape_ok) {
        el_error("direct: a160: shape not supported");
      }
    }
  }

  int V1r = ALIGNUP(this->ic % V, 4) / this->V1;
  this->Ir = V1r % this->V1 ? V1r % this->V1 : this->V1;
  this->Or = this->oc % V ? this->oc % V : V;

  compact_ir_weights_ = false;
  if (this->ic == 3 && this->Ir == 1 && xopt_ == 0xa160 &&
      this->input_fmt == nhwc) { // FBD/FBF kernel
    compact_ir_weights_ = true;
  }

  // oc4, (oc3, oc3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  // TODO
#if 0
  if ((this->output_fmt != nChw16c || this->weights_fmt != OIhw16i16o) &&
      (this->Or != V || this->O2r != this->O2 || this->oc3r != this->oc3)) {
    el_error("direct: int8: no Or support for plain format");
  }
#endif

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC) {
    el_error("IC blocking error");
  }

  attr_ = set_attr(attr_, fma_opt_idx);
  is_first_run_ = true;
  inference_acc_ = this->prop_kind == forward_inference;
  if (xopt_ == 0xa160) {
    attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
    // attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;
  }

  prepare_weights_acc();
  prepare_quant_calibration(dc);
  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, Vx=%d, I2=%d, ic3=%d, ic4=%d, IC=%d\n",
      V, this->Ir, this->Vx, this->I2, this->ic3, this->ic4, this->IC);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC);
}

Template_elx_conv_direct_lp_t
int Instance_elx_conv_direct_lp_t::prepare_execute_opt()
{
  if (this->with_ip_sum && this->with_relu && this->output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t tweights_s8_size = 0, input_scale_size = 0, weights_scale_size = 0,
      weights_factor_size;

  toutput_size_ = 0;
  tweights_s8_size_ = 0;
  input_scale_size_ = 0;
  weights_scale_size_ = 0;
  weights_factor_size_ = 0;
  toutput_ = nullptr;
  tweights_s8_ = nullptr;
  input_scale_ = nullptr;
  weights_scale_ = nullptr;
  weights_factor_ = nullptr;

  switch (xopt_) {
  /*case 0xb160:*/
  case 0xa160:
  case 0xd160:
    toutput_size = this->n * this->OC * this->oh * this->ow * sizeof(ToutputType);
    tweights_s8_size = this->kh * this->kw * this->IC * this->OC * sizeof(int8_t);
    input_scale_size = 2 * this->T * sizeof(TscaleType);
    weights_scale_size = this->OC * sizeof(TscaleType);
    weights_factor_size = wacc_h_ * wacc_wt_ * wacc_wT_ * this->OC * sizeof(TscaleType);
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

  const size_t align = PAGE_SIZE;
  toutput_size_ = toutput_size > 0 ? alignup(toutput_size, align) : 0;
  tweights_s8_size_ = tweights_s8_size > 0 ? alignup(tweights_s8_size, align) : 0;
  input_scale_size_ = input_scale_size > 0 ? alignup(input_scale_size, align) : 0;
  weights_scale_size_ = weights_scale_size > 0 ? alignup(weights_scale_size, align) : 0;
  weights_factor_size_ = weights_factor_size > 0 ? alignup(weights_factor_size, align) : 0;

  workspace_size_ = tweights_s8_size_ + weights_scale_size_
      + weights_factor_size_ + input_scale_size_;
  scratch_size_ = toutput_size_;

  return 0;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    weights_scale_ = (TscaleType *)base;
    weights_factor_ = (TscaleType *)((char *)weights_scale_ + weights_scale_size_);
    input_scale_ = (TscaleType *)((char *)weights_factor_ + weights_factor_size_);
    tweights_s8_ = (int8_t *)((char *)input_scale_ + input_scale_size_);
  }
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::set_scratch_buffers(void *base)
{
  if (base != nullptr)
    toutput_ = (ToutputType *)base;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::prepare_quant_calibration(eld_conv_t &dc)
{
  this->input_quant_S = dc.input_quant.scale;
  this->input_quant_repS = 1 / dc.input_quant.scale;
  this->input_quant_z = dc.input_quant.z;
  this->output_quant_S = dc.output_quant.scale;
  this->output_quant_repS = 1 / dc.output_quant.scale;
  this->output_quant_z = dc.output_quant.z;

  if (this->sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_conv_direct_lp_t
Instance_elx_conv_direct_lp_t::~elx_conv_direct_lp_t()
{
}

Template_elx_conv_direct_lp_t void
Instance_elx_conv_direct_lp_t::prepare_weights_acc() {
  if (xopt_ == 0xa160) {
    wacc_wT_ = T;
  } else {
    wacc_wT_ = 1;
    if (this->input_quant_z != 0 &&
        (this->lp != 0 || this->rp != 0 || this->tp != 0 || this->bp != 0)) {
      el_error("direct: int8: input_z != 0 does not support in provided xopt");
    }
  }
  if (this->input_quant_z == 0) {
    wacc_h_ranges_.push_back(std::make_tuple(0, this->kh - 1));
    wacc_w_ranges_.push_back(std::make_tuple(0, this->kw - 1));
    wacc_wt_ = 1;
    wacc_h_ = 1;
    wacc_w_ = 1;
    _wacc_hf_ = 0;
    _wacc_hfr_ = 0;
    _wacc_wf_ = 0;
    _wacc_wfr_ = 0;
    _wacc_ohfs_ = 0;
    _wacc_ohfe_ = this->oh - 1;
  } else {
    if (this->lp + this->ow < this->kw || this->rp + this->ow < this->kw ||
        this->tp + this->oh < this->kh || this->bp + this->oh < this->kh) {
      el_error("Shape not support for U8 input with zero-point");
    }

    for (int khs = this->tp; khs > 0; khs -= this->hs) {
      wacc_h_ranges_.push_back(std::make_tuple(khs, this->kh - 1));
    }
    _wacc_hf_ = wacc_h_ranges_.size();
    wacc_h_ranges_.push_back(std::make_tuple(0, this->kh - 1));

    std::vector<std::tuple<int, int>> kh_ranges_tmp;
    for (int khe = this->bp; khe > 0; khe -= this->hs) {
      kh_ranges_tmp.push_back(std::make_tuple(0, this->kh - khe - 1));
    }
    _wacc_hfr_ = kh_ranges_tmp.size();
    for (auto t = kh_ranges_tmp.rbegin(); t != kh_ranges_tmp.rend(); ++t) {
      wacc_h_ranges_.push_back(*t);
    }
    for (auto i : wacc_h_ranges_) {
      printf("kh_ranges: %d, %d\n", std::get<0>(i), std::get<1>(i));
    }

    // kw-ranges
    for (int kws = this->lp; kws > 0; kws -= this->ws) {
      wacc_w_ranges_.push_back(std::make_tuple(kws, this->kw - 1));
    }
    _wacc_wf_ = wacc_w_ranges_.size();
    wacc_w_ranges_.push_back(std::make_tuple(0, this->kw - 1));

    std::vector<std::tuple<int, int>> kw_ranges_tmp;
    for (int kwe = this->rp; kwe > 0; kwe -= this->ws) {
      kw_ranges_tmp.push_back(std::make_tuple(0, this->kw - kwe - 1));
    }
    _wacc_wfr_ = kw_ranges_tmp.size();
    for (auto t = kw_ranges_tmp.rbegin(); t != kw_ranges_tmp.rend(); ++t) {
      wacc_w_ranges_.push_back(*t);
    }
    for (auto i : wacc_w_ranges_) {
      printf("kw_ranges: %d, %d\n", std::get<0>(i), std::get<1>(i));
    }

    wacc_wt_ = this->wt >= 3 ? 3 : this->wt; // left T, middle T (full), right Tr
    wacc_h_ = wacc_h_ranges_.size();
    wacc_w_ = wacc_w_ranges_.size();
    _wacc_ohfs_ = _wacc_hf_;
    _wacc_ohfe_ = this->oh - _wacc_hfr_ - 1;
    _wacc_owfs_ = _wacc_wf_;
    _wacc_owfe_ = this->ow - _wacc_wfr_ - 1;
  }

  // Debug
  printf(
      "wacc_h_:%d, wacc_w_:%d, _wacc_hf_:%d, _wacc_wf_:%d, _wacc_hfr_:%d, "
      "_wacc_wfr_:%d, wacc_wt_:%d, wacc_wT_:%d, _wacc_ohfs_:%d, "
      "_wacc_ohfe_:%d, _wacc_owfs_:%d, _wacc_owfe_:%d\n",
      wacc_h_, wacc_w_, _wacc_hf_, _wacc_wf_, _wacc_hfr_, _wacc_wfr_, wacc_wt_,
      wacc_wT_, _wacc_ohfs_, _wacc_ohfe_, _wacc_owfs_, _wacc_owfe_);
}

Template_elx_conv_direct_lp_t void
Instance_elx_conv_direct_lp_t::__trans_weights_acc(TscaleType *weights_scale,
                                                   TscaleType *weights_factor,
                                                   int8_t *tweights_s8,
                                                   BiasType *bias) {
  auto V1 = compact_ir_weights_ ? this->Ir : this->V1;
  TscaleType *weights_factor_buf =
      (TscaleType *)malloc(wacc_h_ * wacc_w_ * this->OC * sizeof(TscaleType));

  // weights-acc
  parallel_for<7>(mthr_, [&](int _wacc_h, int _wacc_w, int _oc4, int _oc3, int _O1, int _O, int _oV) {
    MD12(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4, this->oc3,
         this->ic3, this->kh, this->kw, this->O1, this->I2, V1, this->O,
         V, this->Vx);
    MD7(TscaleType, atweights_factor_buf, weights_factor_buf, wacc_h_, wacc_w_, this->oc4,
        this->oc3, this->O1, this->O, V);

    auto khs = std::get<0>(wacc_h_ranges_[_wacc_h]);
    auto khe = std::get<1>(wacc_h_ranges_[_wacc_h]);
    auto kws = std::get<0>(wacc_w_ranges_[_wacc_w]);
    auto kwe = std::get<1>(wacc_w_ranges_[_wacc_w]);
    int acc  = 0;
    iter_each(_ic4, this->ic4) {
      iter_each(_ic3, this->ic3) {
        for (int _kh = khs; _kh <= khe; ++_kh) {
          // iter_each (_kh, this->kh) {
          for (int _kw = kws; _kw <= kwe; ++_kw) {
            // iter_each (_kw, this->kw) {
            iter_each(_I2, this->I2) {
              bool last_IV = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 &&
                             _I2 == this->I2 - 1;
              auto V1r = last_IV ? Ir : this->V1;
              iter_each(_V1, V1r) {
                iter_each(_Vx, this->Vx) {
                  acc += md12(atweights_s8, _oc4, _ic4, _oc3, _ic3, _kh, _kw,
                              _O1, _I2, _V1, _O, _oV, _Vx);
                }
              }
            }
          }
        }
      }
    }
    md7(atweights_factor_buf, _wacc_h, _wacc_w, _oc4, _oc3, _O1, _O, _oV) = acc;
  }, wacc_h_, wacc_w_, this->oc4, this->oc3, this->O1, this->O, V);

  auto out_repS = _mm<V>::set1_ps(this->output_quant_repS);
  auto out_z = _mm<V>::set1_ps(this->output_quant_z);
  auto input_S = _mm<V>::set1_ps(this->input_quant_S);
  auto input_z = _mm<V>::set1_ps(this->input_quant_z);

  // Combine output restore and requantization scale and factor
  parallel_for<1>(mthr_, [&](int _oc2) {
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);
    MD2(BiasType, abias, bias, this->oc2, V);
    __m<V> &qs = *(__m<V> *)&md2(atweights_scale, _oc2, 0);

    if (std::is_same<OutputType, float>::value) {
      qs = input_S * qs;
    } else {
      qs = input_S * qs * out_repS;
    }
  }, this->oc2);

  parallel_for<1>(mthr_, [&](int _oc2) {
    MD2(BiasType, abias, bias, this->oc2, V);
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);
    MD5(TscaleType, atweights_factor, weights_factor, wacc_h_, wacc_wt_, this->oc2, wacc_wT_, V);
    MD4(TscaleType, atweights_factor_buf, weights_factor_buf, wacc_h_, wacc_w_, this->oc2, V);

    __m<V> qs = *(__m<V> *)&md2(atweights_scale, _oc2, 0);
    __m<V> b = this->with_bias ? *(__m<V> *)&md2(abias, _oc2, 0) : _mm<V>::setzero_ps();

    iter_each(_wacc_h, wacc_h_) {
      iter_each(_wt, wacc_wt_) {
        iter_each(_T, wacc_wT_) {
          __m<V> &qf =
              *(__m<V> *)&md5(atweights_factor, _wacc_h, _wt, _oc2, _T, 0);
          int _wacc_w = _wacc_wf_;
          if (wacc_wt_ == 1) {
            if (this->input_quant_z == 0) {
              _wacc_w = _wacc_wf_;
            } else {
              _wacc_w = _T < _wacc_owfs_ ? _T : _T > _wacc_owfe_
                                  ? _T - (this->T - _wacc_wfr_) + _wacc_wf_ + 1
                                  : _wacc_wf_;
            }
          } else { // wacc_wt_ == 2, 3
            if (_wt == 0) { // first
              _wacc_w = _T < _wacc_wf_ ? _T : _wacc_wf_;
            } else if (_wt == wacc_wt_ - 1) { // last
              _wacc_w = _T >= this->Tr ? _wacc_wf_ : _T < (this->Tr - _wacc_wfr_)
                         ? _wacc_wf_
                         : _T - (this->Tr - _wacc_wfr_) + _wacc_wf_ + 1;
            } else { // middle, if wacc_wt_ == 3
              _wacc_w = _wacc_wf_;
            }
          }
          __m<V> qf_tmp =
              *(__m<V> *)&md4(atweights_factor_buf, _wacc_h, _wacc_w, _oc2, 0);
          if (std::is_same<OutputType, float>::value) {
            qf = b - input_z * qf_tmp * qs;
          } else {
            qf = b * out_repS + out_z - input_z * qf_tmp * qs;
          }
        }
      }
    }
  }, this->oc2);

  if (weights_factor_buf)
    free(weights_factor_buf);
}

// weights (blocked): oc2, ic2, kh, kw, V, V
// tweights: oc4, ic4, oc3, _ic3, kh, kw, O1, I2, V1, O, V, Vx
Template_elx_conv_direct_lp_t void Instance_elx_conv_direct_lp_t::
trans_weights(TscaleType *weights_scale, TscaleType *weights_factor,
                int8_t *tweights_s8, WeightsType *weights, BiasType *bias) {
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);

  auto Vr = this->ic % V ? this->ic % V : V;
  auto V1 = compact_ir_weights_ ? this->Ir : this->V1;

  // abs-max
  parallel_for<1>(mthr_, [&](int _oc2) {
    MD6(WeightsType, aweights, weights, this->oc2, this->ic2, this->kh, this->kw, V, V);
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);

    __m<V> abs_max = _mm<V>::set1_ps(0.0);
    iter_each (_ic2, this->ic2) {
      auto IV = _ic2 == this->ic2 - 1 ? Vr : V;
      iter_each (_kh, this->kh) {
      iter_each (_kw, this->kw) {
      iter_each (_iV, IV) {
        abs_max = _mm<V>::max_ps(abs_max, _mm512_abs_ps(
            *(__m<V> *)&md6(aweights, _oc2, _ic2, _kh, _kw, _iV, 0)));
      }}}
    }
    _mm512_store_ps(&md2(atweights_scale, _oc2, 0), abs_max);
  }, this->oc2);

  // quantization
  parallel_for<11>(mthr_, [&](int _oc4, int _oc3, int _O1,
      int _O, int _ic4, int _ic3, int _I2, int _kh, int _kw, int _V1, int _Vx) {
    MD12(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4, this->oc3,
         this->ic3, this->kh, this->kw, this->O1, this->I2, V1, this->O, V, this->Vx);
    MD12(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O,
         this->ic4, this->ic3, this->I2, this->kh, this->kw, this->V1, this->Vx, V);
    MD5(TscaleType, atweights_scale, weights_scale, this->oc4, this->oc3,
        this->O1, this->O, V);
    bool last_IV = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1 && _I2 == this->I2 - 1;

    if (!last_IV || _V1 * this->Vx + _Vx < Vr) {
      __m<V> t0;
      // multi scal
      t0 = _mm<V>::mul_ps(*(__m<V> *)&md12(aweights, _oc4, _oc3, _O1, _O,
                              _ic4, _ic3, _I2, _kh, _kw, _V1, _Vx, 0), mmscale);
      t0 = _mm<V>::div_ps(t0, *(__m<V> *)&md5(atweights_scale, _oc4, _oc3, _O1, _O, 0));

      // rounding
      t0 = _mm<V>::roundscale_ps(t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      // int8_t
      TweightsType *rounded = (TweightsType *)&t0;
      #pragma omp simd
      iter_each (_oV, V) {
        md12(atweights_s8,
            _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _V1, _O, _oV, _Vx) =
            (int8_t)rounded[_oV];
      }
    } else {
      #pragma omp simd
      iter_each (_oV, V) {
        md12(atweights_s8,
            _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _V1, _O, _oV, _Vx) = 0;
      }
    }
  }, this->oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2,
     this->kh, this->kw, V1, this->Vx);

  // weights-scale
  parallel_for<1>(mthr_, [&](int _oc2) {
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);
    auto t0 = _mm<V>::div_ps(
        *(__m<V> *)&md2(atweights_scale, _oc2, 0), mmscale);
    _mm<V>::store_ps(&md2(atweights_scale, _oc2, 0), t0);
  }, this->oc2);

  __trans_weights_acc(weights_scale, weights_factor, tweights_s8, bias);
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::conv_a160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, TscaleType *src_scale, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ic4, int _oc4, int _ht, int _wt)
{
  auto V1 = compact_ir_weights_ ? this->Ir : this->V1;

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->rp : this->kw;

  auto _ih = _ht * this->hs + (this->kh / 2) - this->tp;
  auto _iw = _wt * this->T * this->ws + (this->kw / 2) - this->lp;
  int pad_l = _wt == 0 ? this->lp : 0;
  int pad_r = _wt == this->wt - 1 ? this->rp : 0;

  int _wacc_wt =
      (_wt == 0 || wacc_wt_ == 1) ? 0 : _wt == this->wt - 1 ? wacc_wt_ - 1 : 1;
  int _wacc_h =
      _ht < _wacc_ohfs_
          ? _ht
          : _ht <= _wacc_ohfe_ ? _wacc_hf_ : _wacc_ohfs_ + _ht - _wacc_ohfe_;

  MD3(int8_t, aweights, weights_s8, this->oc3, this->ic3, this->kh * this->kw *
      this->O2 * this->I2 * V1 * V * this->Vx);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);
  MD2(TscaleType, aweights_scale, weights_scale, this->oc3, this->O2 * V);
  MD5(TscaleType, aweights_factor, weights_factor, wacc_h_, wacc_wt_, this->oc4, this->oc3, this->O2 * T * V);
  MD2(TscaleType, asrc_scale, src_scale, 2, T);
  // nhwc
  MD3(InputType, ainput0_nhwc, input_u8, this->ih, this->iw, this->ic);
  MD3(InputType, ainput1_nhwc, &md3(ainput0_nhwc, _ih, _iw, 0), this->ic4, this->ic3, this->I2 * V);
  MD2(OutputType, aoutput_nhwc, output, this->oc3, this->O2 * V);
  MD2(ToutputType, atoutput_nhwc, toutput, this->oc3, this->O2 * V);
  // blocked
  MD5(InputType, ainput_blocked, input_u8, this->ic3, this->I2, this->ih, this->iw, V);
  MD2(OutputType, aoutput_blocked, output, this->oc3, this->O2 * this->ht * this->ow * V);
  MD2(ToutputType, atoutput_blocked, toutput, this->oc3, this->O2 * this->ht * this->ow * V);

  iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr =
          (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;

      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        attr = set_attr(attr, c_output_idx);
        if (this->Ir != this->V1) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      auto ainput = this->input_fmt == nhwc
                          ? &md3(ainput1_nhwc, 0, _ic3, 0)
                          : &md5(ainput_blocked, _ic3, 0, _ih, _iw, 0);
      auto aoutput = this->output_fmt == nhwc
                          ? &md2(aoutput_nhwc, _oc3, 0)
                          : &md2(aoutput_blocked, _oc3, 0);
      auto atoutput = this->output_fmt == nhwc
                          ? &md2(atoutput_nhwc, _oc3, 0)
                          : &md2(atoutput_blocked, _oc3, 0);
      ker_conv(*this, atoutput, aoutput, ainput, &md3(aweights, _oc3, _ic3, 0),
               &md2(abias, _oc3, 0), &md2(asrc_scale, 0, 0),
               &md2(asrc_scale, 1, 0), &md2(aweights_scale, _oc3, 0),
               &md5(aweights_factor, _wacc_h, _wacc_wt, 0, _oc3, 0),
               khs, khe, kws, kwe, pad_l, pad_r, attr);
    }
  }
}

// slow path
Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::gemm_d160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, TscaleType *src_scale, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ic4, int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V1, Vx
  // output:  oc3*, O2, ht*, wt*, T, V
  MD5(int8_t, aweights, weights_s8, this->oc3, this->ic3, this->kh, this->kw,
      this->O2 * this->I2 * this->V1 * V * this->Vx);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);
  MD3(TscaleType, aweights_scale, weights_scale, this->oc3, this->O2, V);
  MD3(TscaleType, aweights_factor, weights_factor, this->oc3, this->O2, V);
  MD2(TscaleType, asrc_scale, src_scale, 2, T);

  int Tz = _wt == this->wt - 1 ? this->Tr : this->T;
  int ows0 = _wt * this->T;
  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  assert(this->T > this->lp);
  assert(this->Tr > this->rp);

  // nhwc
  MD3(InputType, ainput0_nhwc, input_u8, this->ih, this->iw, this->ic);
  MD3(OutputType, aoutput0_nhwc, output, this->ht, this->ow, this->oc);
  MD3(ToutputType, atoutput0_nhwc, toutput, this->ht, this->ow, this->OC);
  // blocked
  MD5(InputType, ainput_blocked, input_u8,
      this->ic3, this->I2, this->ih, this->iw, this->V1 * this->Vx);
  MD5(OutputType, aoutput_blocked, output,
      this->oc3, this->O2, this->ht, this->ow, V);
  MD5(ToutputType, atoutput_blocked, toutput,
      this->oc3, this->O2, this->ht, this->ow, V);

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
    if (_ic4 == 0 && _ic3 == 0) {
      __m<V> s = _mm<V>::setzero_ps();
      iter_each (_O2, this->O2) {
      iter_each (_T, Tz) {
        MD4(ToutputType, atoutput1_nhwc,
            &md3(atoutput0_nhwc, _ht, ows0 + _T, 0),
            this->oc4, this->oc3, this->O2, V);
        auto atout = this->output_fmt == nhwc
                   ? &md4(atoutput1_nhwc, 0, _oc3, _O2, 0)
                   : &md5(atoutput_blocked, _oc3, _O2, _ht, ows0 + _T, 0);

        if (I == ISA_SKX_AVX512 && std::is_same<ToutputType, float>::value)
          _mm<V>::store_ps(atout, s);
        else
          el_error("direct: d160: unimplemented");
      }}
    }
    int attr = attr_;
    if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
      if (this->Ir != this->V1) attr = set_attr(attr, has_Ir_idx);
    }

    for (int _kh = khs; _kh < khe; ++_kh) {
      auto _ih = this->hs * _ht + _kh - this->tp;
      for (int _kw = 0; _kw < this->kw; ++_kw) {
        auto _iws = this->ws * ows0 + _kw - this->lp;
        while (_iws < 0) _iws += this->ws;
        auto _ows = (_iws + this->lp - _kw) / this->ws;

        MD4(InputType, ainput1_nhwc, &md3(ainput0_nhwc, _ih, _iws, 0),
            this->ic4, this->ic3, this->I2, V);
        MD4(ToutputType, atoutput1_nhwc, &md3(atoutput0_nhwc, _ht, _ows, 0),
            this->oc4, this->oc3, this->O2, V);
        auto ain = this->input_fmt == nhwc
                 ? &md4(ainput1_nhwc, 0, _ic3, 0, 0)
                 : &md5(ainput_blocked, _ic3, 0, _ih, _iws, 0);
        auto atout = this->output_fmt == nhwc
                 ? &md4(atoutput1_nhwc, 0, _oc3, 0, 0)
                 : &md5(atoutput_blocked, _oc3, 0, _ht, _ows, 0);
        ker_gemm_[_wt][_kw](*this, atout, nullptr, ain,
            &md5(aweights, _oc3, _ic3, _kh, _kw, 0),
            &md3(abias, _oc3, 0, 0), attr,
            nullptr, nullptr, nullptr, nullptr);
      }
    }

    __mmask16 k = _cvtu32_mask16((1 << this->Or) - 1);
    if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
      iter_each (_O2, this->O2) {
      iter_each (_T, Tz) {
        MD4(ToutputType, atoutput1_nhwc, &md3(atoutput0_nhwc, _ht, ows0 + _T, 0),
            this->oc4, this->oc3, this->O2, V);
        MD4(OutputType, aoutput1_nhwc, &md3(aoutput0_nhwc, _ht, ows0 + _T, 0),
            this->oc4, this->oc3, this->O2, V);
        auto atout = this->output_fmt == nhwc
            ? &md4(atoutput1_nhwc, 0, _oc3, _O2, 0)
            : &md5(atoutput_blocked, _oc3, _O2, _ht, ows0 + _T, 0);
        auto aout = this->output_fmt == nhwc
            ? &md4(aoutput1_nhwc, 0, _oc3, _O2, 0)
            : &md5(aoutput_blocked, _oc3, _O2, _ht, ows0 + _T, 0);

        if (I == ISA_SKX_AVX512 && std::is_same<ToutputType, float>::value) {
          __m<V> tout = _mm<V>::cvtepi32_ps(*(__i<V> *)atout);
          // restore and requantization
          auto scale = *(__m<V> *)&md3(aweights_scale, _oc3, _O2, 0);
          auto factor = *(__m<V> *)&md3(aweights_factor, _oc3, _O2, 0);
          tout = tout * scale + factor;
          // fuse relu
          if (this->with_relu) {
            auto lower = *(__m<V> *)(this->relu_bound_lower_vec);
            auto upper = *(__m<V> *)(this->relu_bound_upper_vec);
            tout = _mm<V>::max_ps(tout, lower);
            tout = _mm<V>::min_ps(tout, upper);
          }

          if (std::is_same<OutputType, int8_t>::value ||
              std::is_same<OutputType, uint8_t>::value) {
            // rounding
            __i<V> s_out = _mm<V>::cvt_roundps_epi32(
                tout, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m128i x8_out;
            if (std::is_same<OutputType, int8_t>::value)
              x8_out = _mm<V>::cvtsepi32_epi8(s_out);
            else // uint8
              x8_out = _mm<V>::cvtusepi32_epi8(s_out);
            // store output
            _mm_store_si128((__m128i *)aout, x8_out);
          } else if (std::is_same<OutputType, float>::value) {
            if (this->with_argmax)
              _mm<V>::store_ps(atout, tout);
            else {
              bool last_oc = (_oc4 == this->oc4 - 1
                              && _oc3 == this->oc3 - 1
                              && _O2 == this->O2 - 1);
              if (last_oc) {
                _mm<V>::mask_store_ps(aout, k, tout);
              } else {
                _mm<V>::store_ps(aout, tout);
              }
            }
          } else {
            el_error("direct: d160: unimplemented");
          }
        } else
          el_error("direct: d060: unimplemented");
      }}
    }
  }}
}

} // namespace euler
