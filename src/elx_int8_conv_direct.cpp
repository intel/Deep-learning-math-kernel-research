#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_int8_conv_direct.hpp"
#include "elx_int8_conv_direct_bind.hpp"
#include "elx_int8_conv_direct_xopt.hpp"

namespace euler {

Template_elx_int8_conv_direct_t
Instance_elx_int8_conv_direct_t::elx_int8_conv_direct_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  xopt_ = 0; //ep.execution_mode;
  if (xopt_ == 0) {
    if (estl::any_of(ep.kw, 3, 5, 7))
      xopt_ = 0xc160; // conv kernel
    else
      xopt_ = 0xa160; // gemm kernel
  }

  mthr_ = estl::max_concurrency();

  ep.Vx = 4;
  ep.V1 = V / ep.Vx;
  ep.IC = ALIGNUP(ep.ic, V);
  ep.OC = ALIGNUP(ep.oc, V);

  if (ep.T > ep.ow) ep.T = ep.ow;
  if (ep.I2 == 0) ep.I2 = 1;
  if (ep.T == 0)  ep.T = 1;
  if (ep.O == 0)  ep.O = 1;
  if (ep.O1 == 0) ep.O1 = 1;
  ep.O2 = ep.O * ep.O1;

  if (ep.O4 == 0) ep.O4 = 1;
  if (ep.I4 == 0) ep.I4 = 1;

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;
  attr_ = 0x0;

  if (ep.sampling_kind != CALIBRATED) {
    el_error("int8 direct: non-calibrated sampling_kind not supported");
  }

  // n, t2, (T, Tr)
  if (xopt_ == 0xc160 || xopt_ == 0xa160 /*|| xopt_ == 0xb160*/) {
    ep.ht = ep.oh;
    ep.wt = (ep.ow + ep.T - 1)/ ep.T;
    ep.Tr = ep.ow % ep.T ? ep.ow % ep.T : ep.T;
    ep.nt = ep.oh * ep.ow;
    ep.t2 = ep.nt / ep.T;
    ep.t = ep.nt * ep.n;

    if (ep.T < ep.lp || ep.Tr < ep.rp) {
      el_error("Unimplemented T: (T,Tr) < (lp,rp)");
    }

    bool format_ok = (ep.weights_fmt == OIhw16i16o) &&
                     estl::any_of(ep.input_fmt, nChw16c, nhwc) &&
                     estl::any_of(ep.output_fmt, nChw16c, nhwc);
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xc160 /*|| xopt_ == 0xb160*/) {
      bool shape_ok = estl::any_of(ep.kh, 3, 5, 7)
          && estl::any_of(ep.kw, 3, 5, 7)
          && (ep.ws == 1 || ep.ws == 2)
          && estl::any_of(ep.lp, ep.kw / 2 - 1, ep.kw / 2)
          && estl::any_of(ep.rp, ep.kw / 2 - 1, ep.kw / 2)
          && estl::any_of(ep.tp, ep.kh / 2 - 1, ep.kh / 2)
          && estl::any_of(ep.bp, ep.kh / 2 - 1, ep.kh / 2);
      if (!shape_ok) {
        el_error("direct: c160: shape not supported");
      }
    }
  }

  int V1r = ALIGNUP(ep.ic % V, 4) / ep.Vx;
  ep.Ir = V1r % ep.V1 ? V1r % ep.V1 : ep.V1;
  ep.Or = ep.oc % V ? ep.oc % V : V;

  compact_ir_weights_ = false;
  if (ep.ic == 3 && ep.Ir == 1 && xopt_ == 0xc160 &&
      ep.input_fmt == nhwc) { // FBD/FBF kernel
    compact_ir_weights_ = true;
  }

  // O4, (O3, O3r), (O2, O2r)
  ep.oc34 = (ep.oc2 + ep.O2 - 1) / ep.O2;
  ep.O2r = ep.oc2 % ep.O2;
  if (ep.O2r == 0) ep.O2r = ep.O2;
  ep.O3 = ep.O4; // FIXME, swap order
  ep.O4 = (ep.oc34 + ep.O3 - 1) / ep.O3;
  ep.O3r = ep.oc34 % ep.O3;
  if (ep.O3r == 0) ep.O3r = ep.O3;

  // TODO
#if 0
  if ((ep.output_fmt != nChw16c || ep.weights_fmt != OIhw16i16o) &&
      (ep.Or != V || ep.O2r != ep.O2 || ep.O3r != ep.O3)) {
    el_error("direct: int8: no Or support for plain format");
  }
#endif

  // I4, I3, I3
  ep.ic34 = ep.ic2 / ep.I2;
  ep.I3 = ep.ic34 / ep.I4;
  if (ep.I4 * ep.I3 * ep.I2 * V != ep.IC) {
    el_error("IC blocking error");
  }

  attr_ = set_bit(attr_, AT_FMAOPT_MASK);
  is_first_run_ = true;
  inference_acc_ = ep.prop_kind == forward_inference;
  if (xopt_ == 0xc160) {
    attr_ = ep.with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
    // attr_ = ep.with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;
  }

  prepare_weights_acc();
  prepare_quant_calibration(dc);
  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, Vx=%d, I2=%d, I3=%d, I4=%d, IC=%d",
         V, ep.Ir, ep.Vx, ep.I2, ep.I3, ep.I4, ep.IC);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC);
}

Template_elx_int8_conv_direct_t
int Instance_elx_int8_conv_direct_t::prepare_execute_opt()
{
  if (ep.with_ip_sum && ep.with_relu && ep.output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }
  size_t tweights_size = 0, tinput_size = 0, toutput_size = 0;
  size_t tweights_s8_size = 0, input_scale_size = 0, weights_scale_size = 0,
      weights_shift_size;

  toutput_size_ = 0;
  tweights_s8_size_ = 0;
  input_scale_size_ = 0;
  weights_scale_size_ = 0;
  weights_shift_size_ = 0;
  toutput_ = nullptr;
  tweights_s8_ = nullptr;
  input_scale_ = nullptr;
  weights_scale_ = nullptr;
  weights_shift_ = nullptr;

  switch (xopt_) {
  /*case 0xb160:*/
  case 0xc160:
  case 0xa160:
    toutput_size = ep.n * ep.OC * ep.oh * ep.ow * sizeof(ToutputType);
    tweights_s8_size = ep.kh * ep.kw * ep.IC * ep.OC * sizeof(int8_t);
    input_scale_size = 2 * ep.T * sizeof(float);
    weights_scale_size = ep.OC * sizeof(float);
    weights_shift_size = wacc_h_ * wacc_wt_ * wacc_wT_ * ep.OC * sizeof(float);
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
  weights_shift_size_ = weights_shift_size > 0 ? alignup(weights_shift_size, align) : 0;

  workspace_size_ = tweights_s8_size_ + weights_scale_size_
      + weights_shift_size_ + input_scale_size_;
  scratch_size_ = toutput_size_;

  return 0;
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::set_workspace_buffers(void *base)
{
  if (base != nullptr) {
    weights_scale_ = (float *)base;
    weights_shift_ = (float *)((char *)weights_scale_ + weights_scale_size_);
    input_scale_ = (float *)((char *)weights_shift_ + weights_shift_size_);
    tweights_s8_ = (int8_t *)((char *)input_scale_ + input_scale_size_);
  }
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::set_scratch_buffers(void *base)
{
  if (base != nullptr)
    toutput_ = (ToutputType *)base;
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::prepare_quant_calibration(eld_conv_t &dc)
{
  ep.input_quant_S = dc.input_quant.scale;
  ep.input_quant_repS = 1 / dc.input_quant.scale;
  ep.input_quant_z = dc.input_quant.z;
  ep.output_quant_S = dc.output_quant.scale;
  ep.output_quant_repS = 1 / dc.output_quant.scale;
  ep.output_quant_z = dc.output_quant.z;

  if (ep.sampling_kind != CALIBRATED)
    el_error("Unsupported quantization mode in int8 direct 1x1");
}

Template_elx_int8_conv_direct_t
Instance_elx_int8_conv_direct_t::~elx_int8_conv_direct_t()
{
}

Template_elx_int8_conv_direct_t void
Instance_elx_int8_conv_direct_t::prepare_weights_acc() {
  if (xopt_ == 0xc160) {
    wacc_wT_ = ep.T;
  } else {
    wacc_wT_ = 1;
    if (ep.input_quant_z != 0 &&
        (ep.lp != 0 || ep.rp != 0 || ep.tp != 0 || ep.bp != 0)) {
      el_error("direct: int8: input_z != 0 does not support in provided xopt");
    }
  }
  if (ep.input_quant_z == 0) {
    wacc_h_ranges_.push_back(std::make_tuple(0, ep.kh - 1));
    wacc_w_ranges_.push_back(std::make_tuple(0, ep.kw - 1));
    wacc_wt_ = 1;
    wacc_h_ = 1;
    wacc_w_ = 1;
    _wacc_hf_ = 0;
    _wacc_hfr_ = 0;
    _wacc_wf_ = 0;
    _wacc_wfr_ = 0;
    _wacc_ohfs_ = 0;
    _wacc_ohfe_ = ep.oh - 1;
  } else {
    if (ep.lp + ep.ow < ep.kw || ep.rp + ep.ow < ep.kw ||
        ep.tp + ep.oh < ep.kh || ep.bp + ep.oh < ep.kh) {
      el_error("Shape not support for U8 input with zero-point");
    }

    for (int khs = ep.tp; khs > 0; khs -= ep.hs) {
      wacc_h_ranges_.push_back(std::make_tuple(khs, ep.kh - 1));
    }
    _wacc_hf_ = wacc_h_ranges_.size();
    wacc_h_ranges_.push_back(std::make_tuple(0, ep.kh - 1));

    std::vector<std::tuple<int, int>> kh_ranges_tmp;
    for (int khe = ep.bp; khe > 0; khe -= ep.hs) {
      kh_ranges_tmp.push_back(std::make_tuple(0, ep.kh - khe - 1));
    }
    _wacc_hfr_ = kh_ranges_tmp.size();
    for (auto t = kh_ranges_tmp.rbegin(); t != kh_ranges_tmp.rend(); ++t) {
      wacc_h_ranges_.push_back(*t);
    }
    for (auto i : wacc_h_ranges_) {
      el_log(__DEBUG, "kh_ranges: %d, %d", std::get<0>(i), std::get<1>(i));
    }

    // kw-ranges
    for (int kws = ep.lp; kws > 0; kws -= ep.ws) {
      wacc_w_ranges_.push_back(std::make_tuple(kws, ep.kw - 1));
    }
    _wacc_wf_ = wacc_w_ranges_.size();
    wacc_w_ranges_.push_back(std::make_tuple(0, ep.kw - 1));

    std::vector<std::tuple<int, int>> kw_ranges_tmp;
    for (int kwe = ep.rp; kwe > 0; kwe -= ep.ws) {
      kw_ranges_tmp.push_back(std::make_tuple(0, ep.kw - kwe - 1));
    }
    _wacc_wfr_ = kw_ranges_tmp.size();
    for (auto t = kw_ranges_tmp.rbegin(); t != kw_ranges_tmp.rend(); ++t) {
      wacc_w_ranges_.push_back(*t);
    }
    for (auto i : wacc_w_ranges_) {
      el_log(__DEBUG, "kw_ranges: %d, %d", std::get<0>(i), std::get<1>(i));
    }

    wacc_wt_ = ep.wt >= 3 ? 3 : ep.wt; // left T, middle T (full), right Tr
    wacc_h_ = wacc_h_ranges_.size();
    wacc_w_ = wacc_w_ranges_.size();
    _wacc_ohfs_ = _wacc_hf_;
    _wacc_ohfe_ = ep.oh - _wacc_hfr_ - 1;
    _wacc_owfs_ = _wacc_wf_;
    _wacc_owfe_ = ep.ow - _wacc_wfr_ - 1;
  }

  // Debug
  el_log(__DEBUG, 
         "wacc_h_:%d, wacc_w_:%d, _wacc_hf_:%d, _wacc_wf_:%d, _wacc_hfr_:%d, "
         "_wacc_wfr_:%d, wacc_wt_:%d, wacc_wT_:%d, _wacc_ohfs_:%d, "
         "_wacc_ohfe_:%d, _wacc_owfs_:%d, _wacc_owfe_:%d",
         wacc_h_, wacc_w_, _wacc_hf_, _wacc_wf_, _wacc_hfr_, _wacc_wfr_, wacc_wt_,
         wacc_wT_, _wacc_ohfs_, _wacc_ohfe_, _wacc_owfs_, _wacc_owfe_);
}

Template_elx_int8_conv_direct_t void
Instance_elx_int8_conv_direct_t::__trans_weights_acc(float *weights_scale,
                                                   float *weights_shift,
                                                   int8_t *tweights_s8,
                                                   BiasType *bias) {
  auto V1 = compact_ir_weights_ ? ep.Ir : ep.V1;
  float *weights_shift_buf =
      (float *)aligned_alloc(64, wacc_h_ * wacc_w_ * ep.OC * sizeof(float));

  // weights-acc
  estl::parallel_for<7>([&](int _wacc_h, int _wacc_w, int _O4, int _O3, int _O1, int _O, int _oV) {
    MD12(int8_t, atweights_s8, tweights_s8, ep.O4, ep.I4, ep.O3,
         ep.I3, ep.kh, ep.kw, ep.O1, ep.I2, V1, ep.O,
         V, ep.Vx);
    MD7(float, atweights_shift_buf, weights_shift_buf, wacc_h_, wacc_w_, ep.O4,
        ep.O3, ep.O1, ep.O, V);

    auto khs = std::get<0>(wacc_h_ranges_[_wacc_h]);
    auto khe = std::get<1>(wacc_h_ranges_[_wacc_h]);
    auto kws = std::get<0>(wacc_w_ranges_[_wacc_w]);
    auto kwe = std::get<1>(wacc_w_ranges_[_wacc_w]);
    int acc  = 0;
    iter_each(_I4, ep.I4) {
      iter_each(_I3, ep.I3) {
        for (int _kh = khs; _kh <= khe; ++_kh) {
          // iter_each (_kh, ep.kh) {
          for (int _kw = kws; _kw <= kwe; ++_kw) {
            // iter_each (_kw, ep.kw) {
            iter_each(_I2, ep.I2) {
              bool last_IV = _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1 &&
                             _I2 == ep.I2 - 1;
              auto V1r = last_IV ? ep.Ir : ep.V1;
              iter_each(_V1, V1r) {
                iter_each(_Vx, ep.Vx) {
                  acc += md12(atweights_s8, _O4, _I4, _O3, _I3, _kh, _kw,
                              _O1, _I2, _V1, _O, _oV, _Vx);
                }
              }
            }
          }
        }
      }
    }
    md7(atweights_shift_buf, _wacc_h, _wacc_w, _O4, _O3, _O1, _O, _oV) = acc;
  }, wacc_h_, wacc_w_, ep.O4, ep.O3, ep.O1, ep.O, V);

  auto out_repS = _mm<V>::set1_ps(ep.output_quant_repS);
  auto out_z = _mm<V>::set1_ps(ep.output_quant_z);
  auto input_S = _mm<V>::set1_ps(ep.input_quant_S);
  auto input_z = _mm<V>::set1_ps(ep.input_quant_z);

  // Combine output restore and requantization scale and shift
  estl::parallel_for<1>([&](int _oc2) {
    MD2(float, atweights_scale, weights_scale, ep.oc2, V);
    MD2(BiasType, abias, bias, ep.oc2, V);
    __m<V> &qs = *(__m<V> *)&md2(atweights_scale, _oc2, 0);

    if (std::is_same<OutputType, float>::value) {
      qs = input_S * qs;
    } else {
      qs = input_S * qs * out_repS;
    }
  }, ep.oc2);

  estl::parallel_for<1>([&](int _oc2) {
    MD2(BiasType, abias, bias, ep.oc2, V);
    MD2(float, atweights_scale, weights_scale, ep.oc2, V);
    MD5(float, atweights_shift, weights_shift, wacc_h_, wacc_wt_, ep.oc2, wacc_wT_, V);
    MD4(float, atweights_shift_buf, weights_shift_buf, wacc_h_, wacc_w_, ep.oc2, V);

    __m<V> qs = *(__m<V> *)&md2(atweights_scale, _oc2, 0);
    __m<V> b = ep.with_bias ? *(__m<V> *)&md2(abias, _oc2, 0) : _mm<V>::setzero_ps();

    iter_each(_wacc_h, wacc_h_) {
      iter_each(_wt, wacc_wt_) {
        iter_each(_T, wacc_wT_) {
          __m<V> &qf =
              *(__m<V> *)&md5(atweights_shift, _wacc_h, _wt, _oc2, _T, 0);
          int _wacc_w = _wacc_wf_;
          if (wacc_wt_ == 1) {
            if (ep.input_quant_z == 0) {
              _wacc_w = _wacc_wf_;
            } else {
              _wacc_w = _T < _wacc_owfs_ ? _T : _T > _wacc_owfe_
                                  ? _T - (ep.T - _wacc_wfr_) + _wacc_wf_ + 1
                                  : _wacc_wf_;
            }
          } else { // wacc_wt_ == 2, 3
            if (_wt == 0) { // first
              _wacc_w = _T < _wacc_wf_ ? _T : _wacc_wf_;
            } else if (_wt == wacc_wt_ - 1) { // last
              _wacc_w = _T >= ep.Tr ? _wacc_wf_ : _T < (ep.Tr - _wacc_wfr_)
                         ? _wacc_wf_
                         : _T - (ep.Tr - _wacc_wfr_) + _wacc_wf_ + 1;
            } else { // middle, if wacc_wt_ == 3
              _wacc_w = _wacc_wf_;
            }
          }
          __m<V> qf_tmp =
              *(__m<V> *)&md4(atweights_shift_buf, _wacc_h, _wacc_w, _oc2, 0);
          if (std::is_same<OutputType, float>::value) {
            qf = b - input_z * qf_tmp * qs;
          } else {
            qf = b * out_repS + out_z - input_z * qf_tmp * qs;
          }
        }
      }
    }
  }, ep.oc2);

  if (weights_shift_buf)
    free(weights_shift_buf);
}

// weights (blocked): oc2, ic2, kh, kw, V, V
// tweights: O4, I4, O3, _I3, kh, kw, O1, I2, V1, O, V, Vx
Template_elx_int8_conv_direct_t void Instance_elx_int8_conv_direct_t::
trans_weights(float *weights_scale, float *weights_shift,
                int8_t *tweights_s8, WeightsType *weights, BiasType *bias) {
  __m<V> mmscale = _mm<V>::set1_ps(EL_INT8_MAX);

  auto Vr = ep.ic % V ? ep.ic % V : V;
  auto V1 = compact_ir_weights_ ? ep.Ir : ep.V1;

  // abs-max
  estl::parallel_for<1>([&](int _oc2) {
    MD6(WeightsType, aweights, weights, ep.oc2, ep.ic2, ep.kh, ep.kw, V, V);
    MD2(float, atweights_scale, weights_scale, ep.oc2, V);

    __m<V> abs_max = _mm<V>::set1_ps(0.0);
    iter_each (_ic2, ep.ic2) {
      auto IV = _ic2 == ep.ic2 - 1 ? Vr : V;
      iter_each (_kh, ep.kh) {
      iter_each (_kw, ep.kw) {
      iter_each (_iV, IV) {
        abs_max = _mm<V>::max_ps(abs_max, _mm512_abs_ps(
            *(__m<V> *)&md6(aweights, _oc2, _ic2, _kh, _kw, _iV, 0)));
      }}}
    }
    _mm512_store_ps(&md2(atweights_scale, _oc2, 0), abs_max);
  }, ep.oc2);

  // quantization
  estl::parallel_for<11>([&](int _O4, int _O3, int _O1,
      int _O, int _I4, int _I3, int _I2, int _kh, int _kw, int _V1, int _Vx) {
    MD12(int8_t, atweights_s8, tweights_s8, ep.O4, ep.I4, ep.O3,
         ep.I3, ep.kh, ep.kw, ep.O1, ep.I2, V1, ep.O, V, ep.Vx);
    MD12(WeightsType, aweights, weights, ep.O4, ep.O3, ep.O1, ep.O,
         ep.I4, ep.I3, ep.I2, ep.kh, ep.kw, ep.V1, ep.Vx, V);
    MD5(float, atweights_scale, weights_scale, ep.O4, ep.O3,
        ep.O1, ep.O, V);
    bool last_IV = _I4 == ep.I4 - 1 && _I3 == ep.I3 - 1 && _I2 == ep.I2 - 1;

    if (!last_IV || _V1 * ep.Vx + _Vx < Vr) {
      __m<V> t0;
      // multi scal
      t0 = _mm<V>::mul_ps(*(__m<V> *)&md12(aweights, _O4, _O3, _O1, _O,
                              _I4, _I3, _I2, _kh, _kw, _V1, _Vx, 0), mmscale);
      t0 = _mm<V>::div_ps(t0, *(__m<V> *)&md5(atweights_scale, _O4, _O3, _O1, _O, 0));

      // rounding
      t0 = _mm<V>::roundscale_ps(t0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
      // int8_t
      TweightsType *rounded = (TweightsType *)&t0;
      #pragma omp simd
      iter_each (_oV, V) {
        md12(atweights_s8,
            _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _V1, _O, _oV, _Vx) =
            (int8_t)rounded[_oV];
      }
    } else {
      #pragma omp simd
      iter_each (_oV, V) {
        md12(atweights_s8,
            _O4, _I4, _O3, _I3, _kh, _kw, _O1, _I2, _V1, _O, _oV, _Vx) = 0;
      }
    }
  }, ep.O4, ep.O3, ep.O1, ep.O, ep.I4, ep.I3, ep.I2,
     ep.kh, ep.kw, V1, ep.Vx);

  // weights-scale
  estl::parallel_for<1>([&](int _oc2) {
    MD2(float, atweights_scale, weights_scale, ep.oc2, V);
    auto t0 = _mm<V>::div_ps(
        *(__m<V> *)&md2(atweights_scale, _oc2, 0), mmscale);
    _mm<V>::store_ps(&md2(atweights_scale, _oc2, 0), t0);
  }, ep.oc2);

  __trans_weights_acc(weights_scale, weights_shift, tweights_s8, bias);
}

Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::conv_c160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, float *src_scale, float *weights_scale,
    float *weights_shift, int _I4, int _O4, int _ht, int _wt)
{
  auto V1 = compact_ir_weights_ ? ep.Ir : ep.V1;

  auto ker_conv = _wt == ep.wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  int kws = _wt == 0 ? ep.lp : 0;
  int kwe = _wt == ep.wt - 1 ? ep.kw - ep.rp : ep.kw;

  auto _ih = _ht * ep.hs + (ep.kh / 2) - ep.tp;
  auto _iw = _wt * ep.T * ep.ws + (ep.kw / 2) - ep.lp;
  int pad_l = _wt == 0 ? ep.lp : 0;
  int pad_r = _wt == ep.wt - 1 ? ep.rp : 0;

  int _wacc_wt =
      (_wt == 0 || wacc_wt_ == 1) ? 0 : _wt == ep.wt - 1 ? wacc_wt_ - 1 : 1;
  int _wacc_h =
      _ht < _wacc_ohfs_
          ? _ht
          : _ht <= _wacc_ohfe_ ? _wacc_hf_ : _wacc_ohfs_ + _ht - _wacc_ohfe_;

  MD3(int8_t, aweights, weights_s8, ep.O3, ep.I3, ep.kh * ep.kw *
      ep.O2 * ep.I2 * V1 * V * ep.Vx);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);
  MD2(float, aweights_scale, weights_scale, ep.O3, ep.O2 * V);
  MD5(float, aweights_shift, weights_shift, wacc_h_, wacc_wt_, ep.O4, ep.O3, ep.O2 * ep.T * V);
  MD2(float, asrc_scale, src_scale, 2, ep.T);
  // nhwc
  MD3(InputType, ainput0_nhwc, input_u8, ep.ih, ep.iw, ep.ic);
  MD3(InputType, ainput1_nhwc, &md3(ainput0_nhwc, _ih, _iw, 0), ep.I4, ep.I3, ep.I2 * V);
  MD2(OutputType, aoutput_nhwc, output, ep.O3, ep.O2 * V);
  MD2(ToutputType, atoutput_nhwc, toutput, ep.O3, ep.O2 * V);
  // blocked
  MD5(InputType, ainput_blocked, input_u8, ep.I3, ep.I2, ep.ih, ep.iw, V);
  MD2(OutputType, aoutput_blocked, output, ep.O3, ep.O2 * ep.ht * ep.ow * V);
  MD2(ToutputType, atoutput_blocked, toutput, ep.O3, ep.O2 * ep.ht * ep.ow * V);

  iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr =
          (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;

      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        attr = set_bit(attr, AT_RESTORE_OUTPUT_MASK);
        if (ep.Ir != ep.V1) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      auto ainput = ep.input_fmt == nhwc
                          ? &md3(ainput1_nhwc, 0, _I3, 0)
                          : &md5(ainput_blocked, _I3, 0, _ih, _iw, 0);
      auto aoutput = ep.output_fmt == nhwc
                          ? &md2(aoutput_nhwc, _O3, 0)
                          : &md2(aoutput_blocked, _O3, 0);
      auto atoutput = ep.output_fmt == nhwc
                          ? &md2(atoutput_nhwc, _O3, 0)
                          : &md2(atoutput_blocked, _O3, 0);
      ker_conv(ep, atoutput, aoutput, ainput, &md3(aweights, _O3, _I3, 0),
               &md2(abias, _O3, 0), &md2(asrc_scale, 0, 0),
               &md2(asrc_scale, 1, 0), &md2(aweights_scale, _O3, 0),
               &md5(aweights_shift, _wacc_h, _wacc_wt, 0, _O3, 0),
               khs, khe, kws, kwe, pad_l, pad_r, attr);
    }
  }
}

// slow path
Template_elx_int8_conv_direct_t
void Instance_elx_int8_conv_direct_t::gemm_a160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, float *src_scale, float *weights_scale,
    float *weights_shift, int _I4, int _O4, int _ht, int _wt)
{
  // input:   I3*, I2, ht*, hs*, wt*, T, ws, V1, Vx
  // output:  O3*, O2, ht*, wt*, T, V
  MD5(int8_t, aweights, weights_s8, ep.O3, ep.I3, ep.kh, ep.kw,
      ep.O2 * ep.I2 * ep.V1 * V * ep.Vx);
  MD3(BiasType, abias, bias, ep.O3, ep.O2, V);
  MD3(float, aweights_scale, weights_scale, ep.O3, ep.O2, V);
  MD3(float, aweights_shift, weights_shift, ep.O3, ep.O2, V);
  MD2(float, asrc_scale, src_scale, 2, ep.T);

  int Tz = _wt == ep.wt - 1 ? ep.Tr : ep.T;
  int ows0 = _wt * ep.T;
  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  assert(ep.T > ep.lp);
  assert(ep.Tr > ep.rp);

  // nhwc
  MD3(InputType, ainput0_nhwc, input_u8, ep.ih, ep.iw, ep.ic);
  MD3(OutputType, aoutput0_nhwc, output, ep.ht, ep.ow, ep.oc);
  MD3(ToutputType, atoutput0_nhwc, toutput, ep.ht, ep.ow, ep.OC);
  // blocked
  MD5(InputType, ainput_blocked, input_u8,
      ep.I3, ep.I2, ep.ih, ep.iw, ep.V1 * ep.Vx);
  MD5(OutputType, aoutput_blocked, output,
      ep.O3, ep.O2, ep.ht, ep.ow, V);
  MD5(ToutputType, atoutput_blocked, toutput,
      ep.O3, ep.O2, ep.ht, ep.ow, V);

  iter_each (_O3, ep.O3) {
  iter_each (_I3, ep.I3) {
    if (_I4 == 0 && _I3 == 0) {
      __m<V> s = _mm<V>::setzero_ps();
      iter_each (_O2, ep.O2) {
      iter_each (_T, Tz) {
        MD4(ToutputType, atoutput1_nhwc,
            &md3(atoutput0_nhwc, _ht, ows0 + _T, 0),
            ep.O4, ep.O3, ep.O2, V);
        auto atout = ep.output_fmt == nhwc
                   ? &md4(atoutput1_nhwc, 0, _O3, _O2, 0)
                   : &md5(atoutput_blocked, _O3, _O2, _ht, ows0 + _T, 0);

        if (I == ISA_AVX512 && std::is_same<ToutputType, float>::value)
          _mm<V>::store_ps(atout, s);
        else
          el_error("direct: a160: unimplemented");
      }}
    }
    int attr = attr_;
    if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
      if (ep.Ir != ep.V1) attr = set_bit(attr, AT_Ir_MASK);
    }

    for (int _kh = khs; _kh < khe; ++_kh) {
      auto _ih = ep.hs * _ht + _kh - ep.tp;
      for (int _kw = 0; _kw < ep.kw; ++_kw) {
        auto _iws = ep.ws * ows0 + _kw - ep.lp;
        while (_iws < 0) _iws += ep.ws;
        auto _ows = (_iws + ep.lp - _kw) / ep.ws;

        MD4(InputType, ainput1_nhwc, &md3(ainput0_nhwc, _ih, _iws, 0),
            ep.I4, ep.I3, ep.I2, V);
        MD4(ToutputType, atoutput1_nhwc, &md3(atoutput0_nhwc, _ht, _ows, 0),
            ep.O4, ep.O3, ep.O2, V);
        auto ain = ep.input_fmt == nhwc
                 ? &md4(ainput1_nhwc, 0, _I3, 0, 0)
                 : &md5(ainput_blocked, _I3, 0, _ih, _iws, 0);
        auto atout = ep.output_fmt == nhwc
                 ? &md4(atoutput1_nhwc, 0, _O3, 0, 0)
                 : &md5(atoutput_blocked, _O3, 0, _ht, _ows, 0);
        ker_gemm_[_wt][_kw](ep, atout, nullptr, ain,
            &md5(aweights, _O3, _I3, _kh, _kw, 0),
            &md3(abias, _O3, 0, 0), attr,
            nullptr, nullptr, nullptr, nullptr);
      }
    }

    __mmask16 k = _cvtu32_mask16((1 << ep.Or) - 1);
    if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
      iter_each (_O2, ep.O2) {
      iter_each (_T, Tz) {
        MD4(ToutputType, atoutput1_nhwc, &md3(atoutput0_nhwc, _ht, ows0 + _T, 0),
            ep.O4, ep.O3, ep.O2, V);
        MD4(OutputType, aoutput1_nhwc, &md3(aoutput0_nhwc, _ht, ows0 + _T, 0),
            ep.O4, ep.O3, ep.O2, V);
        auto atout = ep.output_fmt == nhwc
            ? &md4(atoutput1_nhwc, 0, _O3, _O2, 0)
            : &md5(atoutput_blocked, _O3, _O2, _ht, ows0 + _T, 0);
        auto aout = ep.output_fmt == nhwc
            ? &md4(aoutput1_nhwc, 0, _O3, _O2, 0)
            : &md5(aoutput_blocked, _O3, _O2, _ht, ows0 + _T, 0);

        if (I == ISA_AVX512 && std::is_same<ToutputType, float>::value) {
          __m<V> tout = _mm<V>::cvtepi32_ps(*(__i<V> *)atout);
          // restore and requantization
          auto scale = *(__m<V> *)&md3(aweights_scale, _O3, _O2, 0);
          auto shift = *(__m<V> *)&md3(aweights_shift, _O3, _O2, 0);
          tout = tout * scale + shift;
          // fuse relu
          if (ep.with_relu) {
            auto lower = *(__m<V> *)(ep.relu_bound_lower_vec);
            auto upper = *(__m<V> *)(ep.relu_bound_upper_vec);
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
            if (ep.with_argmax)
              _mm<V>::store_ps(atout, tout);
            else {
              bool last_oc = (_O4 == ep.O4 - 1
                              && _O3 == ep.O3 - 1
                              && _O2 == ep.O2 - 1);
              if (last_oc) {
                _mm<V>::mask_store_ps(aout, k, tout);
              } else {
                _mm<V>::store_ps(aout, tout);
              }
            }
          } else {
            el_error("direct: a160: unimplemented");
          }
        } else
          el_error("direct: a160: unimplemented");
      }}
    }
  }}
}

// int8-u8f32u8f32
template class elx_int8_conv_direct_t<conv::U8F32U8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;
// int8-u8f32s8f32
template class elx_int8_conv_direct_t<conv::U8F32S8F32, conv_impl::INT8_F32, 16, ISA_AVX512>;
// int8-u8f32f32f32
template class elx_int8_conv_direct_t<conv::U8F32F32F32, conv_impl::INT8_F32, 16, ISA_AVX512>;

} // namespace euler
