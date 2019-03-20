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
  mthr_ = omp_get_max_threads();

  this->Vx = 4;
  this->V1 = V / this->Vx;
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = 1;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  if (this->oc4 == 0) this->oc4 = 1;
  if (this->ic4 == 0) this->ic4 = 1;

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

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
        ((this->input_fmt == nChw16c && this->output_fmt == nChw16c) ||
         (xopt_ == 0xa160 && this->input_fmt == nhwc && this->output_fmt == nhwc));
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xa160 /*|| xopt_ == 0xb160*/) {
      bool shape_ok = estl::any_of(this->kh, 3, 5, 7)
          && estl::any_of(this->kw, 3, 5, 7)
          && (this->ws == 1 || this->ws == 2)
          && this->lp == (this->kw / 2) && (this->tp == this->kh / 2);
      if (!shape_ok) {
        el_error("direct: a160: shape not supported");
      }
    }
  }

  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;

  // oc4, (oc3, oc3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  if (this->Ir != V)
    el_error("ic / 16 != 0 is not implement while doing int8 gemm");

  if (this->Or != V || this->O2r != this->O2 || this->oc3r != this->oc3) {
    el_error("No oc tailing support");
  }

  // ic4, ic3, I3
  this->ic34 = this->ic2 / this->I2;
  this->ic3 = this->ic34 / this->ic4;
  if (this->ic4 * this->ic3 * this->I2 * V != this->IC) {
    el_error("IC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = this->prop_kind == forward_inference;
  if (xopt_ == 0xa160) {
    attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
    // attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;
  }

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
  scratch_ = nullptr;
  workspace_ = nullptr;
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
    weights_factor_size = this->OC * sizeof(TscaleType);
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

  size_t workspace_size = tweights_s8_size_ + weights_scale_size_
      + weights_factor_size_ + input_scale_size_;
  size_t scratchpad_size = toutput_size_;

  // TODO: user provided buffer
  if (workspace_size != 0) {
    MEMALIGN64(&workspace_, workspace_size);
  }
  if (scratchpad_size != 0) {
    scratch_ = galloc::acquire(scratchpad_size);
  }

  printf("nthreads=%d, mthr_=%d\n", this->nthreads, mthr_);
  printf("sampling_kind = %d\n", this->sampling_kind);
  printf("input_quant_S = %f\n", this->input_quant_S);
  printf("input_quant_z = %f\n", this->input_quant_z);
  printf("output_quant_S = %f\n", this->output_quant_S);
  printf("output_quant_z = %f\n", this->output_quant_z);
  return 0;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::set_trans_buffers()
{
  if (workspace_ != nullptr) {
    weights_scale_ = (TscaleType *)workspace_;
    weights_factor_ = (TscaleType *)((char *)weights_scale_ + weights_scale_size_);
    input_scale_ = (TscaleType *)((char *)weights_factor_ + weights_factor_size_);
    tweights_s8_ = (int8_t *)((char *)input_scale_ + input_scale_size_);
  }
  toutput_ = (ToutputType *)galloc::get();
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
  if (workspace_ != nullptr)
    ::free(workspace_);

  galloc::release();
}

// weights (blocked): oc2, ic2, kh, kw, V, V
// tweights: oc4, ic4, oc3, _ic3, kh, kw, O1, I2, V1, O, V, Vx
Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::trans_weights_s8(TscaleType *weights_scale,
    TscaleType *weights_factor, int8_t *tweights_s8, WeightsType *weights, BiasType *bias)
{
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
  __m<V> mmscale = _mm<V>::set1_ps(INT8GEMM_TWT_QTSCALE);

  int ithr = omp_get_thread_num();

  // abs-max
  thread_parallel_for<1>(mthr_, ithr, [&](int _oc2) {
    MD6(WeightsType, aweights, weights, this->oc2, this->ic2, this->kh, this->kw, V, V);
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);

    __m<V> abs_max = _mm<V>::set1_ps(0.0);
    iter_each (_ic2, this->ic2) {
    iter_each (_kh, this->kh) {
    iter_each (_kw, this->kw) {
    iter_each (_iV, V) {
      abs_max = _mm<V>::max_ps(abs_max, _mm512_abs_ps(
          *(__m<V> *)&md6(aweights, _oc2, _ic2, _kh, _kw, _iV, 0)));
    }}}}
    _mm512_store_ps(&md2(atweights_scale, _oc2, 0), abs_max);
  }, this->oc2);
#pragma omp barrier

  // quantization
  thread_parallel_for<11>(mthr_, ithr, [&](int _oc4, int _oc3, int _O1,
      int _O, int _ic4, int _ic3, int _I2, int _kh, int _kw, int _V1, int _Vx) {
    MD12(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4, this->oc3,
         this->ic3, this->kh, this->kw, this->O1, this->I2, this->V1, this->O,
         V, this->Vx);
    MD12(WeightsType, aweights, weights, this->oc4, this->oc3, this->O1, this->O,
         this->ic4, this->ic3, this->I2, this->kh, this->kw, this->V1, this->Vx, V);
    MD5(TscaleType, atweights_scale, weights_scale, this->oc4, this->oc3,
        this->O1, this->O, V);

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
  }, this->oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2,
     this->kh, this->kw, this->V1, this->Vx);
#pragma omp barrier

  // weights-acc
  thread_parallel_for<5>(mthr_, ithr, [&](int _oc4, int _oc3, int _O1, int _O, int _oV) {
    MD12(int8_t, atweights_s8, tweights_s8, this->oc4, this->ic4, this->oc3,
         this->ic3, this->kh, this->kw, this->O1, this->I2, this->V1, this->O,
         V, this->Vx);
    MD5(TscaleType, atweights_factor, weights_factor, this->oc4, this->oc3,
        this->O1, this->O, V);

    int acc = 0;
    iter_each (_ic4, this->ic4) {
    iter_each (_ic3, this->ic3) {
    iter_each (_kh, this->kh) {
    iter_each (_kw, this->kw) {
    iter_each (_I2, this->I2) {
    iter_each (_V1, this->V1) {
    iter_each (_Vx, this->Vx) {
      acc += md12(atweights_s8, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _V1, _O, _oV, _Vx);
    }}}}}}}
    md5(atweights_factor, _oc4, _oc3, _O1, _O, _oV) = acc;
  }, this->oc4, this->oc3, this->O1, this->O, V);

  // weights-scale
  thread_parallel_for<1>(mthr_, ithr, [&](int _oc2) {
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);
    auto t0 = _mm<V>::div_ps(
        *(__m<V> *)&md2(atweights_scale, _oc2, 0), mmscale);
    _mm<V>::store_ps(&md2(atweights_scale, _oc2, 0), t0);
  }, this->oc2);
#pragma omp barrier

  auto out_repS = _mm<V>::set1_ps(this->output_quant_repS);
  auto out_z = _mm<V>::set1_ps(this->output_quant_z);
  auto input_S = _mm<V>::set1_ps(this->input_quant_S);
  auto input_z = _mm<V>::set1_ps(this->input_quant_z);

  // combine output restore and requantization scale and factor
  thread_parallel_for<1>(mthr_, ithr, [&](int _oc2) {
    MD2(TscaleType, atweights_scale, weights_scale, this->oc2, V);
    MD2(TscaleType, atweights_factor, weights_factor, this->oc2, V);
    MD2(BiasType, abias, bias, this->oc2, V);
    __m<V> &qs = *(__m<V> *)&md2(atweights_scale, _oc2, 0);
    __m<V> &qf = *(__m<V> *)&md2(atweights_factor, _oc2, 0);
    __m<V> &b = *(__m<V> *)&md2(abias, _oc2, 0);
    qs = input_S * qs * out_repS;
    qf = out_z - input_z * qf * qs + b * out_repS;
  }, this->oc2);
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::conv_a160(OutputType *output,
    ToutputType *toutput, InputType *input_u8, int8_t *weights_s8,
    BiasType *bias, TscaleType *src_scale, TscaleType *weights_scale,
    TscaleType *weights_factor, int _ic4, int _oc4, int _ht, int _wt)
{
  MD3(int8_t, aweights, weights_s8, this->oc3, this->ic3, this->kh * this->kw *
      this->O2 * this->I2 * this->V1 * V * this->Vx);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);
  MD2(TscaleType, aweights_scale, weights_scale, this->oc3, this->O2 * V);
  MD2(TscaleType, aweights_factor, weights_factor, this->oc3, this->O2  * V);
  MD2(TscaleType, asrc_scale, src_scale, 2, T);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->lp : this->kw;

  if (this->input_fmt == nhwc) {
    MD2(InputType, ainput, input_u8, this->ic3, this->I2 * V);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * V);
    MD2(ToutputType, atoutput, toutput, this->oc3, this->O2 * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;

      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        attr = set_attr(attr, c_output_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(atoutput, _oc3, 0), &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), &md2(asrc_scale, 0, 0), &md2(asrc_scale, 1, 0),
          &md2(aweights_scale, _oc3, 0), &md2(aweights_factor, _oc3, 0),
          _wt, khs, khe, kws, kwe, attr);
    }}
  } else { // blocked
    MD2(InputType, ainput, input_u8, this->ic3, this->I2 * this->ih * this->iw * V);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * this->ht * this->ow * V);
    MD2(ToutputType, atoutput, toutput, this->oc3, this->O2 * this->ht * this->ow * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;

      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        attr = set_attr(attr, c_output_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(atoutput, _oc3, 0), &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), &md2(asrc_scale, 0, 0), &md2(asrc_scale, 1, 0),
          &md2(aweights_scale, _oc3, 0), &md2(aweights_factor, _oc3, 0),
          _wt, khs, khe, kws, kwe, attr);
    }}
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
  MD5(InputType, ainput, input_u8, this->ic3, this->I2, this->ih, this->iw,
      this->V1 * this->Vx);
  MD5(int8_t, aweights, weights_s8, this->oc3, this->ic3, this->kh, this->kw,
      this->O2 * this->I2 * this->V1 * V * this->Vx);
  MD5(OutputType, aoutput, output, this->oc3, this->O2, this->ht, this->ow, V);
  MD5(ToutputType, atoutput, toutput, this->oc3, this->O2, this->ht, this->ow, V);
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

  iter_each (_oc3, this->oc3) {
  iter_each (_ic3, this->ic3) {
    if (_ic4 == 0 && _ic3 == 0) {
      __m<V> s = _mm<V>::setzero_ps();
      iter_each (_O2, this->O2) {
      iter_each (_T, Tz) {
        if (I == ISA_SKX_AVX512 && std::is_same<ToutputType, float>::value)
          _mm<V>::store_ps(&md5(atoutput, _oc3, _O2, _ht, ows0 + _T, 0), s);
        else
          el_error("direct: d160: unimplemented");
      }}
    }

    for (int _kh = khs; _kh < khe; ++_kh) {
      auto _ih = this->hs * _ht + _kh - this->tp;
      for (int _kw = 0; _kw < this->kw; ++_kw) {
        auto _iws = this->ws * ows0 + _kw - this->lp;
        while (_iws < 0) _iws += this->ws;
        auto _ows = (_iws + this->lp - _kw) / this->ws;

        ker_gemm_[_wt][_kw](*this,
            &md5(atoutput, _oc3, 0, _ht, _ows, 0),
            nullptr,
            &md5(ainput, _ic3, 0, _ih, _iws, 0),
            &md5(aweights, _oc3, _ic3, _kh, _kw, 0),
            &md3(abias, _oc3, 0, 0), attr_,
            nullptr, nullptr, nullptr, nullptr);
      }
    }

    if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
      MD5(int, atoutput, toutput, this->oc3, this->O2, this->ht, this->ow, V);
      __m<V> out_repS = _mm<V>::set1_ps(this->output_quant_repS);
      __m<V> out_z = _mm<V>::set1_ps(this->output_quant_z);
      iter_each (_O2, this->O2) {
      iter_each (_T, Tz) {
        if (I == ISA_SKX_AVX512 && std::is_same<ToutputType, float>::value) {
          __m<V> tout = _mm<V>::cvtepi32_ps(
              *(__i<V> *)&md5(atoutput, _oc3, _O2, _ht, ows0 + _T, 0));
          // restore and requantization
          if (std::is_same<OutputType, uint8_t>::value
              || std::is_same<OutputType, int8_t>::value) {
            auto scale = *(__m<V> *)&md3(aweights_scale, _oc3, _O2, 0);
            auto factor = *(__m<V> *)&md3(aweights_factor, _oc3, _O2, 0);
            tout = tout * scale + factor;
          } else {
            el_error("Unsupported output type for int8 direct 1x1");
          }
          // fuse relu
          if (this->with_relu)
            tout = _mm<V>::max_ps(tout, _mm<V>::setzero_ps());
          // rounding
          __i<V> s_out = _mm<V>::cvt_roundps_epi32(
              tout, _MM_FROUND_TO_NEAREST_INT  | _MM_FROUND_NO_EXC);
          __m128i x8_out;
          if (std::is_same<OutputType, int8_t>::value)
            x8_out = _mm<V>::cvtsepi32_epi8(s_out);
          else if (std::is_same<OutputType, uint8_t>::value)
            x8_out = _mm<V>::cvtusepi32_epi8(s_out);
          else
            el_error("Unsupported output type for int8 direct 1x1");
          // store output
          _mm_store_si128((__m128i *)&md5(aoutput, _oc3, _O2, _ht, ows0 + _T, 0),
                          x8_out);
        } else
          el_error("direct: d060: unimplemented");
      }}
    }
  }}
}

} // namespace euler
