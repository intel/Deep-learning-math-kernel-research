#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct.hpp"

namespace euler {

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::elx_conv_direct_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = this->execution_mode;
  mthr_ = el_get_max_threads();

  this->vmg = 1;
  this->Vx = 1;
  this->V1 = V / this->Vx;
  this->ocg = this->oc / this->g;
  this->icg = this->ic / this->g;

  if (this->g > 1) {
    // nhwc only
    if ((this->icg % V != 0 || this->ocg % V != 0) &&
        (this->input_fmt != nhwc || this->output_fmt != nhwc))
      el_error("Unimplemented: group conv with tail ic/oc needs nhwc input/output");

    this->ic /= this->g;
    this->oc /= this->g;
    if (xopt_ != 0xa060 && xopt_ != 0xd060)
      el_error("Unimplemented: group conv support only 0xa060|0xd060");
  }
  this->IC = ALIGNUP(this->ic, V);
  this->OC = ALIGNUP(this->oc, V);

  if (this->I2 == 0) this->I2 = 1;
  if (this->T == 0)  this->T = 1;
  if (this->O == 0)  this->O = 1;
  if (this->O1 == 0) this->O1 = 1;
  this->O2 = this->O * this->O1;

  this->oc4 = this->oc4 == 0 ? 1 : this->oc4;
  this->ic4 = this->ic4 == 0 ? 1 : this->ic4;

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  // t3, t2, (T, Tr)
  if (xopt_ == 0xa060 || xopt_ == 0xb060 || xopt_ == 0xd060) {
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
    bool format_ok =
        estl::any_of(this->weights_fmt, hwio, ghwio, OIhw16i16o, gOIhw16i16o) &&
        (((this->input_fmt == nhwc) && (this->output_fmt == nhwc)) ||
         (V == 16 && xopt_ == 0xa060 &&
          (estl::any_of(this->input_fmt, nchw, nChw16c)) &&
          (this->output_fmt == nChw16c)) ||
         (V == 16 && xopt_ == 0xb060 && this->g == 1 &&
          (this->input_fmt == nChw16c) && (this->output_fmt == nChw16c)) ||
         (V == 16 && xopt_ == 0xd060 && (this->input_fmt == nChw16c) &&
          (this->output_fmt == nChw16c)));
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xa060 || xopt_ == 0xb060) {
      bool shape_ok = estl::any_of(this->kh, 3, 5, 7)
          && estl::any_of(this->kw, 3, 5, 7)
          && (this->ws == 1 || this->ws == 2)
          && estl::any_of(this->lp, 0, this->kw / 2)
          && estl::any_of(this->tp, 0, this->kh / 2);
      if (!shape_ok) {
        el_error("direct: a060: shape not supported");
      }
    }

    if (this->g == 1 && this->ic < V) {
      bool ok = this->input_fmt == nchw
          && this->weights_fmt == hwio
          && xopt_ == 0xa060;
      if (!ok) {
        el_error("direct: first-conv: support only g=1, xopt=a060 with nchw/hwio");
      }
    }
  }

  this->Ir = this->ic % V ? this->ic % V : V;
  this->Or = this->oc % V ? this->oc % V : V;
  this->ormask = (1 << this->Or) - 1;

  // oc4, (oc3, oc3r), (O2, O2r)
  this->oc34 = (this->oc2 + this->O2 - 1) / this->O2;
  this->O2r = this->oc2 % this->O2;
  if (this->O2r == 0) this->O2r = this->O2;
  this->oc3 = this->oc4; // FIXME, swap order
  this->oc4 = (this->oc34 + this->oc3 - 1) / this->oc3;
  this->oc3r = this->oc34 % this->oc3;
  if (this->oc3r == 0) this->oc3r = this->oc3;

  if (this->O2r != this->O2 || this->oc3r != this->oc3) {
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
  inference_acc_ = false;
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_attr(attr_, bias_idx) : attr_;
  attr_ = this->with_ip_sum ? set_attr(attr_, ip_sum_idx) : attr_;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, ic3=%d, ic4=%d, IC=%d, g=%d\n",
      V, this->Ir, this->I2, this->ic3, this->ic4, this->IC, this->g);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d, g=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC, this->g);
}

Template_elx_conv_direct_t
int Instance_elx_conv_direct_t::prepare_execute_opt()
{
  if (this->with_ip_sum && this->with_relu && this->output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;

  switch (xopt_) {
  case 0xb060:
    toutput_size_ = this->ic4 * this->t3 * this->g * this->OC * this->oh * this->ow * sizeof(ToutputType);
  case 0xa060:
  case 0xd060:
    tweights_size_ = this->g * this->kh * this->kw * this->IC * this->OC * sizeof(TweightsType);
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size_ > 0)
    tweights_size_ += WEIGHTS_MAX_PRELOAD * V;
  scratch_size_ = toutput_size_;

  return 0;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::set_workspace_buffers(void *base)
{
  if (base != nullptr)
    tweights_ = (TweightsType *)base;
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::set_scratch_buffers(void *base)
{
  if (base != nullptr)
    toutput_ = (ToutputType *)base;
}

Template_elx_conv_direct_t
Instance_elx_conv_direct_t::~elx_conv_direct_t()
{
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__trans_weights_post(WeightsType *aweights,
    TweightsType *tweights, int _g, int _oc4, int _ic4, int _oc3, int _ic3,
    int _kh, int _kw, int _O1, int _I2, int _iV, int _O)
{
  int Vr = (this->g == 1) && (this->ic < V) ? this->Ir : V;
  MD12(TweightsType, atweights, tweights, this->g, this->oc4, this->ic4, this->oc3,
       this->ic3, this->kh, this->kw, this->O1, this->I2, Vr, this->O, V);

  if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
    if (std::is_same<TweightsType, float>::value) {
      _mm<V>::store_ps(&md12(atweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw,
                             _O1, _I2, _iV, _O, 0), *(__m<V> *)aweights);
    } else {
      if (this->O == 2) { // fp32 -> bf16
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        if (_O == 0) {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w0 = _mm<V>::and_epi32(si512, mask);
          _mm<V>::store_si512((__i<V> *)&md12(atweights,
              _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0);
        } else {
          auto si512 = _mm<V>::load_si512(aweights);
          auto w1 = _mm<V>::and_epi32(si512, mask);
          auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

          auto w0 = _mm<V>::load_si512(&md12(atweights,
              _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, 0, 0));
          auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
          _mm<V>::store_si512((__i<V> *)&md12(atweights,
              _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0w1);
        }
      } else {            // fp32 -> fp16
        auto fp16v = _mm<V>::cvtps_ph(*(__m<V> *)aweights,
            _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V/2>::store_si256((__m256i *)&md12(atweights,
            _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each (_oV, V) {
      md12(atweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O,
           _oV) = aweights[_oV];
    }
  }
}

Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::__trans_weights_Or_post(WeightsType *aweights,
    TweightsType *tweights, int _g, int _oc4, int _ic4, int _oc3, int _ic3,
    int _kh, int _kw, int _O1, int _I2, int _iV, int _O)
{
  int Vr = (this->g == 1) && (this->ic < V) ? this->Ir : V;
  MD12(TweightsType, atweights, tweights, this->g, this->oc4, this->ic4, this->oc3,
       this->ic3, this->kh, this->kw, this->O1, this->I2, Vr, this->O, V);

  if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
    __mmask16 k = _mm512_int2mask(this->ormask);
    if (std::is_same<TweightsType, float>::value) {
      auto w = _mm<V>::maskz_load_ps(k, aweights);
      _mm<V>::store_ps(&md12(atweights, _g, _oc4, _ic4, _oc3, _ic3,
                       _kh, _kw, _O1, _I2, _iV, _O, 0), w);
    } else {
      if (this->O == 2) { // fp32 -> bf16
        // _O index in this path is 1
        auto mask = _mm<V>::set1_epi32(0xFFFF0000);
        auto si512 = _mm<V>::maskz_load_epi32(k, aweights);
        auto w1 = _mm<V>::and_epi32(si512, mask);
        auto sr_w1 = _mm<V>::bsrli_epi128(w1, 2);

        auto w0 = _mm<V>::load_si512(&md12(atweights,
            _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, 0, 0));
        auto w0w1 = _mm<V>::or_epi32(w0, sr_w1);
        _mm<V>::store_si512((__i<V> *)&md12(atweights,
            _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, 0, 0), w0w1);
      } else {            // fp32 -> fp16
        auto w = _mm<V>::maskz_load_ps(k, aweights);
        auto fp16v = _mm<V>::cvtps_ph(w, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm<V / 2>::store_si256((__m256i *)&md12(atweights,
            _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O, 0), fp16v);
      }
    }
  } else {
    #pragma omp simd
    iter_each(_oV, this->Or) {
      md12(atweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O,
           _oV) = aweights[_oV];
    }
  }
}

// weights (hwio): kh, kw, ic, oc
// weights (blocked): oc2, ic2, kh, kw, V, V
// tweights: oc4, ic4, oc3, _ic3, kh, kw, O1, I2, V, O, V
Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::trans_weights_to_compact(
    TweightsType *tweights, WeightsType *weights)
{
  // clang-format off
  if (this->weights_fmt == OIhw16i16o || this->weights_fmt == gOIhw16i16o) {
    // skip O for tasks allocation, as O == 2 will be optimized by BF16 type
    parallel_for<8, 4>(mthr_, [&](int _g, int _oc4, int _oc3, int _O1, int _O,
                                  int _ic4, int _ic3, int _I2) {
      MD12(WeightsType, aweights, weights, this->g, this->oc4, this->oc3, this->O1,
           this->O, this->ic4, this->ic3, this->I2, this->kh, this->kw, V, V);
      iter_each (_kh, this->kh) {
      iter_each (_kw, this->kw) {
      iter_each (_iV, V) {
        __trans_weights_post(
            &md12(aweights, _g, _oc4, _oc3, _O1, _O, _ic4, _ic3, _I2, _kh, _kw, _iV, 0),
            tweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O);
      }}}
    }, this->g, this->oc4, this->oc3, this->O1, this->O, this->ic4, this->ic3, this->I2);
  } else if (this->weights_fmt == hwio || this->weights_fmt == ghwio) {
    parallel_for<6>(mthr_, [&](int _g, int _kh, int _kw, int _ic4, int _ic3, int _I2) {
      MD5(WeightsType, aweights0, weights, this->g, this->kh, this->kw, this->ic, this->oc);
      auto Ir = _ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1
           && _I2 == this->I2 - 1 ? this->Ir : V;
      iter_each (_iV, Ir) {
      iter_each (_oc4, this->oc4) {
      iter_each (_oc3, this->oc3) {
      iter_each (_O1, this->O1) {
        // handling ic/oc != 16x
        MD5(WeightsType, aweights1, &md5(aweights0, _g, _kh, _kw, 0, 0),
            this->ic4, this->ic3, this->I2, V, this->oc);
        MD5(WeightsType, aweights2, &md5(aweights1, _ic4, _ic3, _I2, _iV, 0),
            this->oc4, this->oc3, this->O1, this->O, V);

        bool is_Or = this->Or != V && _oc4 == this->oc4 - 1
            && _oc3 == this->oc3 - 1 && _O1 == this->O1 - 1;
        auto O = is_Or ? this->O - 1: this->O;
        iter_each(_O, O) {
          __trans_weights_post(&md5(aweights2, _oc4, _oc3, _O1, _O, 0),
              tweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, _O);
        }

        // handling Or
        if (is_Or) {
          __trans_weights_Or_post(&md5(aweights2, _oc4, _oc3, _O1, this->O - 1, 0),
              tweights, _g, _oc4, _ic4, _oc3, _ic3, _kh, _kw, _O1, _I2, _iV, this->O - 1);
        }
      }}}}
    }, this->g, this->kh, this->kw, this->ic4, this->ic3, this->I2);
  } else {
    el_error("Unimplemented weights format\n");
  }
  // clang-format on
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::conv_a060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _ic4, int _oc4,
    int _ht, int _wt)
{
  // input:   ic3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  oc3*, O2, ht*, wt*, T, V
  int Vr = (this->g == 1 && this->ic < V) ? this->Ir : V;
  MD3(TweightsType, aweights, weights, this->oc3, this->ic3,
      this->kh * this->kw * this->O2 * this->I2 * V * Vr);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->rp : this->kw;
  assert(this->T > this->lp && this->Tr > this->rp);

  auto _ih = _ht * this->hs + (this->kh / 2) - this->tp;
  auto _iw = _wt * this->T * this->ws + (this->kw / 2) - this->lp;
  int pad_l = _wt == 0 ? this->lp : 0;
  int pad_r = _wt == this->wt - 1 ? this->rp : 0;

  if (this->input_fmt == nhwc) {
    MD4(InputType, ainput0, input, this->ih, this->iw, this->g, this->ic);
    MD3(InputType, ainput1, &md4(ainput0, _ih, _iw, 0, 0), this->ic4, this->ic3, this->I2 * V);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      if (this->Or != V && _oc4 == this->oc4 - 1 && _oc3 == this->oc3 - 1) {
        attr = set_attr(attr, has_Or_idx);
      }
      ker_conv(*this, &md2(aoutput, _oc3, 0),
          &md3(ainput1, 0, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  } else if (this->input_fmt == nchw) {
    MD4(InputType, ainput, input, this->ic3, this->I2 * V, this->ih, this->iw);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * this->ht * this->ow * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(aoutput, _oc3, 0),
          &md4(ainput, _ic3, 0, _ih, _iw), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  } else { // blocked
    MD5(InputType, ainput, input, this->ic3, this->I2, this->ih, this->iw, V);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * this->ht * this->ow * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(aoutput, _oc3, 0),
          &md5(ainput, _ic3, 0, _ih, _iw, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
    }}
  }
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_t void
Instance_elx_conv_direct_t::conv_b060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _ic4, int _ic3,
    int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  oc3*, O2, ht*, wt*, T, V
  MD3(TweightsType, aweights, weights, this->oc3, this->ic3,
      this->kh * this->kw * this->O2 * this->I2 * V * V);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->rp : this->kw;
  assert(this->T > this->lp && this->Tr > this->rp);

  auto _ih = _ht * this->hs + (this->kh / 2) - this->tp;
  auto _iw = _wt * this->T * this->ws + (this->kw / 2) - this->lp;
  int pad_l = _wt == 0 ? this->lp : 0;
  int pad_r = _wt == this->wt - 1 ? this->rp : 0;

  MD2(OutputType, aoutput_nhwc, output, this->oc3, this->O2 * V);
  MD2(OutputType, aoutput_blocked, output, this->oc3, this->O2 * this->ht * this->ow * V);
  MD3(InputType, ainput_nhwc, input, this->ih, this->iw, this->ic);
  MD4(InputType, ainput_blocked, input, this->I2, this->ih, this->iw, V);

  iter_each(_oc3, this->oc3) {
    OutputType *aout = this->output_fmt == nhwc
                           ? &md2(aoutput_nhwc, _oc3, 0)
                           : &md2(aoutput_blocked, _oc3, 0);
    InputType *ain   = this->input_fmt == nhwc
                           ? &md3(ainput_nhwc, _ih, _iw, 0)
                           : &md4(ainput_blocked, 0, _ih, _iw, 0);
    int attr = 0;
    if (_ic3 == 0) {
      attr = (_ic4 == 0) ? attr_ : attr;
      attr = set_attr(attr, r_output_idx);
    }
    if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
      if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
    }
    if (this->output_fmt == nhwc && this->Or != V && _oc4 == this->oc4 - 1 &&
        _oc3 == this->oc3 - 1) {
      attr = set_attr(attr, has_Or_idx);
    }
    ker_conv(*this, aout, ain, &md3(aweights, _oc3, 0, 0),
             &md2(abias, _oc3, 0), khs, khe, kws, kwe, pad_l, pad_r, attr);
  }
}

// slow path
Template_elx_conv_direct_t
void Instance_elx_conv_direct_t::gemm_d060(OutputType *output, InputType *input,
    TweightsType *weights, BiasType *bias, int _ic4, int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, ht*, hs*, wt*, T, ws, V
  // output:  oc3*, O2, ht*, wt*, T, V
  MD5(TweightsType, aweights, weights, this->oc3, this->ic3, this->kh, this->kw,
      this->O2 * this->I2 * V * V);
  MD3(BiasType, abias, bias, this->oc3, this->O2, V);

  int Tz = _wt == this->wt - 1 ? this->Tr : this->T;
  int ows0 = _wt * this->T;
  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  assert(this->T > this->lp);
  assert(this->Tr > this->rp);

  if (this->input_fmt == nhwc) {
    MD4(InputType, ainput0, input, this->ih, this->iw, this->g, this->ic);
    MD4(OutputType, aoutput0, output, this->ht, this->ow, this->g, this->oc);

    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
      bool oc3_has_Or
          = this->Or != V && _oc4 == this->oc4 - 1 && _oc3 == this->oc3 - 1;
      if (_ic4 == 0 && _ic3 == 0) {
        iter_each (_O2, this->O2) {
          bool O2_has_Or = oc3_has_Or && (_O2 == this->O2 - 1);
          __m<V> s = this->with_bias ? *(__m<V> *)&md3(abias, _oc3, _O2, 0)
                                     : _mm<V>::setzero_ps();
          if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
            __mmask16 k = _mm512_int2mask(O2_has_Or ? this->ormask : 0xFFFF);
            iter_each (_T, Tz) {
              MD4(OutputType, aoutput1, &md4(aoutput0, _ht, ows0 + _T, 0, 0),
                  this->oc4, this->oc3, this->O2, V);
              _mm512_mask_store_ps(&md4(aoutput1, 0, _oc3, _O2, 0), k, s);
            }
          } else el_error("direct: d060: unimplemented");
        }
      }
      int attr = attr_;
      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
      }
      if (oc3_has_Or) {
        attr = set_attr(attr, has_Or_idx);
      }

      for (int _kh = khs; _kh < khe; ++_kh) {
        auto _ih = this->hs * _ht + _kh - this->tp;
        for (int _kw = 0; _kw < this->kw; ++_kw) {
          auto _iws = this->ws * ows0 + _kw - this->lp;
          while (_iws < 0) _iws += this->ws;
          auto _ows = (_iws + this->lp - _kw) / this->ws;

          MD4(InputType, ainput1, &md4(ainput0, _ih, _iws, 0, 0), this->ic4,
              this->ic3, this->I2, V);
          MD4(OutputType, aoutput1, &md4(aoutput0, _ht, _ows, 0, 0), this->oc4,
              this->oc3, this->O2, V);
          ker_gemm_[_wt][_kw](
              *this, &md4(aoutput1, 0, _oc3, 0, 0), &md4(ainput1, 0, _ic3, 0, 0),
              &md5(aweights, _oc3, _ic3, _kh, _kw, 0), &md3(abias, _oc3, 0, 0),
              attr);
        }
      }

      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->with_relu) {
          iter_each (_O2, this->O2) {
            bool O2_has_Or = oc3_has_Or && (_O2 == this->O2 - 1);
            __mmask16 k = _mm512_int2mask(O2_has_Or ? this->ormask : 0xFFFF);
            if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
              iter_each (_T, Tz) {
                MD4(OutputType, aoutput1, &md4(aoutput0, _ht, ows0 + _T, 0, 0),
                    this->oc4, this->oc3, this->O2, V);
                auto s = *(__m<V> *)&md4(aoutput1, 0, _oc3, _O2, 0);
                auto lower = *(__m<V> *)(this->relu_bound_lower_vec);
                auto upper = *(__m<V> *)(this->relu_bound_upper_vec);
                s = _mm<V>::max_ps(s, lower);
                s = _mm<V>::min_ps(s, upper);
                _mm512_mask_store_ps(&md4(aoutput1, 0, _oc3, _O2, 0), k, s);
              }
            } else el_error("direct: d060: unimplemented");
          }
        }
      }
    }}
  } else {
    MD5(InputType, ainput, input, this->ic3, this->I2, this->ih, this->iw, V);
    MD5(OutputType, aoutput, output, this->oc3, this->O2, this->ht, this->ow, V);

    iter_each (_oc3, this->oc3) {
    iter_each (_ic3, this->ic3) {
      if (_ic4 == 0 && _ic3 == 0) {
        iter_each (_O2, this->O2) {
          __m<V> s = this->with_bias ? *(__m<V> *)&md3(abias, _oc3, _O2, 0)
                                     : _mm<V>::setzero_ps();
          iter_each (_T, Tz) {
            if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value)
              _mm<V>::store_ps(&md5(aoutput, _oc3, _O2, _ht, ows0 + _T, 0), s);
            else
              el_error("direct: d060: unimplemented");
          }
        }
      }

      for (int _kh = khs; _kh < khe; ++_kh) {
        auto _ih = this->hs * _ht + _kh - this->tp;
        for (int _kw = 0; _kw < this->kw; ++_kw) {
          auto _iws = this->ws * ows0 + _kw - this->lp;
          while (_iws < 0) _iws += this->ws;
          auto _ows = (_iws + this->lp - _kw) / this->ws;
          ker_gemm_[_wt][_kw](*this, &md5(aoutput, _oc3, 0, _ht, _ows, 0),
              &md5(ainput, _ic3, 0, _ih, _iws, 0),
              &md5(aweights, _oc3, _ic3, _kh, _kw, 0), &md3(abias, _oc3, 0, 0),
              attr_);
        }
      }

      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->with_relu) {
          iter_each (_O2, this->O2) {
          iter_each (_T, Tz) {
            if (I == ISA_SKX_AVX512 && std::is_same<OutputType, float>::value) {
              auto s = *(__m<V> *)&md5(aoutput, _oc3, _O2, _ht, ows0 + _T, 0);
              auto lower = *(__m<V> *)(this->relu_bound_lower_vec);
              auto upper = *(__m<V> *)(this->relu_bound_upper_vec);
              s = _mm<V>::max_ps(s, lower);
              s = _mm<V>::min_ps(s, upper);
              _mm<V>::store_ps(&md5(aoutput, _oc3, _O2, _ht, ows0 + _T, 0), s);
            } else
              el_error("direct: d060: unimplemented");
          }}
        }
      }
    }}
  }
}

} // namespace euler
