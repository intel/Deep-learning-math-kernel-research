#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct_vmg.hpp"

namespace euler {

Template_elx_conv_direct_vmg_t
Instance_elx_conv_direct_vmg_t::elx_conv_direct_vmg_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  xopt_ = this->execution_mode;
  mthr_ = omp_get_max_threads();

  this->G = 1;
  this->vmg = 1;
  this->grp = this->g;
  this->Vx = 1;
  this->V1 = V / this->Vx;
  this->ocg = this->oc / this->g;
  this->icg = this->ic / this->g;

  // opt: g > 1 && oc = ic = 16x && ocg = 1|2|4|8 (V=16)
  if ((this->g > 1) && (this->ocg < V) && (V % this->ocg == 0) &&
      (this->oc % V == 0) && (this->ic == this->oc)) {
    // compute multiple groups in one FMA
    // vector multi-group number
    // C: ocg; G: vmg, V = C * G
    this->vmg = V / this->ocg;
    this->g /= this->vmg;
    if (this->O != 1) {
      this->O = 1;
      el_warn("conv: group: O!=1 found for vector multi-group");
    }
  }
  this->ic /= this->g;
  this->oc /= this->g;

  this->G = this->vmg;
  this->C = this->ocg;

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
  if (xopt_ == 0xa060 || xopt_ == 0xb060) {
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
          (this->input_fmt == nChw16c) && (this->output_fmt == nChw16c)));
    if (!format_ok) {
      el_error("direct: format not supported");
    }

    if (xopt_ == 0xa060 || xopt_ == 0xb060) {
      bool shape_ok = estl::any_of(this->kh, 3, 5, 7)
          && estl::any_of(this->kw, 3, 5, 7)
          && (this->ws == 1 || this->ws == 2)
          && this->lp == (this->kw / 2) && (this->tp == this->kh / 2);
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

  this->Ir = this->ic % C ? this->ic % C : C;
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
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), oc3=%d, oc4=%d, O2r=%d, oc3r=%d, OC=%d, g=%d, G=%d, C=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->oc3, this->oc4, this->O2r, this->oc3r, this->OC, this->g, G, C);
}

Template_elx_conv_direct_vmg_t
int Instance_elx_conv_direct_vmg_t::prepare_execute_opt()
{
  if (this->with_ip_sum && this->with_relu && this->output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;
  scratch_ = nullptr;
  workspace_ = nullptr;

  switch (xopt_) {
  case 0xb060:
    toutput_size_ = this->ic4 * this->t3 * this->g * this->OC * this->oh * this->ow * sizeof(ToutputType);
  case 0xa060:
    tweights_size_ = this->g * G * this->kh * this->kw * C * C * sizeof(TweightsType);
    break;
  default:
    el_error("Unknown xopt!");
    return -1;
    break;
  }

#define WEIGHTS_MAX_PRELOAD 4
  if (tweights_size_ > 0)
    tweights_size_ += WEIGHTS_MAX_PRELOAD * V;

  size_t workspace_size = tweights_size_;
  // TODO: user provided buffer
  if (workspace_size != 0) {
    MEMALIGN64(&workspace_, workspace_size);
    tweights_ = (TweightsType *)workspace_;
  }
  size_t scratchpad_size = toutput_size_;
  if (scratchpad_size != 0) {
    scratch_ = galloc::acquire(scratchpad_size);
  }

  return 0;
}

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::set_trans_buffers()
{
  toutput_ = (ToutputType*)galloc::get();
}

Template_elx_conv_direct_vmg_t
Instance_elx_conv_direct_vmg_t::~elx_conv_direct_vmg_t()
{
  if (workspace_ != nullptr)
    ::free(workspace_);

  galloc::release();
}

// weights: g, G, kh, kw, C(i), C(o)
// tweights: g, kh, kw, C(i), G, C(o)
Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::trans_weights_to_compact(
    TweightsType *tweights, WeightsType *weights)
{
  if (this->weights_fmt == hwio || this->weights_fmt == ghwio) {
    parallel_for<4>(mthr_, [&](int _g, int _kh, int _kw, int _iV) {
      MD6(WeightsType, aweights, weights, this->g, G, this->kh, this->kw, C, C);
      MD5(TweightsType, atweights, tweights, this->g, this->kh, this->kw, C, V);
      WeightsType w[V];
      iter_each (_G, G) {
        iter_each (_oV, C) {
          w[_G * C + _oV] = md6(aweights, _g, _G, _kh, _kw, _iV, _oV);
        }
      }

      if (I == ISA_SKX_AVX512 && std::is_same<WeightsType, float>::value) {
        if (std::is_same<TweightsType, float>::value) {
          _mm<V>::store_ps(&md5(atweights, _g, _kh, _kw, _iV, 0), *(__m<V> *)w);
        } else {
          auto fp16v = _mm<V>::cvtps_ph(
              *(__m<V> *)w, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
          _mm<V / 2>::store_si256(
              (__m256i *)&md5(atweights, _g, _kh, _kw, _iV, 0), fp16v);
        }
      }
    }, this->g, this->kh, this->kw, C);
  } else {
    el_error("Unimplemented weights format\n");
  }
  // clang-format on
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_vmg_t void
Instance_elx_conv_direct_vmg_t::conv_a060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _ic4, int _oc4,
    int _ht, int _wt)
{
  // input:   ic3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  oc3*, O2, ht*, wt*, T, V
  int Vr = (this->g == 1 && this->ic < C) ? this->Ir : C;
  MD3(TweightsType, aweights, weights, this->oc3, this->ic3,
      this->kh * this->kw * this->O2 * this->I2 * V * Vr);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->lp : this->kw;
  assert(this->T > this->lp && this->Tr > this->rp);

  if (this->input_fmt == nhwc) {
    MD2(InputType, ainput, input, this->ic3, this->I2 * V);
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
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  } else {
    // blocked or nchw
    MD2(InputType, ainput, input, this->ic3, this->I2 * V * this->ih * this->iw);
    MD2(OutputType, aoutput, output, this->oc3, this->O2 * this->ht * this->ow * V);

    iter_each(_oc3, this->oc3) {
    iter_each(_ic3, this->ic3) {
      int attr = (_ic4 == 0 && _ic3 == 0) ? set_attr(attr_, r_output_idx) : attr_;
      if (_ic4 == this->ic4 - 1 && _ic3 == this->ic3 - 1) {
        if (this->Ir != V) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  }
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_vmg_t void
Instance_elx_conv_direct_vmg_t::conv_b060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _ic4, int _ic3,
    int _oc4, int _ht, int _wt)
{
  // input:   ic3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  oc3*, O2, ht*, wt*, T, V
  MD3(TweightsType, aweights, weights, this->oc3, this->ic3,
      this->kh * this->kw * this->O2 * this->I2 * C * V);
  MD2(BiasType, abias, bias, this->oc3, this->O2 * V);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->lp : this->kw;
  assert(this->T > this->lp && this->Tr > this->rp);

  MD2(OutputType, aoutput_nhwc, output, this->oc3, this->O2 * V);
  MD2(OutputType, aoutput_blocked, output, this->oc3, this->O2 * this->ht * this->ow * V);

  iter_each(_oc3, this->oc3) {
    OutputType *aout = this->input_fmt == nhwc ? &md2(aoutput_nhwc, _oc3, 0)
                                              : &md2(aoutput_blocked, _oc3, 0);
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
    ker_conv(*this, aout, input, &md3(aweights, _oc3, 0, 0),
             &md2(abias, _oc3, 0), _wt, khs, khe, kws, kwe, attr);
  }
}


} // namespace euler
