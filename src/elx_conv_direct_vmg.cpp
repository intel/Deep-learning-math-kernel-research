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
  mthr_ = el_get_max_threads();

  this->G = 1;
  this->vmg = 1;
  this->grp = this->g;
  this->Vx = 1;
  this->V1 = V / this->Vx;
  this->ocg = this->oc / this->g;
  this->icg = this->ic / this->g;

  // oc = ic = 16x && ocg = icg = 1|2|4|8|16 (V=16)
  bool shape_ok = this->ic % this->g == 0 && this->oc % this->g == 0 &&
                  (this->ocg <= V) && (V % this->ocg == 0) &&
                  (this->oc % V == 0) && (this->ic == this->oc) &&
                  estl::any_of(this->kh, 3, 5, 7) &&
                  estl::any_of(this->kw, 3, 5, 7) && (this->ws == 1) &&
                  this->lp == (this->kw / 2) && (this->tp == this->kh / 2);
  if (!shape_ok) {
    el_error("direct_vmg: shape not supported");
  }

  // compute multiple groups in one FMA
  // vector multi-group number
  // C = ocg|icg; G = vmg, V = C * G
  // grp = g * G
  this->vmg = V / this->ocg;
  this->g /= this->vmg;
  if (this->O != 1) {
    this->O = 1;
    el_warn("conv: group: O!=1 found for vector multi-group");
  }
  this->ic /= this->g;
  this->oc /= this->g;

  this->G = this->vmg;
  this->C = this->ocg;

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

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  xopt_ = 0xa060;

  // t3, t2, (T, Tr)
  this->t3 = this->n;
  this->ht = this->oh;
  this->wt = (this->ow + this->T - 1) / this->T;
  this->Tr = this->ow % this->T ? this->ow % this->T : this->T;
  this->nt = this->oh * this->ow;
  this->t2 = this->nt / this->T;
  this->t  = this->nt * this->n;

  if (this->T <= this->lp || this->Tr <= this->rp) {
    el_error("Unimplemented T: (T,Tr) must greater than (lp,rp)");
  }
  bool format_ok = estl::any_of(this->weights_fmt, ghwio) &&
                   estl::any_of(this->input_fmt, nchw, nChw16c) &&
                   estl::any_of(this->output_fmt, nchw, nChw16c);
  if (!format_ok) {
    el_error("direct: format not supported");
  }

  // TODO: Ir/Or support?
  this->Ir = this->ic % C ? this->ic % C : C;
  this->Or = this->oc % V ? this->oc % V : V;
  this->ormask = (1 << this->Or) - 1;

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

  switch (xopt_) {
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

  workspace_size_ = tweights_size_;
  scratch_size_ = 0;

  return 0;
}

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::set_workspace_buffers(void *base)
{
  if (base != nullptr)
    tweights_ = (TweightsType *)base;
}

Template_elx_conv_direct_vmg_t
void Instance_elx_conv_direct_vmg_t::set_scratch_buffers(void *base)
{
}

Template_elx_conv_direct_vmg_t
Instance_elx_conv_direct_vmg_t::~elx_conv_direct_vmg_t()
{
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
      WeightsType w[V] = { 0 };
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
  MD3(TweightsType, aweights, weights, this->oc3, this->ic3,
      this->kh * this->kw * this->O2 * this->I2 * V * C);
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
        if (this->Ir != C) attr = set_attr(attr, has_Ir_idx);
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
        if (this->Ir != C) attr = set_attr(attr, has_Ir_idx);
        if (this->with_relu) attr = set_attr(attr, relu_idx);
      }
      ker_conv(*this, &md2(aoutput, _oc3, 0),
          &md2(ainput, _ic3, 0), &md3(aweights, _oc3, _ic3, 0),
          &md2(abias, _oc3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  }
}

} // namespace euler
