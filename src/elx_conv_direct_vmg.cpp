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
  mthr_ = estl::max_concurrency();

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

  this->O4 = 1;
  this->O3 = 1;
  this->O = 1;
  this->O1 = 1;
  this->O2 = this->O * this->O1;
  
  this->I4 = 1;
  this->I3 = 1;
  this->I2 = 1;

  this->ic2 = this->IC / V;
  this->oc2 = this->OC / V;

  xopt_ = 0xa060;

  // n, t2, (T, Tr)
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
  if (this->I4 * this->I3 * this->I2 * V != this->IC) {
    el_error("IC blocking error");
  }
  // OC = V = G * C; oc_orig = g * G * C
  if (this->O4 * this->O3 * this->O2 * V != this->OC) {
    el_error("OC blocking error");
  }

  attr_ = 0x0;
  is_first_run_ = true;
  inference_acc_ = false;
  inference_acc_ = this->prop_kind == forward_inference;

  attr_ = this->with_bias ? set_bit(attr_, AT_BIAS_MASK) : attr_;
  attr_ = this->with_ip_sum ? set_bit(attr_, AT_INP_SUM_MASK) : attr_;

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  printf("T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d\n",
      this->T, this->Tr, this->t2, this->ht, this->wt, this->t);
  printf("V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d, g=%d\n",
      V, this->Ir, this->I2, this->I3, this->I4, this->IC, this->g);
  printf("V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d, g=%d, G=%d, C=%d\n",
      V, this->Or, this->O2, this->O, this->O1,
      this->O3, this->O4, this->O2r, this->O3r, this->OC, this->g, G, C);
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
    estl::parallel_for<4>(mthr_, [&](int _g, int _kh, int _kw, int _iV) {
      MD6(WeightsType, aweights, weights, this->g, G, this->kh, this->kw, C, C);
      MD5(TweightsType, atweights, tweights, this->g, this->kh, this->kw, C, V);
      WeightsType w[V] = { 0 };
      iter_each (_G, G) {
        iter_each (_oV, C) {
          w[_G * C + _oV] = md6(aweights, _g, _G, _kh, _kw, _iV, _oV);
        }
      }

      if (I == ISA_AVX512 && std::is_same<WeightsType, float>::value) {
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
    InputType *input, TweightsType *weights, BiasType *bias, int _I4, int _O4,
    int _ht, int _wt)
{
  // input:   I3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  O3*, O2, ht*, wt*, T, V
  MD3(TweightsType, aweights, weights, this->O3, this->I3,
      this->kh * this->kw * this->O2 * this->I2 * V * C);
  MD2(BiasType, abias, bias, this->O3, this->O2 * V);

  auto ker_conv = _wt == this->wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, this->tp - this->hs * _ht);
  int khe = estl::min(this->kh, this->ih + this->tp - this->hs * _ht);
  int kws = _wt == 0 ? this->lp : 0;
  int kwe = _wt == this->wt - 1 ? this->kw - this->lp : this->kw;
  assert(this->T > this->lp && this->Tr > this->rp);

  if (this->input_fmt == nhwc) {
    MD2(InputType, ainput, input, this->I3, this->I2 * V);
    MD2(OutputType, aoutput, output, this->O3, this->O2 * V);

    iter_each(_O3, this->O3) {
    iter_each(_I3, this->I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == this->I4 - 1 && _I3 == this->I3 - 1) {
        if (this->Ir != C) attr = set_bit(attr, AT_Ir_MASK);
        if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      if (this->Or != V && _O4 == this->O4 - 1 && _O3 == this->O3 - 1) {
        attr = set_bit(attr, AT_Or_MASK);
      }
      ker_conv(*this, &md2(aoutput, _O3, 0),
          &md2(ainput, _I3, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  } else {
    // blocked or nchw
    MD2(InputType, ainput, input, this->I3, this->I2 * V * this->ih * this->iw);
    MD2(OutputType, aoutput, output, this->O3, this->O2 * this->ht * this->ow * V);

    iter_each(_O3, this->O3) {
    iter_each(_I3, this->I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == this->I4 - 1 && _I3 == this->I3 - 1) {
        if (this->Ir != C) attr = set_bit(attr, AT_Ir_MASK);
        if (this->with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      ker_conv(*this, &md2(aoutput, _O3, 0),
          &md2(ainput, _I3, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  }
}

} // namespace euler
