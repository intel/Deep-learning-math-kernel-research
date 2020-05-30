#include "el_intrin.hpp"
#include "el_stl.hpp"
#include "el_utils.hpp"
#include "el_parallel.hpp"
#include "elx_conv_direct_vmg.hpp"
#include "elx_conv_direct_vmg_bind.hpp"
#include "elx_conv_direct_vmg_xopt.hpp"

namespace euler {

Template_elx_conv_direct_vmg_t
Instance_elx_conv_direct_vmg_t::elx_conv_direct_vmg_t(eld_conv_t &dc)
    : elx_conv_t(dc)
{
  // user input
  //xopt_ = ep.execution_mode;
  xopt_ = 0xc060;
  mthr_ = estl::max_concurrency();

  ep.G = 1;
  ep.vmg = 1;
  ep.grp = ep.g;
  ep.Vx = 1;
  ep.V1 = V / ep.Vx;
  ep.ocg = ep.oc / ep.g;
  ep.icg = ep.ic / ep.g;

  // oc = ic = 16x && ocg = icg = 1|2|4|8|16 (V=16)
  bool shape_ok = ep.ic % ep.g == 0 && ep.oc % ep.g == 0 &&
                  (ep.ocg <= V) && (V % ep.ocg == 0) &&
                  (ep.oc % V == 0) && (ep.ic == ep.oc) &&
                  estl::any_of(ep.kh, 3, 5, 7) &&
                  estl::any_of(ep.kw, 3, 5, 7) && (ep.ws == 1) &&
                  ep.lp == (ep.kw / 2) && (ep.tp == ep.kh / 2);
  if (!shape_ok) {
    el_error("direct_vmg: shape not supported");
  }

  // compute multiple groups in one FMA
  // vector multi-group number
  // C = ocg|icg; G = vmg, V = C * G
  // grp = g * G
  ep.vmg = V / ep.ocg;
  ep.g /= ep.vmg;
  if (ep.O != 1) {
    ep.O = 1;
    el_warn("conv: group: O!=1 found for vector multi-group");
  }
  ep.ic /= ep.g;
  ep.oc /= ep.g;

  ep.G = ep.vmg;
  C = ep.ocg;

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

  ep.ic2 = ep.IC / V;
  ep.oc2 = ep.OC / V;

  // n, t2, (T, Tr)
  ep.ht = ep.oh;
  ep.wt = (ep.ow + ep.T - 1) / ep.T;
  ep.Tr = ep.ow % ep.T ? ep.ow % ep.T : ep.T;
  ep.nt = ep.oh * ep.ow;
  ep.t2 = ep.nt / ep.T;
  ep.t  = ep.nt * ep.n;

  if (ep.T < ep.lp || ep.Tr < ep.rp) {
    el_error("Unimplemented T: (T,Tr) < (lp,rp)");
  }
  bool format_ok = estl::any_of(ep.weights_fmt, ghwio) &&
                   estl::any_of(ep.input_fmt, nchw, nChw16c) &&
                   estl::any_of(ep.output_fmt, nchw, nChw16c);
  if (!format_ok) {
    el_error("direct: format not supported");
  }

  // TODO: Ir/Or support?
  ep.Ir = ep.ic % C ? ep.ic % C : C;
  ep.Or = ep.oc % V ? ep.oc % V : V;
  ep.ormask = (1 << ep.Or) - 1;

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

  prepare_execute_opt();
  bind_execute_functions();

  // dbg
  el_log(__DEBUG, "T=%d, Tr=%d, t2=%d, ht=%d, wt=%d, t=%d",
         ep.T, ep.Tr, ep.t2, ep.ht, ep.wt, ep.t);
  el_log(__DEBUG, "V=%d, Ir=%d, I2=%d, I3=%d, I4=%d, IC=%d, g=%d",
         V, ep.Ir, ep.I2, ep.I3, ep.I4, ep.IC, ep.g);
  el_log(__DEBUG, "V=%d, Or=%d, O2=%d (O=%d, O1=%d), O3=%d, O4=%d, O2r=%d, O3r=%d, OC=%d, g=%d, G=%d, C=%d",
         V, ep.Or, ep.O2, ep.O, ep.O1,
         ep.O3, ep.O4, ep.O2r, ep.O3r, ep.OC, ep.g, ep.G, C);
}

Template_elx_conv_direct_vmg_t
int Instance_elx_conv_direct_vmg_t::prepare_execute_opt()
{
  if (ep.with_ip_sum && ep.with_relu && ep.output_fmt != nChw16c) {
    el_error("Unimplemented: fuse sum (plain format) and relu together");
  }

  toutput_size_ = 0;
  tweights_size_ = 0;
  tweights_ = nullptr;
  toutput_ = nullptr;

  switch (xopt_) {
  case 0xc060:
    tweights_size_ = ep.g * ep.G * ep.kh * ep.kw * C * C * sizeof(TweightsType);
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
  if (ep.weights_fmt == hwio || ep.weights_fmt == ghwio) {
    estl::parallel_for<4>([&](int _g, int _kh, int _kw, int _iV) {
      MD6(WeightsType, aweights, weights, ep.g, ep.G, ep.kh, ep.kw, C, C);
      MD5(TweightsType, atweights, tweights, ep.g, ep.kh, ep.kw, C, V);
      WeightsType w[V] = { 0 };
      iter_each (_G, ep.G) {
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
    }, ep.g, ep.kh, ep.kw, C);
  } else {
    el_error("Unimplemented weights format\n");
  }
  // clang-format on
}

// kh,kw=odd, lp=rp=standard, ih=oh*hs, iw=ow*ws, hs=ws=1
Template_elx_conv_direct_vmg_t void
Instance_elx_conv_direct_vmg_t::conv_c060(OutputType *output,
    InputType *input, TweightsType *weights, BiasType *bias, int _I4, int _O4,
    int _ht, int _wt)
{
  // input:   I3*, I2, V, ht*, hs*, wt*, T, ws
  // output:  O3*, O2, ht*, wt*, T, V
  MD3(TweightsType, aweights, weights, ep.O3, ep.I3,
      ep.kh * ep.kw * ep.O2 * ep.I2 * V * C);
  MD2(BiasType, abias, bias, ep.O3, ep.O2 * V);

  auto ker_conv = _wt == ep.wt - 1 ? ker_conv_Tr_ : ker_conv_;

  int khs = estl::max(0, ep.tp - ep.hs * _ht);
  int khe = estl::min(ep.kh, ep.ih + ep.tp - ep.hs * _ht);
  int kws = _wt == 0 ? ep.lp : 0;
  int kwe = _wt == ep.wt - 1 ? ep.kw - ep.lp : ep.kw;
  assert(ep.T > ep.lp && ep.Tr > ep.rp);

  if (ep.input_fmt == nhwc) {
    MD2(InputType, ainput, input, ep.I3, ep.I2 * V);
    MD2(OutputType, aoutput, output, ep.O3, ep.O2 * V);

    iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != C) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      if (ep.Or != V && _O4 == ep.O4 - 1 && _O3 == ep.O3 - 1) {
        attr = set_bit(attr, AT_Or_MASK);
      }
      ker_conv(ep, &md2(aoutput, _O3, 0),
          &md2(ainput, _I3, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  } else {
    // blocked or nchw
    MD2(InputType, ainput, input, ep.I3, ep.I2 * V * ep.ih * ep.iw);
    MD2(OutputType, aoutput, output, ep.O3, ep.O2 * ep.ht * ep.ow * V);

    iter_each(_O3, ep.O3) {
    iter_each(_I3, ep.I3) {
      int attr = (_I4 == 0 && _I3 == 0) ? set_bit(attr_, AT_CLEAR_OUTPUT_MASK) : attr_;
      if (_I4 == ep.I4 - 1 && _I3 == ep.I3 - 1) {
        if (ep.Ir != C) attr = set_bit(attr, AT_Ir_MASK);
        if (ep.with_relu) attr = set_bit(attr, AT_RELU_MASK);
      }
      ker_conv(ep, &md2(aoutput, _O3, 0),
          &md2(ainput, _I3, 0), &md3(aweights, _O3, _I3, 0),
          &md2(abias, _O3, 0), _wt, khs, khe, kws, kwe, attr);
    }}
  }
}


// fp32-f32f32f32
template class elx_conv_direct_vmg_t<conv::FP32, conv_impl::FP32, 16, ISA_AVX512>;
// fp32-f32f16f32
template class elx_conv_direct_vmg_t<conv::FP32, conv_impl::FP32_F16w, 16, ISA_AVX512>;

#ifdef ENABLE_USER_FP16
// fp16o-f32f32f16
template class elx_conv_direct_vmg_t<conv::FP16O, conv_impl::FP32_F16o, 16, ISA_AVX512>;
#endif

} // namespace euler
