#include "elx_conv_direct_lp.hpp"
#include "el_parallel.hpp"

// XOPT
//
// fusion:  same as winograd
// dup:     same as winograd
// ------+-----+--------+-----+------------------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+------------------------------------------------
//  a160 |conv |   t+o  |  -  | blocked/nhwc, Ir, Tr, K=3,5,7 S=1,2
// ------+-----+--------+-----+------------------------------------------------
//  d160 |gemm |   t+o  |  -  | blocked/nhwc, Ir, Tr
// ------+-----+--------+-----+------------------------------------------------
//
namespace euler {

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::__execute_a160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ht*, S, wt*, T, S, V(V1, Vx)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  t3*, oc4*, oc3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    {
      trans_weights_s8(weights_scale_, weights_factor_, tweights_s8_, weights, bias);
    }
    if (this->sampling_kind == CALIBRATED) {
      MD2(TscaleType, atinput_scale, input_scale_, 2, this->T);
      iter_each(_T, this->T) {
        md2(atinput_scale, 0, _T) = this->input_quant_S;
        md2(atinput_scale, 1, _T) = this->input_quant_z;
      }
    }
  }

  auto V1 = compact_ir_weights_ ? this->Ir : this->V1;
  if (this->input_fmt == nhwc) {
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      MD5(InputType, ainput0, input, this->t3, this->ht, this->hs, this->iw,
          this->ic);
      MD4(InputType, ainput1, &md5(ainput0, _t3, _ht, 0, 0, 0), this->wt,
          this->T, this->ws, this->ic);
      MD2(InputType, ainput2, &md4(ainput1, _wt, 0, 0, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
          V1 * this->Vx * V * this->kh * this->kw * this->ic3 * this->oc3
          * this->I2 * this->O2);
      MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->oc);
      MD3(OutputType, aoutput1, &md4(aoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);
      MD4(ToutputType, atoutput0, toutput_, this->t3, this->ht, this->ow, this->oc);
      MD3(ToutputType, atoutput1, &md4(atoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(ToutputType, atoutput2, &md3(atoutput1, _wt, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TscaleType, atweights_scale, weights_scale_, this->oc4,
          this->oc3 * this->O2 * V);
      MD2(TscaleType, aweights_factor, weights_factor_, this->oc4,
          this->oc3 * this->O2 * V);

      conv_a160(&md2(aoutput2, _oc4, 0), &md2(atoutput2, _oc4, 0),
                &md2(ainput2, _ic4, 0), &md3(atweights_s8, _oc4, _ic4, 0),
                &md2(abias, _oc4, 0), input_scale_, &md2(atweights_scale, _oc4, 0),
                &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  } else { // blocked
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      MD6(InputType, ainput0, input, this->t3, this->ic4, this->ic3 * this->I2,
          this->ht, this->hs, this->iw * V);
      MD3(InputType, ainput1, &md6(ainput0, _t3, _ic4, 0, _ht, 0, 0), this->wt,
          this->T * this->ws, V);
      MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
          V1 * this->Vx * V * this->kh * this->kw * this->ic3 * this->oc3
          * this->I2 * this->O2);
      MD5(OutputType, aoutput0, output, this->t3, this->oc4, this->oc3 * this->O2,
          this->ht, this->ow * V);
      MD3(OutputType, aoutput1, &md5(aoutput0, _t3, _oc4, 0, _ht, 0), this->wt,
          this->T, V);
      MD5(ToutputType, atoutput0, toutput_, this->t3, this->oc4, this->oc3 * this->O2,
          this->ht, this->ow * V);
      MD3(ToutputType, atoutput1, &md5(atoutput0, _t3, _oc4, 0, _ht, 0), this->wt,
          this->T, V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TscaleType, atweights_scale, weights_scale_, this->oc4,
          this->oc3 * this->O2 * V);
      MD2(TscaleType, aweights_factor, weights_factor_, this->oc4,
          this->oc3 * this->O2 * V);

      conv_a160(&md3(aoutput1, _wt, 0, 0), &md3(atoutput1, _wt, 0, 0),
                &md3(ainput1, _wt, 0, 0), &md3(atweights_s8, _oc4, _ic4, 0),
                &md2(abias, _oc4, 0), input_scale_, &md2(atweights_scale, _oc4, 0),
                &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::__execute_d160(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // input (blocked): t3*, ic4*, ic3, I2, ih, iw, V(V1, Vx)
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, kh, kw, V(V1, Vx), V
  // output (blocked):  t3*, oc4*, oc3, O2, ht*wt*, T(Tr), V
  if (is_first_run_) {
#pragma omp parallel num_threads(mthr_) proc_bind(close)
    {
      trans_weights_s8(weights_scale_, weights_factor_, tweights_s8_, weights, bias);
    }
    if (this->sampling_kind == CALIBRATED) {
      MD2(TscaleType, atinput_scale, input_scale_, 2, this->T);
      iter_each(_T, this->T) {
        md2(atinput_scale, 0, _T) = this->input_quant_S;
        md2(atinput_scale, 1, _T) = this->input_quant_z;
      }
    }
  }

  if (this->input_fmt == nhwc) {
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      MD4(InputType, ainput0, input, this->t3, this->ih, this->iw, this->ic);
      MD2(InputType, ainput1, &md4(ainput0, _t3, 0, 0, 0),
          this->ic4, this->ic3 * this->I2 * V);
      MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3
          * this->I2 * this->O2);
      MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->oc);
      MD2(OutputType, aoutput1, &md4(aoutput0, _t3, 0, 0, 0),
          this->oc4, this->oc3 * this->O2 * V);
      MD4(OutputType, atoutput0, toutput_, this->t3, this->ht, this->ow, this->oc);
      MD2(ToutputType, atoutput1, &md4(atoutput0, _t3, 0, 0, 0),
          this->oc4, this->oc3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TscaleType, atweights_scale, weights_scale_, this->oc4,
          this->oc3 * this->O2 * V);
      MD2(TscaleType, aweights_factor, weights_factor_, this->oc4,
          this->oc3 * this->O2 * V);

      gemm_d160(&md2(aoutput1, _oc4, 0), &md2(atoutput1, _oc4, 0),
                &md2(ainput1, _ic4, 0), &md3(atweights_s8, _oc4, _ic4, 0),
                &md2(abias, _oc4, 0), input_scale_, &md2(atweights_scale, _oc4, 0),
                &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  } else { // blocked
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      MD3(InputType, ainput, input, this->t3, this->ic4, this->ic3
          * this->I2 * this->ih * this->iw * V);
      MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
          V * V * this->kh * this->kw * this->ic3 * this->oc3
          * this->I2 * this->O2);
      MD3(OutputType, aoutput, output, this->t3, this->oc4,
          this->oc3 * this->O2 * this->ht * this->ow * V);
      MD3(ToutputType, atoutput, toutput_, this->t3, this->oc4,
          this->oc3 * this->O2 * this->ht * this->ow * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TscaleType, atweights_scale, weights_scale_, this->oc4,
          this->oc3 * this->O2 * V);
      MD2(TscaleType, aweights_factor, weights_factor_, this->oc4,
          this->oc3 * this->O2 * V);

      gemm_d160(&md3(aoutput, _t3, _oc4, 0), &md3(atoutput, _t3, _oc4, 0),
                &md3(ainput, _t3, _ic4, 0), &md3(atweights_s8, _oc4, _ic4, 0),
                &md2(abias, _oc4, 0), input_scale_, &md2(atweights_scale, _oc4, 0),
                &md2(aweights_factor, _oc4, 0), _ic4, _oc4, _ht, _wt);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_lp_t
void Instance_elx_conv_direct_lp_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  set_trans_buffers();

  (this->*execute_opt_)((OutputType *)output,
      (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
}

} // namespace euler
