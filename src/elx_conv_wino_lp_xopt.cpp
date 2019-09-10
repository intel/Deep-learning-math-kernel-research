#include "el_parallel.hpp"
#include "elx_conv_wino_lp.hpp"

namespace euler {

//
// -------------+------------+--------------+-------------
//  execute-opt | gemm dtype | fusion-along | duplication
// -------------+------------+--------------+-------------
//     A133     |   INT8     |    i + o     |  I + O
// -------------+------------+--------------+-------------
//     A161     |   INT8     |    t + o     |    I
// -------------+------------+--------------+-------------
//     A173     |   INT8     |  i + t + o   |  I + O
// -------------+------------+--------------+-------------
//


Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::__execute_a133(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_quant_scale_, tweights_quant_factor_,
                       tweights_s8_, tweights_, weights, this->oc4);
    });
  }

  MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
  MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

  MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
      A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);

  MD3(TscaleType, atweights_quant_scale, tweights_quant_scale_,
      this->oc4, this->ic4, this->oc3 * this->O2 * V * A * A);
  MD3(TscaleType, aweights_quant_factor, tweights_quant_factor_,
      this->oc4, this->ic4, this->oc3 * this->O2 * V * A * A);

  THREAD_PARALLEL()
  {
    int last_ic4 = -1;
    iter_each (_ic4, this->ic4) {
    iter_each (_oc4, this->oc4) {
      if (_ic4 != last_ic4) {
        trans_input_u8(
            tinput_quant_scale_, tinput_u8_, tinput_, input, _ic4);
        last_ic4 = _ic4;
      }
      THREAD_BARRIER()
      u8s8_gemm.execute_na(toutput_, tinput_u8_,
          &md3(atweights_s8, _oc4, _ic4, 0),
          tinput_quant_scale_, nullptr,
          &md3(atweights_quant_scale, _oc4, _ic4, 0),
          &md3(aweights_quant_factor, _oc4, _ic4, 0),
          _ic4);
      THREAD_BARRIER()
      trans_output(output, toutput_, &md2(abias, _oc4, 0), _oc4, _ic4);
    }}
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::__execute_a161(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_quant_scale_, tweights_quant_factor_,
                       tweights_s8_, tweights_, weights, this->oc4);
    });
  }

  auto t2_history = -1;
  parallel_for<2>(mthr_, [&, t2_history](int _t2, int _oc4) mutable {
    int ithr = el_get_thread_num();
    MD2(TinputType, atinput2, tinput_, mthr_, this->sampling_kind == COARSE ?
        A * A * this->IC * this->T : A * A * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * this->T * this->oc3 * this->O2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
    MD2(TscaleType, atinput_quant_scale, tinput_quant_scale_, mthr_,
        this->sampling_kind == CALIBRATED ? 2 * this->T
                                          : this->ic3 * A * A * 2 * this->T);
    MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
        A * A * this->T * this->IC);
    MD2(int8_t, atweights_s8, tweights_s8_, this->oc4,
        A * A * this->IC * this->oc3 * this->O2 * V);
    MD2(TscaleType, atweights_quant_scale, tweights_quant_scale_,
        this->oc4, this->oc3 * this->O2 * V * A * A);
    MD2(TscaleType, aweights_quant_factor, tweights_quant_factor_,
        this->oc4, this->oc3 * this->O2 * V * A * A);

    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;

    if (t2_history != _t2) {
      trans_input_u8(&md2(atinput_quant_scale, ithr, 0),
          &md2(atinput2_u8, ithr, 0), &md2(atinput2, ithr, 0), input, _t2, Tz);
      t2_history = _t2;
    }
    u8s8_gemm.execute(&md2(atoutput2, ithr, 0),
        &md2(atinput2_u8, ithr, 0),
        &md2(atweights_s8, _oc4, 0),
        &md2(atinput_quant_scale, ithr, 0),
        &md2(atweights_quant_scale, _oc4, 0),
        &md2(aweights_quant_factor, _oc4, 0), _t2, Tz);
    trans_output(output, &md2(atoutput2, ithr, 0),
                 &md2(abias, _oc4, 0), Tz, _t2, _oc4, 0);
  }, this->t2, this->oc4);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::__execute_a173(
    OutputType * __restrict output, InputType * __restrict input,
    WeightsType * __restrict weights, BiasType * __restrict bias)
{
  if (is_first_run_) {
    setup_workspace([&]() {
      trans_weights_s8(tweights_quant_scale_, tweights_quant_factor_,
                       tweights_s8_, tweights_, weights, this->oc4);
    });
  }

  int last_ic4 = -1, last_t2 = -1;
  parallel_for<3, 1>(
      mthr_, [&, last_ic4, last_t2](int _t2, int _ic4, int _oc4) mutable {
    int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
    size_t ithr = el_get_thread_num();
    MD2(TinputType, atinput2, tinput_, mthr_,
        A * A * this->ic3 * this->I2 * V);
    MD2(ToutputType, atoutput2, toutput_, mthr_,
        A * A * this->T * this->oc3 * this->O2 * V);
    MD3(InputType, ainput, input, this->n, this->ic4,
        this->ih * this->iw * this->ic3 * this->I2 * V);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
    MD2(uint8_t, atinput2_u8, tinput_u8_, mthr_,
        A * A * this->T * this->ic3 * this->I2 * V);
    MD3(int8_t, atweights_s8, tweights_s8_, this->oc4, this->ic4,
        A * A * this->ic3 * this->I2 * V * this->oc3 * this->O2 * V);
    MD2(TscaleType, atinput_quant_scale, tinput_quant_scale_, mthr_,
        this->sampling_kind == CALIBRATED ? 2 * this->T
                                          : this->ic3 * A * A * 2 * this->T);
    MD3(TscaleType, atweights_quant_scale, tweights_quant_scale_,
        this->oc4, this->ic4, this->oc3 * this->O2 * V * A * A);
    MD3(TscaleType, aweights_quant_factor, tweights_quant_factor_,
        this->oc4, this->ic4, this->oc3 * this->O2 * V * A * A);

    if (last_ic4 != _ic4 || last_t2 != _t2) {
      trans_input_u8(
          &md2(atinput_quant_scale, ithr, 0),
          &md2(atinput2_u8, ithr, 0), &md2(atinput2, ithr, 0),
          &md3(ainput, 0, _ic4, 0), _t2, Tz);
      last_t2 = _t2; last_ic4 = _ic4;
    }
    u8s8_gemm.execute_na(
        &md2(atoutput2, ithr, 0),
        &md2(atinput2_u8, ithr, 0),
        &md3(atweights_s8, _oc4, _ic4, 0),
        &md2(atinput_quant_scale, ithr, 0),
        &md3(atweights_quant_scale, _oc4, _ic4, 0),
        &md3(aweights_quant_factor, _oc4, _ic4, 0), _t2, Tz, _ic4);
    trans_output(output, &md2(atoutput2, ithr, 0),
        &md2(abias, _oc4, 0), Tz, _t2, _oc4, _ic4);
  }, this->t2, this->ic4, this->oc4);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_wino_lp_t
void Instance_elx_conv_wino_lp_t::execute(
    void * __restrict output, void * __restrict input,
    void * __restrict weights, void * __restrict bias)
{
  if (is_bfmt_)
    return (this->*execute_opt_)((OutputType *)output,
        (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
  else {
    InputType *in = (InputType *)input;
    WeightsType *wei = (WeightsType *)weights;
    OutputType *out = output_as_bfmt_ ? boutput_ : (OutputType *)output;

    if (input_as_bfmt_) {
      parallel_for<3>(mthr_, [&](int _n, int _ic2, int _ih) {
        int v = _ic2 == this->ic2 - 1 ? this->Ir : V;
        MD5(InputType, abinput, binput_, this->n, this->ic2, this->ih, this->iw, V);
        MD4(InputType, ainput, input, this->n, this->ic, this->ih, this->iw);
        iter_each (_iw, this->iw) {
#pragma omp simd
          iter_each (_v, v)
            md5(abinput, _n, _ic2, _ih, _iw, _v)
                = md4(ainput, _n, _ic2 * V + _v, _ih, _iw);
        }
      }, this->n, this->ic2, this->ih);
      in = binput_;
    }

    if (weights_as_bfmt_) {
      parallel_for<3>(mthr_, [&](int _oc2, int _ic2, int _kh) {
        MD6(WeightsType, abweights, bweights_, this->oc2, this->ic2,
            this->kh, this->kw, V, V);
        MD4(WeightsType, aweights, weights, this->oc, this->ic, this->kh, this->kw);
        int iv = _ic2 == this->ic2 - 1 ? this->Ir : V;
        int ov = _oc2 == this->oc2 - 1 ? this->Or : V;
        iter_each (_kw, this->kw) {
        iter_each (_iv, iv) {
#pragma omp simd
        iter_each (_ov, ov) {
          md6(abweights, _oc2, _ic2, _kh, _kw, _iv, _ov)
            = md4(aweights, _oc2 * V + _ov, _ic2 * V + _iv, _kh, _kw);
        }}}
      }, this->oc2, this->ic2, this->kh);
      wei = bweights_;
    }

    // TODO: padding bias

    (this->*execute_opt_)((OutputType *)out,
        (InputType *)in, (WeightsType *)wei, (BiasType *)bias);

    if (output_as_bfmt_) {
      parallel_for<3>(mthr_, [&](int _n, int _oc2, int _oh) {
        MD5(OutputType, aboutput, boutput_, this->n, this->oc2, this->oh, this->ow, V);
        MD4(OutputType, aoutput, output, this->n, this->oc, this->oh, this->ow);
        int v = _oc2 == this->oc2 - 1 ? this->Or : V;

        if (this->with_ip_sum)
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              += md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
        else
          iter_each (_V, v) {
          iter_each (_ow, this->ow) {
            md4(aoutput, _n, _oc2 * V + _V, _oh, _ow)
              = md5(aboutput, _n, _oc2, _oh, _ow, _V);
          }}
      }, this->n, this->oc2, this->oh);
    }
  }
}

} // namespace euler
