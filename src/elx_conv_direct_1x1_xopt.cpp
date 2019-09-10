#include <string.h>
#include "el_parallel.hpp"
#include "elx_conv_direct_1x1.hpp"

// XOPT
// kernel options:
//   - a: CCC, s1
//   - b: CCD, s1
//   - c: DCD: s1
// fusion:  same as winograd
// dup:     same as winograd
//
// ------+-----+--------+-----+--------------------------------------
//       | ker | fusion | dup |             notes
// ------+-----+--------+-----+--------------------------------------
//  a061 |  a  |   t+o  |  I  | plain, stride>=1, padding, Ir, oh=wt*T, ic4=1
// ------+-----+--------+-----+--------------------------------------
//  f061 |  a  |   t+o  |  I  | plain, stride=1, Ir, Tr, ic4=1
// ------+-----+--------+-----+--------------------------------------
//  b061 |  b  |   t+o  |  I  | blocked, stride>=1, oh=wt*T
// ------+-----+--------+-----+--------------------------------------
//  c060 |  c  |   t+o  |  -  | blocked, Tr, Or, stride=1
// ------+-----+--------+-----+--------------------------------------
//

namespace euler {

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__execute_c060(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: oc4*, oc3, O2, ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2, t2*, T(Tr), V

  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  parallel_for<4, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _t2) {
    MD3(InputType, ainput, input, this->t3, this->ic4,
        this->ic3 * this->I2 * this->ih * this->iw * V);
    MD2(OutputType, aoutput, output, this->t3, this->OC * this->oh * this->ow);
    MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

    MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
      this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
    MD2(OutputType, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
        this->oc3 * this->O2 * this->oh * this->ow * V);
    gemm_c060(
        &md2(aoutput2, _oc4, 0),
        &md3(ainput, _t3, _ic4, 0),
        &md3(atweights, _oc4, _ic4, 0),
        &md2(abias, _oc4, 0),
        _ic4, _oc4, _t2);
  }, this->t3, this->ic4, this->oc4, this->t2);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__execute_b061(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: oc4*, oc3, O2(O2r), ic4*, ic3, I2, V, V
  // input:   t3*, ic4*, ic3, I2, t2*, T(Tr), V
  // output:  t3*, oc4*, oc3, O2(O2r), t2*, T(Tr), V

  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  if (this->oc4 == 1) {
    parallel_for<5, 1>(mthr_, [&](int _t3, int _ic4, int _oc4, int _ht, int _wt) {
      MD3(InputType, ainput, input, this->t3, this->ic4,
          this->ic3 * this->I2 * this->ih * this->iw * V);
      MD2(OutputType, aoutput, output, this->t3, this->OC * this->oh * this->ow);
      MD5(OutputType, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
          this->oc3 * this->O2, this->ht, this->wt, this->T * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

      MD2(TinputType, atinput, tinput_, mthr_, this->ic3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
      size_t ithr = el_get_thread_num();
      trans_input(
          &md2(atinput, ithr, 0),
          &md3(ainput, _t3, _ic4, 0),
          _ht, _wt);
      gemm_b061(
          &md5(aoutput2, _oc4, 0, _ht, _wt, 0),
          &md2(atinput, ithr, 0),
          &md3(atweights, _oc4, _ic4, 0),
          &md2(abias, _oc4, 0),
          _ic4);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  } else {
    int t3_history = -1;

    parallel_for<5, 1>(mthr_, [&, t3_history](int _t3, int _ic4, int _oc4,
                                              int _ht, int _wt) mutable {
      MD3(InputType, ainput, input, this->t3, this->ic4,
          this->ic3 * this->I2 * this->ih * this->iw * V);
      MD2(OutputType, aoutput, output, this->t3, this->OC * this->oh * this->ow);
      MD5(OutputType, aoutput2, &md2(aoutput, _t3, 0), this->oc4,
          this->oc3 * this->O2, this->ht, this->wt, this->T * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);

      MD4(TinputType, atinput, tinput_, mthr_, this->ht, this->wt,
          this->ic3 * this->I2 * this->T * V);
      MD4(unsigned char, atinput_msk, tinput_msk_, mthr_,
          this->ic4, this->ht, this->wt);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);

      int ithr = el_get_thread_num();

      if (_t3 != t3_history) {
        memset(&md4(atinput_msk, ithr, 0, 0, 0), 0, this->ic4 * this->ht * this->wt);
        t3_history = _t3;
      }
      if (md4(atinput_msk, ithr, _ic4, _ht, _wt) == 0) {
        trans_input(
            &md4(atinput, ithr, _ht, _wt, 0),
            &md3(ainput, _t3, _ic4, 0),
            _ht, _wt);
        md4(atinput_msk, ithr, _ic4, _ht, _wt) = 1;
      }
      gemm_b061(
          &md5(aoutput2, _oc4, 0, _ht, _wt, 0),
          &md4(atinput, ithr, _ht, _wt, 0),
          &md3(atweights, _oc4, _ic4, 0),
          &md2(abias, _oc4, 0),
          _ic4);
    }, this->t3, this->ic4, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__execute_a061(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{

  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  if (this->input_fmt == nhwc) {
    parallel_for<4>(mthr_, [&](int _t3, int _oc4, int _ht, int _wt) {
      MD5(InputType, ainput0, input, this->t3, this->ht, this->hs,
          this->iw, this->ic);
      MD4(InputType, ainput1, &md5(ainput0, _t3, _ht, 0, 0, 0), this->wt,
          this->T, this->ws, this->ic);
      MD2(InputType, ainput2, &md4(ainput1, _wt, 0, 0, 0), this->ic4,
          this->ic3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->t3, this->ht, this->ow, this->oc);
      MD3(OutputType, aoutput1, &md4(aoutput0, _t3, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);

      gemm_a061(
          &md2(aoutput2, _oc4, 0),
          &md2(ainput2, 0, 0),
          &md3(atweights, _oc4, 0, 0),
          &md2(abias, _oc4, 0),
          0);
    }, this->t3, this->oc4, this->ht, this->wt);
  } else if (this->oc4 == 1) { // nchw
    parallel_for<4>(mthr_, [&](int _t3, int _oc4, int _ht, int _wt) {
      MD2(InputType, ainput, input, this->t3, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->t3, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TinputType, atinput, tinput_, mthr_, this->ic3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->oc3 * this->O2 * this->T * V);

      size_t ithr = el_get_thread_num();
        trans_input(
            &md2(atinput, ithr, 0),
            &md2(ainput, _t3, 0),
            _ht, _wt);
        gemm_a061(
            &md2(atoutput, ithr, 0),
            &md2(atinput, ithr, 0),
            &md3(atweights, _oc4, 0, 0),
            &md2(abias, _oc4, 0),
            0);
        trans_output(
            &md2(aoutput, _t3, 0),
            &md2(atoutput, ithr, 0),
            _oc4, _ht, _wt);
    }, this->t3, this->oc4, this->ht, this->wt);
  } else { // nchw
    int t3_history = -1;
    parallel_for<4>(mthr_, [&, t3_history]
                    (int _t3, int _oc4, int _ht, int _wt) mutable {
      MD2(InputType, ainput, input, this->t3, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->t3, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD4(TinputType, atinput, tinput_, mthr_, this->ht, this->wt,
          this->ic3 * this->I2 * this->T * V);
      MD3(unsigned char, atinput_msk, tinput_msk_, mthr_, this->ht, this->wt);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->oc3 * this->O2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
      int ithr = el_get_thread_num();

      if (_t3 != t3_history) {
        memset(&md3(atinput_msk, ithr, 0, 0), 0, this->ht * this->wt);
        t3_history = _t3;
      }
      if (md3(atinput_msk, ithr,  _ht, _wt) == 0) {
        trans_input(
            &md4(atinput, ithr, _ht, _wt, 0),
            &md2(ainput, _t3, 0),
            _ht, _wt);
        md3(atinput_msk, ithr, _ht, _wt) = 1;
      }
      gemm_a061(
          &md2(atoutput, ithr, 0),
          &md4(atinput, ithr, _ht, _wt, 0),
          &md3(atweights, _oc4, 0, 0),
          &md2(abias, _oc4, 0),
          0);
      trans_output(
          &md2(aoutput, _t3, 0),
          &md2(atoutput, ithr, 0),
          _oc4, _ht, _wt);
    }, this->t3, this->oc4, this->ht, this->wt);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__execute_f061(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  if (this->input_fmt == nhwc) {
    parallel_for<3>(mthr_, [&](int _t3, int _oc4, int _t2) {
      MD2(InputType, ainput, input, this->t3, this->ih * this->iw * this->ic);
      MD3(InputType, ainput1, &md2(ainput, _t3, 0), this->t2, this->T, this->ic);
      MD2(InputType, ainput2, &md3(ainput1, _t2, 0, 0), this->ic4, this->ic3 * this->I2 * V);
      MD2(OutputType, aoutput, output, this->t3, this->oh * this->ow * this->oc);
      MD3(OutputType, aoutput1, &md2(aoutput, _t3, 0), this->t2, this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _t2, 0, 0), this->oc4,
          this->oc3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);

      gemm_f061(
          &md2(aoutput2, _oc4, 0),
          &md2(ainput2, 0, 0),
          &md3(atweights, _oc4, 0, 0),
          &md2(abias, _oc4, 0),
          _t2, 0);
    }, this->t3, this->oc4, this->t2);
  } else if (this->oc4 == 1) { // nchw
    parallel_for<3>(mthr_, [&](int _t3, int _oc4, int _t2) {
      MD2(InputType, ainput, input, this->t3, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->t3, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD2(TinputType, atinput, tinput_, mthr_, this->ic3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->oc3 * this->O2 * this->T * V);

      size_t ithr = el_get_thread_num();
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      trans_input2(
           &md2(atinput, ithr, 0),
           &md2(ainput, _t3, 0),
           _t2, Tz);
      gemm_f061(
          &md2(atoutput, ithr, 0),
          &md2(atinput, ithr, 0),
          &md3(atweights, _oc4, 0, 0),
          &md2(abias, _oc4, 0),
          _t2, Tz);
      trans_output2(
          &md2(aoutput, _t3, 0),
          &md2(atoutput, ithr, 0),
          _oc4, _t2, Tz);
    }, this->t3, this->oc4, this->t2);
  } else { // nchw
    int t3_history = -1;
    parallel_for<3>(mthr_, [&, t3_history]
                    (int _t3, int _oc4, int _t2) mutable {
      MD2(InputType, ainput, input, this->t3, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->t3, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->oc4, this->oc3 * this->O2 * V);
      MD3(TinputType, atinput, tinput_, mthr_, this->t2,
          this->ic3 * this->I2 * this->T * V);
      MD2(unsigned char, atinput_msk, tinput_msk_, mthr_, this->t2);
      MD3(TweightsType, atweights, tweights_, this->oc4, this->ic4,
          this->oc3 * this->ic3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->oc3 * this->O2 * this->T * V);

      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      int ithr = el_get_thread_num();

      if (_t3 != t3_history) {
        memset(&md2(atinput_msk, ithr, 0), 0, this->t2);
        t3_history = _t3;
      }
      if (md2(atinput_msk, ithr, _t2) == 0) {
        trans_input2(
            &md3(atinput, ithr, _t2, 0),
            &md2(ainput, _t3, 0),
            _t2, Tz);
        md2(atinput_msk, ithr, _t2) = 1;
      }
      gemm_f061(
          &md2(atoutput, ithr, 0),
          &md3(atinput, ithr, _t2, 0),
          &md3(atweights, _oc4, 0, 0),
          &md2(abias, _oc4, 0),
          _t2, Tz);
      trans_output2(
          &md2(aoutput, _t3, 0),
          &md2(atoutput, ithr, 0),
          _oc4, _t2, Tz);
    }, this->t3, this->oc4, this->t2);
  }

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::execute(
    void *output, void *input, void *weights, void *bias)
{
  if (is_bfmt_)
    (this->*execute_opt_)((OutputType *)output,
        (InputType *)input, (WeightsType *)weights, (BiasType *)bias);
  else {
    InputType *in = input_as_bfmt_ ? binput_ : (InputType *)input;
    WeightsType *wei = weights_as_bfmt_ ? bweights_ : (WeightsType *)weights;
    OutputType *out = output_as_bfmt_ ? boutput_ : (OutputType *)output;

    if (input_as_bfmt_) {
      trans_input_2_blocked(in, (InputType *)input);
    }

    if (weights_as_bfmt_) {
      trans_weights_2_blocked(wei, (WeightsType *)weights);
    }

    // TODO: padding bias
    (this->*execute_opt_)((OutputType *)out,
        (InputType *)in, (WeightsType *)wei, (BiasType *)bias);

    if (output_as_bfmt_) {
      trans_output_2_plain((OutputType *)output, out);
    }
  }
}

} // namespace euler
