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
//  a061 |  a  |   t+o  |  I  | plain, stride>=1, padding, Ir, oh=wt*T, I4=1
// ------+-----+--------+-----+--------------------------------------
//  f061 |  a  |   t+o  |  I  | plain, stride=1, Ir, Tr, I4=1
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
  // weights: O4*, O3, O2, I4*, I3, I2, V, V
  // input:   n*, I4*, I3, I2, t2*, T(Tr), V
  // output:  n*, O4*, O3, O2, t2*, T(Tr), V

  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  estl::parallel_for<4, 1>(mthr_, [&](int _n, int _I4, int _O4, int _t2) {
    MD3(InputType, ainput, input, this->n, this->I4,
        this->I3 * this->I2 * this->ih * this->iw * V);
    MD2(OutputType, aoutput, output, this->n, this->OC * this->oh * this->ow);
    MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);

    MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
      this->O3 * this->I3 * this->O2 * this->I2 * V * V);
    MD2(OutputType, aoutput2, &md2(aoutput, _n, 0), this->O4,
        this->O3 * this->O2 * this->oh * this->ow * V);
    gemm_c060(
        &md2(aoutput2, _O4, 0),
        &md3(ainput, _n, _I4, 0),
        &md3(atweights, _O4, _I4, 0),
        &md2(abias, _O4, 0),
        _I4, _O4, _t2);
  }, this->n, this->I4, this->O4, this->t2);

  if (inference_acc_)
    is_first_run_ = false;
}

Template_elx_conv_direct_1x1_t
void Instance_elx_conv_direct_1x1_t::__execute_b061(
    OutputType *output, InputType *input, WeightsType *weights, BiasType *bias)
{
  // weights: O4*, O3, O2(O2r), I4*, I3, I2, V, V
  // input:   n*, I4*, I3, I2, t2*, T(Tr), V
  // output:  n*, O4*, O3, O2(O2r), t2*, T(Tr), V

  if (is_first_run_) {
    setup_workspace([&]() { trans_weights(tweights_, weights); });
  }

  if (this->O4 == 1) {
    estl::parallel_for<5, 1>(mthr_, [&](int _n, int _I4, int _O4, int _ht, int _wt) {
      MD3(InputType, ainput, input, this->n, this->I4,
          this->I3 * this->I2 * this->ih * this->iw * V);
      MD2(OutputType, aoutput, output, this->n, this->OC * this->oh * this->ow);
      MD5(OutputType, aoutput2, &md2(aoutput, _n, 0), this->O4,
          this->O3 * this->O2, this->ht, this->wt, this->T * V);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);

      MD2(TinputType, atinput, tinput_, mthr_, this->I3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);
      size_t ithr = estl::current_thread_index();
      trans_input(
          &md2(atinput, ithr, 0),
          &md3(ainput, _n, _I4, 0),
          _ht, _wt);
      gemm_b061(
          &md5(aoutput2, _O4, 0, _ht, _wt, 0),
          &md2(atinput, ithr, 0),
          &md3(atweights, _O4, _I4, 0),
          &md2(abias, _O4, 0),
          _I4);
    }, this->n, this->I4, this->O4, this->ht, this->wt);
  } else {
    int n_history = -1;

    estl::parallel_for<5, 1>(mthr_, [&, n_history](int _n, int _I4, int _O4,
                                                   int _ht, int _wt) mutable {
      MD3(InputType, ainput, input, this->n, this->I4,
          this->I3 * this->I2 * this->ih * this->iw * V);
      MD2(OutputType, aoutput, output, this->n, this->OC * this->oh * this->ow);
      MD5(OutputType, aoutput2, &md2(aoutput, _n, 0), this->O4,
          this->O3 * this->O2, this->ht, this->wt, this->T * V);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);

      MD4(TinputType, atinput, tinput_, mthr_, this->ht, this->wt,
          this->I3 * this->I2 * this->T * V);
      MD4(unsigned char, atinput_msk, tinput_msk_, mthr_,
          this->I4, this->ht, this->wt);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);

      int ithr = estl::current_thread_index();

      if (_n != n_history) {
        memset(&md4(atinput_msk, ithr, 0, 0, 0), 0, this->I4 * this->ht * this->wt);
        n_history = _n;
      }
      if (md4(atinput_msk, ithr, _I4, _ht, _wt) == 0) {
        trans_input(
            &md4(atinput, ithr, _ht, _wt, 0),
            &md3(ainput, _n, _I4, 0),
            _ht, _wt);
        md4(atinput_msk, ithr, _I4, _ht, _wt) = 1;
      }
      gemm_b061(
          &md5(aoutput2, _O4, 0, _ht, _wt, 0),
          &md4(atinput, ithr, _ht, _wt, 0),
          &md3(atweights, _O4, _I4, 0),
          &md2(abias, _O4, 0),
          _I4);
    }, this->n, this->I4, this->O4, this->ht, this->wt);
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
    estl::parallel_for<4>(mthr_, [&](int _n, int _O4, int _ht, int _wt) {
      MD5(InputType, ainput0, input, this->n, this->ht, this->hs,
          this->iw, this->ic);
      MD4(InputType, ainput1, &md5(ainput0, _n, _ht, 0, 0, 0), this->wt,
          this->T, this->ws, this->ic);
      MD2(InputType, ainput2, &md4(ainput1, _wt, 0, 0, 0), this->I4,
          this->I3 * this->I2 * V);
      MD4(OutputType, aoutput0, output, this->n, this->ht, this->ow, this->oc);
      MD3(OutputType, aoutput1, &md4(aoutput0, _n, _ht, 0, 0), this->wt,
          this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _wt, 0, 0), this->O4,
          this->O3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);

      gemm_a061(
          &md2(aoutput2, _O4, 0),
          &md2(ainput2, 0, 0),
          &md3(atweights, _O4, 0, 0),
          &md2(abias, _O4, 0),
          0);
    }, this->n, this->O4, this->ht, this->wt);
  } else if (this->O4 == 1) { // nchw
    estl::parallel_for<4>(mthr_, [&](int _n, int _O4, int _ht, int _wt) {
      MD2(InputType, ainput, input, this->n, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->n, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD2(TinputType, atinput, tinput_, mthr_, this->I3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->O3 * this->O2 * this->T * V);

      size_t ithr = estl::current_thread_index();
        trans_input(
            &md2(atinput, ithr, 0),
            &md2(ainput, _n, 0),
            _ht, _wt);
        gemm_a061(
            &md2(atoutput, ithr, 0),
            &md2(atinput, ithr, 0),
            &md3(atweights, _O4, 0, 0),
            &md2(abias, _O4, 0),
            0);
        trans_output(
            &md2(aoutput, _n, 0),
            &md2(atoutput, ithr, 0),
            _O4, _ht, _wt);
    }, this->n, this->O4, this->ht, this->wt);
  } else { // nchw
    int n_history = -1;
    estl::parallel_for<4>(mthr_, [&, n_history]
                          (int _n, int _O4, int _ht, int _wt) mutable {
      MD2(InputType, ainput, input, this->n, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->n, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD4(TinputType, atinput, tinput_, mthr_, this->ht, this->wt,
          this->I3 * this->I2 * this->T * V);
      MD3(unsigned char, atinput_msk, tinput_msk_, mthr_, this->ht, this->wt);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->O3 * this->O2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);
      int ithr = estl::current_thread_index();

      if (_n != n_history) {
        memset(&md3(atinput_msk, ithr, 0, 0), 0, this->ht * this->wt);
        n_history = _n;
      }
      if (md3(atinput_msk, ithr,  _ht, _wt) == 0) {
        trans_input(
            &md4(atinput, ithr, _ht, _wt, 0),
            &md2(ainput, _n, 0),
            _ht, _wt);
        md3(atinput_msk, ithr, _ht, _wt) = 1;
      }
      gemm_a061(
          &md2(atoutput, ithr, 0),
          &md4(atinput, ithr, _ht, _wt, 0),
          &md3(atweights, _O4, 0, 0),
          &md2(abias, _O4, 0),
          0);
      trans_output(
          &md2(aoutput, _n, 0),
          &md2(atoutput, ithr, 0),
          _O4, _ht, _wt);
    }, this->n, this->O4, this->ht, this->wt);
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
    estl::parallel_for<3>(mthr_, [&](int _n, int _O4, int _t2) {
      MD2(InputType, ainput, input, this->n, this->ih * this->iw * this->ic);
      MD3(InputType, ainput1, &md2(ainput, _n, 0), this->t2, this->T, this->ic);
      MD2(InputType, ainput2, &md3(ainput1, _t2, 0, 0), this->I4, this->I3 * this->I2 * V);
      MD2(OutputType, aoutput, output, this->n, this->oh * this->ow * this->oc);
      MD3(OutputType, aoutput1, &md2(aoutput, _n, 0), this->t2, this->T, this->oc);
      MD2(OutputType, aoutput2, &md3(aoutput1, _t2, 0, 0), this->O4,
          this->O3 * this->O2 * V);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);

      gemm_f061(
          &md2(aoutput2, _O4, 0),
          &md2(ainput2, 0, 0),
          &md3(atweights, _O4, 0, 0),
          &md2(abias, _O4, 0),
          _t2, 0);
    }, this->n, this->O4, this->t2);
  } else if (this->O4 == 1) { // nchw
    estl::parallel_for<3>(mthr_, [&](int _n, int _O4, int _t2) {
      MD2(InputType, ainput, input, this->n, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->n, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD2(TinputType, atinput, tinput_, mthr_, this->I3 * this->I2 * this->T * V);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->O3 * this->O2 * this->T * V);

      size_t ithr = estl::current_thread_index();
      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      trans_input2(
           &md2(atinput, ithr, 0),
           &md2(ainput, _n, 0),
           _t2, Tz);
      gemm_f061(
          &md2(atoutput, ithr, 0),
          &md2(atinput, ithr, 0),
          &md3(atweights, _O4, 0, 0),
          &md2(abias, _O4, 0),
          _t2, Tz);
      trans_output2(
          &md2(aoutput, _n, 0),
          &md2(atoutput, ithr, 0),
          _O4, _t2, Tz);
    }, this->n, this->O4, this->t2);
  } else { // nchw
    int n_history = -1;
    estl::parallel_for<3>(mthr_, [&, n_history]
                          (int _n, int _O4, int _t2) mutable {
      MD2(InputType, ainput, input, this->n, this->ic * this->ih * this->iw);
      MD2(OutputType, aoutput, output, this->n, this->oc * this->oh * this->ow);
      MD2(BiasType, abias, bias, this->O4, this->O3 * this->O2 * V);
      MD3(TinputType, atinput, tinput_, mthr_, this->t2,
          this->I3 * this->I2 * this->T * V);
      MD2(unsigned char, atinput_msk, tinput_msk_, mthr_, this->t2);
      MD3(TweightsType, atweights, tweights_, this->O4, this->I4,
          this->O3 * this->I3 * this->O2 * this->I2 * V * V);
      MD2(ToutputType, atoutput, toutput_, mthr_, this->O3 * this->O2 * this->T * V);

      int Tz = _t2 == (this->t2 - 1) ? this->Tr : this->T;
      int ithr = estl::current_thread_index();

      if (_n != n_history) {
        memset(&md2(atinput_msk, ithr, 0), 0, this->t2);
        n_history = _n;
      }
      if (md2(atinput_msk, ithr, _t2) == 0) {
        trans_input2(
            &md3(atinput, ithr, _t2, 0),
            &md2(ainput, _n, 0),
            _t2, Tz);
        md2(atinput_msk, ithr, _t2) = 1;
      }
      gemm_f061(
          &md2(atoutput, ithr, 0),
          &md3(atinput, ithr, _t2, 0),
          &md3(atweights, _O4, 0, 0),
          &md2(abias, _O4, 0),
          _t2, Tz);
      trans_output2(
          &md2(aoutput, _n, 0),
          &md2(atoutput, ithr, 0),
          _O4, _t2, Tz);
    }, this->n, this->O4, this->t2);
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
