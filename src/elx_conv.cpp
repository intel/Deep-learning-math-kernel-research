#include <assert.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"

namespace euler {

template<typename Type>
elx_conv_t<Type>::elx_conv_t (eld_conv_t<Type> &dc)
{
    this->n  = dc.dims.input.n;
    this->ic = dc.dims.input.c;
    this->oc = dc.dims.output.c;
    this->ih = dc.dims.input.h;
    this->iw = dc.dims.input.w;
    this->oh = dc.dims.output.h;
    this->ow = dc.dims.output.w;
    this->kh = dc.dims.weights.h;
    this->kw = dc.dims.weights.w;
    this->lp = dc.pads.l;
    this->rp = dc.pads.r;
    this->tp = dc.pads.t;
    this->bp = dc.pads.b;
    this->hs = dc.strides.h;
    this->ws = dc.strides.w;
    this->hd = dc.dilations.h;
    this->wd = dc.dilations.w;

    this->input_fmt   = dc.formats.input;
    this->weights_fmt = dc.formats.weights;
    this->output_fmt  = dc.formats.output;
    this->with_relu   = dc.with_relu;
    this->with_bias   = dc.with_bias;

    this->tinput   = nullptr;
    this->tweights = nullptr;
    this->toutput  = nullptr;
}

template<typename Type, const int A, const int K, const int V, const int I>
elx_conv_impl_t<Type, A, K, V, I>::elx_conv_impl_t (eld_conv_t<Type> &dc)
: elx_conv_t<Type>(dc) 
{
    this->V = V;
    this->ic2 = this->ic / V;
    this->oc2 = this->oc / V;

    this->A   = A;
    this->oh2 = (this->oh + A - 3) / (A - 2);
    this->ow2 = (this->ow + A - 3) / (A - 2);
    this->ih2 = this->oh2;
    this->iw2 = this->ow2;

    int tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
    int tinput_size   = sizeof(Type) * A * A * this->oh2 * this->ow2
                                             * this->ic * this->n;
    int toutput_size  = sizeof(Type) * A * A * this->oh2 * this->ow2
                                             * this->oc * this->n;
    this->tweights = (Type *)malloc(tweights_size);
    this->tinput   = (Type *)malloc(tinput_size);
    //this->toutput  = (T *)malloc(toutput_size);

    if (this->input_fmt == nChw16c) {
        this->input_strides[0] = 1;
        this->input_strides[1] = V;
        this->input_strides[2] = V * this->iw;
        this->input_strides[3] = V * this->iw * this->ih;
        this->input_strides[4] = V * this->iw * this->ih * this->ic2;
    } else {
        // TODO
    }
    if (this->weights_fmt == OIhw16i16o) {
        this->weights_strides[0] = 1;
        this->weights_strides[1] = V;
        this->weights_strides[2] = V * V;
        this->weights_strides[3] = V * V * this->kw;
        this->weights_strides[4] = V * V * this->kw * this->kh;
        this->weights_strides[5] = V * V * this->kw * this->kh * this->ic2;
    } else {
        // TODO
    }
    if (this->output_fmt == nChw16c) {
        this->output_strides[0] = 1;
        this->output_strides[1] = V;
        this->output_strides[2] = V * this->ow;
        this->output_strides[3] = V * this->ow * this->oh;
        this->output_strides[4] = V * this->ow * this->oh * this->oc2;
    } else {
        // TODO
    }
}

template<typename T, const int A, const int K, const int V, const int I>
elx_conv_impl_t<T, A, K, V, I>::~elx_conv_impl_t ()
{
    if (this->tweights != nullptr) {
        free(this->tweights);
        this->tweights = nullptr;
    }
    if (this->tinput != nullptr) {
        free(this->tinput);
        this->tinput = nullptr;
    }
    if (this->toutput != nullptr) {
        free(this->toutput);
        this->toutput = nullptr;
    }
}

template<typename T, const int A, const int K, const int V, const int I> void
elx_conv_impl_t<T, A, K, V, I>::trans_weights(T *tweights, T *weights)
{
#pragma omp parallel for collapse(2)
    for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
        for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
            int d = 16 * 16 * (_ic2 + _oc2 * this->ic2);
            MD(T, aweights,  [K][K][V][V], weights + K * K * d);
            MD(T, atweights, [A][A][V][V], tweights + A * A * d);

            elk_trans_weights<T, A, K, V, I>(atweights, aweights);
        }
    }
}

template<typename T, const int A, const int K, const int V, const int I> void
elx_conv_impl_t<T, A, K, V, I>::trans_input(T *tinput, T *input)
{
    MD(T, atinput, [this->n][this->ic2][this->oh2][this->ow2][A][A][V], tinput);
    MD(T, ainput,  [this->n][this->ic2][this->ih][this->iw][V], input);

#pragma omp parallel for collapse(4)
    for (int _n = 0; _n < this->n; ++_n) {
        for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
            for (int _oh2 = 0; _oh2 < this->oh2; ++_oh2) {
                for (int _ow2 = 0; _ow2 < this->ow2; ++_ow2) {
                    elk_trans_input<T, A, K, V, I>
                        (*this, atinput[_n][_ic2][_oh2][_ow2],
                         (T *)ainput[_n][_ic2],
                         _oh2, _ow2);
                }}}}
}

template<typename T, const int A, const int K, const int V, const int I> void
elx_conv_impl_t<T, A, K, V, I>::product_trans_output(T *tinput, T *tweights, T *output)
{
    MD(T, atweights, [this->oc2][this->ic2][A][A][V][V], tweights);
    MD(T, atinput,   [this->n][this->ic2][this->oh2][this->ow2][A][A][V], tinput);
    MD(T, aoutput,   [this->n][this->oc2][this->oh][this->ow][V], output);

#pragma omp parallel for collapse(4)
    for (int _n = 0; _n < this->n; ++_n) {
        for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
            for (int _oh2 = 0; _oh2 < this->oh2; ++_oh2) {
                for (int _ow2 = 0; _ow2 < this->ow2; ++_ow2) {
                    elk_product_trans_output<T, A, K, V, I>
                        (*this,
                         (T *)atinput[_n],
                         (T *)atweights[_oc2],
                         (T *)aoutput[_n][_oc2],
                         _oh2, _ow2);
                }}}}
}

template<typename T, const int A, const int K, const int V, const int I> void
elx_conv_impl_t<T, A, K, V, I>::winograd(T *input, T *weights, T *output, T *bias)
{
    trans_weights(this->tweights, weights);
    trans_input(this->tinput, input);
    product_trans_output(this->tinput, this->tweights, output);
}

template<typename T>
int elx_conv(eld_conv_t<T> &desc, T *input, T *weights, T *output, T *bias)
{
    elx_conv_t<T> &xc = *desc.xc;

    // Sanity check
    if (any_null(input, weights, output)
        || (desc.with_bias && bias == nullptr)) {
        elx_error("Parameter error");
        return ELX_GENERAL_ERROR;
    }

    if (desc.algorithm == CONV_DIRECT) {
        elx_error("Unimplemented");
        return ELX_UNIMPLEMENTED;

    } else {
        assert(desc.algorithm == CONV_WINOGRAD);
        xc.winograd(input, weights, output, bias);
    }
    return ELX_OK;
}

template int elx_conv<float>
(eld_conv_t<float> &desc, float *input, float *weights, float *output, float *bias);

}
