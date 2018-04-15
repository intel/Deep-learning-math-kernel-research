#include <assert.h>
#include <stdlib.h>
#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"
#include "elx_conv.hpp"
#include "elk_conv.hpp"
#include "elx_conv_wino_gemm.hpp"

namespace euler {

template<typename Type, const int A, const int K, const int T, const int V, const int I>
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::elx_conv_wino_gemm_t (eld_conv_t<Type> &dc)
: elx_conv_t<Type>(dc) 
{
    this->V = V;
    this->ic2 = this->ic / V;
    this->oc2 = this->oc / V;

    this->T = T;
    this->O2 = 4; // TODO: O2 selection
    this->I2 = 2; // TODO: I2 selection

    this->A   = A;
    this->ht = (this->oh + A - 3) / (A - K + 1);
    this->wt = (this->ow + A - 3) / (A - K + 1);

    int tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
    int tinput_size   = sizeof(Type) * A * A * this->ht * this->wt
                                             * this->ic * this->n;
    //int toutput_size  = sizeof(Type) * A * A * this->ht * this->wt
    //                                         * this->oc * this->n;
    this->tweights = (Type *)malloc(tweights_size);
    this->tinput   = (Type *)malloc(tinput_size);
    //this->toutput  = (Type *)malloc(toutput_size);

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

template<typename Type, const int A, const int K, const int T, const int V, const int I>
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::~elx_conv_wino_gemm_t ()
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

#pragma optimization_parameter target_arch=CORE-AVX512
template<typename Type, const int A, const int K, const int T, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, T, V, I>::
trans_weights(Type *tweights, Type *weights)
{
    // oc2, ic2, K, K, V, V => oc3, ic3, A, A, O2, I2, V, V
    mdarray<Type, 6> aweights(weights, this->oc2, this->ic2, K, K, V, V);
    mdarray<Type, 8> atweights(tweights, this->oc3, this->ic3, A, A, this->O2,
                               this->I2, V, V);
#pragma omp parallel for collapse(4)
    for_each(_oc3, this->oc3) {
    for_each(_ic3, this->ic3) {
    for_each(_O2,  this->O2)  {
    for_each(_I2,  this->I2)  {
        Type aout[A][A][V][V];
        Type in = aweights(_oc3 * this->O2 + _O2, _ic3 * this->I2 + _I2,
                           0, 0, 0, 0);
        MD(Type, ain, [K][K][V][V], &in);
        elk_trans_weights<Type, A, K, V, I>(aout, ain);

        for_each(_hA, A) {
        for_each(_wA, A) {
        for_each(_iV, V) {
#pragma omp simd
        for_each(_oV, V) {
            atweights(_oc3, _ic3, _hA, _wA, _O2, _I2, _iV, _oV) =
                aout[_hA][_wA][_iV][_oV];
        }}}}
    }}}}
}

#pragma optimization_parameter target_arch=CORE-AVX512
template<typename Type, const int A, const int K, const int T, const int V, const int I>
void elx_conv_wino_gemm_t<Type, A, K, T, V, I>::
trans_input(Type *tinput, Type *input, int _t2)
{
    // n, ic2, ih, iw, V => t2=1, A, A, ic3, I2, T, V
    mdarray<Type, 5> ainput(input, this->n, this->ic2, this->ih, this->iw, V);
    mdarray<Type, 7> atinput(tinput, this->t2, A, A, this->ic3, this->I2, T, V);

    Type aout[A][A][V];

    for_each(_ic3, this->ic3) {
    for_each(_I2,  this->I2)  {
    for_each(_T,   T)         {
        int _t = _t2 * this->T + _T;
        int _nt = _t % this->nt;
        int _ht = _nt / this->wt;
        int _wt = _nt % this->wt;
        Type *in = &ainput(_t / this->nt,
                           _ic3 * this->I2 + _I2,
                           _ht * (A - K + 1) - this->lp,
                           _wt % (A - K + 1) - this->tp, 0);
        bool margin = (_ht == 0 || _ht == this->ht - 1 ||
                       _wt == 0 || _wt == this->wt - 1);
        elk_trans_input<Type, A, K, V, I>(*this, aout, in, margin);

        for_each(_hA, A) {
        for_each(_wA, A) {
#pragma omp simd
        for_each(_V,  V) {
            atinput(_t2, _hA, _wA, _ic3, _I2, _T, _V) =
                aout[_hA][_wA][_V];
        }}}
    }}}
}

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::
gemm(Type *tinput, Type *tweights, Type *toutput)
{
    mdarray<Type, 8> atweights(tweights, this->oc3, this->ic3,
                                         A, A, this->O2,  this->I2, V, V);
    mdarray<Type, 6> atinput  (tinput,   A, A, this->ic3, this->I2, T, V);
    mdarray<Type, 6> atoutput (toutput,  A, A, this->oc3, this->O2, T, V);

    for_each(_hA, A) {
    for_each(_wA, A) {
    for_each(_oc3, this->oc3) {
    for_each(_ic3, this->ic3) {
        // TODO: _ic3 == 0, zeros
        elk_gemm<Type, T, V, I>(*this,
                                &atoutput (_hA, _wA, _oc3, 0, 0, 0),
                                &atinput  (_hA, _wA, _ic3, 0, 0, 0),
                                &atweights(_oc3, _ic3, _hA, _wA, 0, 0, 0, 0));
    }}}}
}

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::
trans_output(Type *output, Type *toutput, int _t2)
{
    // A, A, oc3, O2, T, V -> n, oc2, oh, ow, V
    mdarray<Type, 6> atoutput(toutput, A, A, this->oc3, this->O2, T, V);
    mdarray<Type, 5> aoutput(output, this->n, this->oc2, this->oh, this->ow, V);

    Type ain[A][A][V];

    for_each(_oc3, this->oc3) {
    for_each(_O2,  this->O2)  {
    for_each(_T,   T) {
        for_each(_hA, A) {
        for_each(_wA, A) {
#pragma omp simd
        for_each(_V,  V) {
            ain[_hA][_wA][_V] = atoutput(_hA, _wA, _oc3, _O2, _T, _V);
        }}}

        int _t = _t2 * this->T + _T;
        int _nt = _t % this->nt;
        int _ht = _nt / this->wt;
        int _wt = _nt % this->wt;
        Type *out = &aoutput(_t / this->nt,
                             _oc3 * this->O2 + _O2,
                             _ht * (A - K + 1),
                             _wt % (A - K + 1), 0);
        bool margin = (_ht == this->ht - 1 || _wt == this->wt - 1);
        elk_trans_output<Type, A, K, V, I>(*this, out, ain, margin);
    }}}
}

// tweights: oc3, ic3, A, A, O2, I2, V, V
// tinputs:  t2, A, A, ic3, I2, T, V
// toutput:  t2, A, A, oc3, O2, T, V
template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::
winograd(Type *input, Type *weights, Type *output, Type *bias)
{
    // TODO: support bias
    if (bias != nullptr)
        return;
    trans_weights(this->tweights, weights);

#pragma omp parallel for
    for_each(_t2, this->t2) {
        trans_input(this->tinput, input, _t2);
        gemm(this->tinput, this->tweights, this->toutput);
        trans_output(output, this->toutput, _t2);
    }
}

}
