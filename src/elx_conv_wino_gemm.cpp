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
    this->oh2 = (this->oh + A - 3) / (A - 2);
    this->ow2 = (this->ow + A - 3) / (A - 2);
    this->ih2 = this->oh2;
    this->iw2 = this->ow2;

    int tweights_size = sizeof(Type) * A * A * this->ic * this->oc;
    int tinput_size   = sizeof(Type) * A * A * this->oh2 * this->ow2
                                             * this->ic * this->n;
    //int toutput_size  = sizeof(Type) * A * A * this->oh2 * this->ow2
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

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::trans_weights(Type *tweights, Type *weights)
{
    // oc2, ic2, K, K, V, V => oc3, ic3, A, A, O2, I2, V, V
    mdarray<Type, 6> aweights(weights, this->oc2, this->ic2, K, K, V, V);
    mdarray<Type, 8> atweights(tweights, this->oc3, this->ic3, A, A, this->O2, this->I2, V, V);

#pragma omp parallel for collapse(4)
    for (int _oc3 = 0; _oc3 < this->oc3; ++_oc3) {
        for (int _ic3 = 0; _ic3 < this->ic3; ++_ic3) {
            for (int _O2 = 0; _O2 < this->O2; ++_O2) {
                for (int _I2 =  0; _I2 < this->I2; ++_I2) {

                    Type aout[A][A][V][V];
                    Type in = aweights(_oc3 * this->O2, _ic3 * this->I2, 0, 0, 0, 0);
                    MD(Type, ain, [K][K][V][V], &in);
                    elk_trans_weights<Type, A, K, V, I>(aout, ain);

                    for (int _hA = 0; _hA < A; ++_hA) {
                        for (int _wA = 0; _wA < A; ++_wA) {
                            for (int _iV = 0; _iV < V; ++_iV) {
#pragma omp simd
                                for (int _oV = 0; _oV < V; ++_oV) {
                                    atweights(_oc3, _ic3, _hA, _wA, _O2, _I2, _iV, _oV) =
                                        aout[_hA][_wA][_iV][_oV];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::trans_input(Type *tinput, Type *input)
{
    MD(Type, atinput, [this->n][this->ic2][this->oh2][this->ow2][A][A][V], tinput);
    MD(Type, ainput,  [this->n][this->ic2][this->ih][this->iw][V], input);

#pragma omp parallel for collapse(4)
    for (int _n = 0; _n < this->n; ++_n) {
        for (int _ic2 = 0; _ic2 < this->ic2; ++_ic2) {
            for (int _oh2 = 0; _oh2 < this->oh2; ++_oh2) {
                for (int _ow2 = 0; _ow2 < this->ow2; ++_ow2) {
                    elk_trans_input<Type, A, K, V, I>
                        (*this, atinput[_n][_ic2][_oh2][_ow2],
                         (Type *)ainput[_n][_ic2],
                         _oh2, _ow2);
                }
            }
        }
    }
}

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::gemm(Type *tinput, Type *tweights, Type *output)
{
    MD(Type, atweights, [this->oc2][this->ic2][A][A][V][V], tweights);
    MD(Type, atinput,   [this->n][this->ic2][this->oh2][this->ow2][A][A][V], tinput);
    MD(Type, aoutput,   [this->n][this->oc2][this->oh][this->ow][V], output);

#pragma omp parallel for collapse(4)
    for (int _n = 0; _n < this->n; ++_n) {
        for (int _oc2 = 0; _oc2 < this->oc2; ++_oc2) {
            for (int _oh2 = 0; _oh2 < this->oh2; ++_oh2) {
                for (int _ow2 = 0; _ow2 < this->ow2; ++_ow2) {
                    elk_product_trans_output<Type, A, K, V, I>
                        (*this,
                         (Type *)atinput[_n],
                         (Type *)atweights[_oc2],
                         (Type *)aoutput[_n][_oc2],
                         _oh2, _ow2);
                }
            }
        }
    }
}

template<typename Type, const int A, const int K, const int T, const int V, const int I> void
elx_conv_wino_gemm_t<Type, A, K, T, V, I>::winograd(Type *input, Type *weights, Type *output, Type *bias)
{
    // TODO: support bias
    if (bias != nullptr)
        return;
    trans_weights(this->tweights, weights);
    trans_input(this->tinput, input);
    gemm(this->tinput, this->tweights, output);
}

}
