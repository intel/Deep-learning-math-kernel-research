#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"

#ifndef __ELX_CONV_HPP__
#define __ELX_CONV_HPP__

namespace euler {

template<typename F>
class elx_conv_t {
public:
    // dims
    int ic,  oc,  ih,  iw,  oh,  ow, n, t, kh, kw;
    // dims by block
    int ic2, oc2, ih2, iw2, oh2, ow2;
    // dims by double block
    int ic3, oc3, ih3, iw3, oh3, ow3;
    // blocking unit. {vector, tile, out-tile}-size
    int V, T, OT;
    // blocking unit. 2nd level
    int I2, O2, T2;
    // padding
    int lp, rp, tp, bp;
    // stride
    int hs, ws;
    // dilation
    int hd, wd;

    // formats
    int input_fmt;
    int weights_fmt;
    int output_fmt;

    // relu, bias
    bool with_relu, with_bias;
    // tensor strides
    int input_strides[8];
    int weights_strides[8];
    int output_strides[8];

    F *tweights;
    F *tinput;
    F *toutput;

    elx_conv_t(eld_conv_t<F> &dc);
    virtual ~elx_conv_t() {}

    virtual void direct(F *input, F *weights, F *output, F *bias) = 0;
    virtual void winograd(F *input, F *weights, F *output, F *bias) = 0;
};

template class elx_conv_t<float>;

template<typename F, const int T, const int K, const int V, const int I>
class elx_conv_impl_t : public elx_conv_t<F> {
public:
    elx_conv_impl_t(eld_conv_t<F> &dc);
    virtual ~elx_conv_impl_t();

    virtual void direct(F *input, F *weights, F *output, F *bias) {}
    virtual void winograd(F *input, F *weights, F *output, F *bias);

private:
    void trans_weights(F *tweights, F *weights);
    void trans_input(F *tinput, F *input);
    void product_trans_output(F *tinput, F *tweights, F *output);

};

template class elx_conv_impl_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_impl_t<float, 5, 3, 16, ISA_SKX_AVX512>;


}
#endif // __ELX_CONV_HPP__
