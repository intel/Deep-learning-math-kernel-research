#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"

#ifndef __ELX_CONV_HPP__
#define __ELX_CONV_HPP__

namespace euler {

template<typename Type>
class elx_conv_t {
public:
    // dims
    int ic,  oc,  ih,  iw,  oh,  ow, n, t, kh, kw;
    // dims by block
    int ic2, oc2, ih2, iw2, oh2, ow2, t2;
    // dims by double block
    int ic3, oc3, ih3, iw3, oh3, ow3, t3;
    // blocking unit: vector-size, tile-size, tile-blocking-size
    int V, A, T;
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

    Type *tweights;
    Type *tinput;
    Type *toutput;

    elx_conv_t(eld_conv_t<Type> &dc);
    virtual ~elx_conv_t() {}

    virtual void direct(Type *input, Type *weights, Type *output, Type *bias) = 0;
    virtual void winograd(Type *input, Type *weights, Type *output, Type *bias) = 0;
};

template class elx_conv_t<float>;

}
#endif // __ELX_CONV_HPP__
