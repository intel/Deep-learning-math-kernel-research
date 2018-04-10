#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"

#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

namespace euler {

template<typename Type, const int A, const int K, const int T, const int V, const int I>
class elx_conv_wino_gemm_t : public elx_conv_t<Type> {
public:
    elx_conv_wino_gemm_t(eld_conv_t<Type> &dc);
    virtual ~elx_conv_wino_gemm_t();

    virtual void direct(Type *input, Type *weights, Type *output, Type *bias) {
        elx_error("Unimplemented");
    }
    virtual void winograd(Type *input, Type *weights, Type *output, Type *bias);

private:
    void trans_weights(Type *tweights, Type *weights);
    void trans_input  (Type *tinput,   Type *input);
    void trans_output (Type *output,   Type *toutput);
    void gemm         (Type *toutput,  Type *tinput, Type *tweights);

};

template class elx_conv_wino_gemm_t<float, 5, 3, 25, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 25, 16, ISA_SKX_AVX512>;

}
#endif // __ELX_CONV_WINO_GEMM_HPP__
