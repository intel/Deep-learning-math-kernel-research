#include "euler.hpp"
#include "el_def.hpp"
#include "el_utils.hpp"

#ifndef __ELX_CONV_WINO_GEMM_HPP__
#define __ELX_CONV_WINO_GEMM_HPP__

namespace euler {

template<typename T, const int A, const int K, const int V, const int I>
class elx_conv_wino_gemm_t : public elx_conv_t<T> {
public:
    elx_conv_wino_gemm_t(eld_conv_t<T> &dc);
    virtual ~elx_conv_wino_gemm_t();

    virtual void direct(T *input, T *weights, T *output, T *bias) {
        elx_error("Unimplemented");
    }
    virtual void winograd(T *input, T *weights, T *output, T *bias);

private:
    void trans_weights(T *tweights, T *weights);
    void trans_input(T *tinput, T *input);
    void product_trans_output(T *tinput, T *tweights, T *output);

};

template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_GENERIC>;
template class elx_conv_wino_gemm_t<float, 5, 3, 16, ISA_SKX_AVX512>;

}
#endif // __ELX_CONV_WINO_GEMM_HPP__
