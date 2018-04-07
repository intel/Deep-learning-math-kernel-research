#include "el_def.hpp"

#ifndef __ELK_CONV_HPP__
#define __ELK_CONV_HPP__

namespace euler {

template<typename T, const int A, const int K, const int V, const int I> void
elk_trans_weights(T atweights[A][A][V][V], T aweights[K][K][V][V]);

template<typename T, const int A, const int K, const int V, const int I> void
elk_trans_input(elx_conv_t<T> &xc, T atinput[A][A][V], T *input, int _oh2, int _ow2);

template<typename T, const int A, const int K, const int V, const int I> void
elk_product_trans_output(elx_conv_t<T> &xc, T *tinput, T *tweights, T *output,
                         int _ih2, int _iw2);

}

#endif // __ELK_CONV_HPP__
