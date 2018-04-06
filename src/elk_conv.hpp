#include "el_def.hpp"

#ifndef __ELK_CONV_HPP__
#define __ELK_CONV_HPP__

namespace euler {

template<typename F, const int T, const int K, const int V, const int I> void
elk_trans_weights(F atweights[T][T][V][V], F aweights[K][K][V][V]);

template<typename F, const int T, const int K, const int V, const int I> void
elk_trans_input(elx_conv_t<F> &xc, F atinput[T][T][V], F *input, int _oh2, int _ow2);

template<typename F, const int T, const int K, const int V, const int I> void
elk_product_trans_output(elx_conv_t<F> &xc, F *tinput, F *tweights, F *output,
                         int _ih2, int _iw2);

}

#endif // __ELK_CONV_HPP__
