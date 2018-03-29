#include <stdio.h>
#include <omp.h>

#ifndef __ELK_CONV_HPP__
namespace euler {

void elk_trans_weights_ref(float to[5][5][16][16], float from[3][3][16][16]);

template<typename F, const int T, const int K> void
elk_trans_weights(F atweights[T][T][16][16], F aweights[K][K][16][16]);

template<typename F, const int T, const int K> void
elk_trans_input(elx_conv_t<F> &xc, F atinput[T][T][16], F *input, int _oh2, int _ow2);

}
#endif // __ELK_CONV_HPP__
