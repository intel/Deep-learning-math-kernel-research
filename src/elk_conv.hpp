#include <stdio.h>
#include <omp.h>

#ifndef __ELK_CONV_HPP__
namespace euler {

void elk_trans_weights(float to[5][5][16][16], float from[3][3][16][16]);
void elk_trans_weights_ref(float to[5][5][16][16], float from[3][3][16][16]);

template<typename F, const int T> int
elk_trans_input(elx_conv_t &xc, F tr_input[T][T][16], F *input, int oh, int ow);

}
#endif // __ELK_CONV_HPP__
