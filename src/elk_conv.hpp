#include <stdio.h>
#include <omp.h>

#ifndef __ELK_CONV_HPP__
namespace euler {

__declspec(cpu_dispatch(generic, skylake_avx512))
void elk_trans_weights(float to[5][5][16][16], float from[3][3][16][16]) { }
void elk_trans_weights_ref(float to[5][5][16][16], float from[3][3][16][16]);

}
#endif // __ELK_CONV_HPP__
