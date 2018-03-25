#include <stdio.h>
#include <omp.h>

#ifndef __ELX_CONV_HPP__
namespace euler {


struct elx_conv_t {
    int n, ih, iw, ic, oc, kh, kw, oh, ow, IC, OC;
    int v, t, ot; // {vector, tile, out-tile}-size
    int lpad, rpad, tpad, bpad;
    float *tr_weights;
};

}
#endif // __ELX_CONV_HPP__
