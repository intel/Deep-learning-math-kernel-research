#include <stdio.h>
#include <omp.h>

#ifndef __ELX_CONV_HPP__
namespace euler {


struct elx_conv_t {
    // plain
    int n, ih, iw, ic, oc, kh, kw, oh, ow;
    // blocks
    int IH, IW, IC, OC, OH, OW;
    // block unit, {vector, tile, out-tile}-size
    int V, T, To;
    int lpad, rpad, tpad, bpad;
    float *tr_weights;
};

}
#endif // __ELX_CONV_HPP__
