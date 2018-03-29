#include <stdio.h>
#include <omp.h>

#ifndef __ELX_CONV_HPP__
namespace euler {


template<typename F> //, const int T, const int K>
class elx_conv_t {
public:
    // dimension length
    int ic,  oc,  ih,  iw,  oh,  ow, n, t, kh, kw;
    int ic2, oc2, ih2, iw2, oh2, ow2; // blocked
    int ic3, oc3, ih3, iw3, oh3, ow3; // 2nd level blocked
    // blocking unit, V=I=O
    int V,      T, OT; // {vector, tile, out-tile}-size
    int I2, O2, T2;

    int lpad, rpad, tpad, bpad;
    F *tweights;

/*
    template<typename F, const int T, const int K>
    using cb_trans_weights = void (*)(elx_conv_t&,
                                      F atweights[T][T][16][16],
                                      F aweights[K][K][16][16]);

    template<typename F, const int T, const int K>
    cb_trans_weights<F,T,K> *trans_weights;
*/
};

}
#endif // __ELX_CONV_HPP__
