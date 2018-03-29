#include <stdio.h>
#include <omp.h>

#ifndef __ELX_CONV_HPP__
namespace euler {


template<typename F>
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

    virtual void trans_weights(elx_conv_t<F>&) = 0;
                          //F atweights[T][T][16][16],
                          //F aweights[K][K][16][16]) = 0;
};

template<typename F, const int T, const int K>
class elk_conv_t : public elx_conv_t<F> {
public:
    virtual void trans_weights(elx_conv_t<F>&)
                          //F atweights[T][T][16][16],
                          //F aweights[K][K][16][16])
    { printf("................in elk_conv_t\n"); }

};

template void elk_conv_t<float, 5, 3>::
trans_weights(elx_conv_t<float>&);
              //float atweights[5][5][16][16],
              //float aweights[3][3][16][16]);

}
#endif // __ELX_CONV_HPP__
