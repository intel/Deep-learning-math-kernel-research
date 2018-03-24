#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "el_utils.hpp"
#include "elk_conv.hpp"

namespace euler {

template<typename T>
int eld_conv_setup(eld_conv_t &desc)
{
    elx_conv_t &x = desc.x;

    // Dimensions
    if (desc.dims.input.c != desc.dims.weights.i ||
        desc.dims.input.n  != desc.dims.output.n ||
        desc.dims.output.c != desc.dims.weights.o) {
        eld_error("Dimension error");
        return ELD_GENERAL_ERROR;
    }
    x.n  = desc.dims.input.n;
    x.ic = desc.dims.input.c;
    x.oc = desc.dims.output.c;
    x.ih = desc.dims.input.h;
    x.iw = desc.dims.input.w;
    x.oh = desc.dims.output.h;
    x.ow = desc.dims.output.w;
    x.kh = desc.dims.weights.h;
    x.kw = desc.dims.weights.w;
    // TODO:Check output dimensions

    // Formats
    // TODO:Check formats
    if (desc.formats.input != nChw16c ||
        desc.formats.output != nChw16c ||
        desc.formats.weights != OIhw16i16o) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }
    // TODO:Check CPUID
    x.v = 16; // avx512

    printf("desc.x.oh=%d\n", desc.x.oh);

    // Direct
    if (desc.algorithm == CONV_DIRECT) {
        eld_error("Unimplemented");
        return ELD_UNIMPLEMENTED;
    }

    // Winograd
    if (desc.algorithm == CONV_WINOGRAD) {
        if (desc.dilations.h > 1 ||
            desc.dilations.w > 1 ||
            desc.tile_size != 5 ||
            desc.strides.h != 1 ||
            desc.strides.w != 1 ||
            desc.dims.weights.h != 3 ||
            desc.dims.weights.w != 3) {
            eld_error("Unimplemented");
            return ELD_UNIMPLEMENTED;
        }
        x.t = desc.tile_size;
        x.ot = desc.tile_size - 2;

        x.twei = (float *)malloc(3 * 3 * 16 * 16 * 4);
    }

    // Winograd
    return ELD_OK;

}

template int eld_conv_setup<float>(eld_conv_t &desc);

}
