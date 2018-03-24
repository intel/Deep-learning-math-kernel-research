#include <stdio.h>
#include <assert.h>
#include "euler.hpp"

int main() {
    using namespace euler;
    // 1. allocate memory
    float *input, *weights, *output, *bias;
    input = (float *)1;
    weights = (float *)1;
    output = (float *)1;
    // allocate memory buffers

    // 2, create convolution desc
    // Or in C++11
    eld_conv_t desc;
    desc.dims = {
        .input   = { 64, 224, 224, 3 },
        .weights = { 3, 3, 3, 128 },
        .output  = { 64, 224, 224, 128 }
    };
    desc.formats = {
        .input   = nChw16c,
        .weights = OIhw16i16o,
        .output  = nChw16c
    },
    desc.pads      = { 1, 1, 1, 1 };
    desc.with_bias = false;
    desc.algorithm = CONV_WINOGRAD;
    desc.tile_size = 5;
    desc.with_relu = false;
    eld_conv_setup<float>(desc);

    // 3. execute convolution
    if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
        // error
        printf("Error\n");
    }
}
