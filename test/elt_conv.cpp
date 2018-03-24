#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "euler.hpp"

using namespace euler;

int main() {
    // 1, create convolution desc
    eld_conv_t desc;
    desc.dims = {
        .input   = { 4, 224, 224, 1024 },
        .weights = { 3, 3, 1024, 512 },
        .output  = { 4, 224, 224, 512 }
    };
    desc.formats = {
        .input   = nChw16c,
        .weights = OIhw16i16o,
        .output  = nChw16c
    };
    desc.pads      = { 1, 1, 1, 1 };
    desc.with_bias = false;
    desc.algorithm = CONV_WINOGRAD;
    desc.tile_size = 5;
    desc.with_relu = false;

    desc.setup<float>();

    // 2. allocate memory
    float *input, *weights, *output, *bias;
    input = (float *)malloc(desc.sizes.input);
    weights = (float *)malloc(desc.sizes.weights);
    output = (float *)malloc(desc.sizes.output);

    for (int i = 0; i < desc.sizes.weights / 4; i++) {
        weights[i] = i % 32;
    }

    // 3. execute convolution
    for (int n = 0; n < 1; ++n) {
        if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
            // error
            printf("Error\n");
        }
    }
}
