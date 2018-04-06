#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "euler.hpp"
#include "elt_utils.hpp"

using namespace euler;

int main() {
    // 1, create convolution desc
    eld_conv_t<float> desc;
    desc.dims = {
        .input   = { 1, 28, 28, 512 },
        .weights = { 3, 3, 512, 512 },
        .output  = { 1, 28, 28, 512 }
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

    desc.setup();

    // 2. allocate memory
    float *input, *weights, *output, *bias;
    input   = (float *)malloc(desc.byte_sizes.input);
    weights = (float *)malloc(desc.byte_sizes.weights);
    output  = (float *)malloc(desc.byte_sizes.output);

    for (int i = 0; i < desc.sizes.input; i++) {
        input[i] = i % 18;
    }
    for (int i = 0; i < desc.sizes.weights; i++) {
        weights[i] = i % 32;
    }

    // 3. execute convolution
    int iterations = 2000;
    time_start(conv);
    for (int n = 0; n < iterations; ++n) {
        if (ELX_OK != elx_conv<float>(desc, input, weights, output, bias)) {
            // error
            printf("Error\n");
        }
    }
    time_end(conv, iterations);
}
