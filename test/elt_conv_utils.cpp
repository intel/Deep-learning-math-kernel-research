#include "elt_conv_utils.hpp"

namespace euler {
namespace test {

  template <typename Type>
  void prepare_conv_data(
      eld_conv_t<Type> &, Type **, Type **, Type **, Type **)
  {
  }

  template <>
  void prepare_conv_data<float>(eld_conv_t<float> &desc, float **input,
      float **weights, float **output, float **bias)
  {
    *input = (float *)memalign(64, desc.byte_sizes.input);
    *weights = (float *)memalign(64, desc.byte_sizes.weights);
    *output = (float *)memalign(64, desc.byte_sizes.output);
    *bias = (float *)memalign(64, desc.byte_sizes.bias);

#pragma omp parallel for
    for (int i = 0; i < desc.sizes.input; i++) {
      (*input)[i] = i % 18;
    }
#pragma omp parallel for
    for (int i = 0; i < desc.sizes.weights; i++) {
      (*weights)[i] = i % 32;
    }
#pragma omp parallel for
    for (int i = 0; i < desc.sizes.bias; i++) {
      (*bias)[i] = i % 13;
    }
  }

  void teardown_conv_data(
      void *input, void *weights, void *output, void *bias)
  {
    if (input)
      free(input);
    if (weights)
      free(weights);
    if (output)
      free(output);
    if (bias)
      free(bias);
  }

  size_t cal_ops(eld_conv_t<float> &desc)
  {
    size_t num_ops = 0;

    for_each (_oh, desc.dims.output.h) {
      for_each (_ow, desc.dims.output.w) {
        for_each (_kh, desc.dims.weights.h) {
          int _ih
              = _oh * desc.strides.h - desc.pads.t + _kh * desc.dilations.h;
          if (_ih < 0 || _ih >= desc.dims.input.h)
            continue;
          for_each (_kw, desc.dims.weights.w) {
            int _iw
                = _ow * desc.strides.w - desc.pads.l + _kw * desc.dilations.w;
            if (_iw < 0 || _iw >= desc.dims.input.w)
              continue;
            num_ops += 1;
          }
        }
      }
    }
    return num_ops * desc.dims.input.n * desc.dims.weights.i
        * desc.dims.weights.o * 2;
  }

  template <typename Type>
  reorder<Type, nchw, nChw16c>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {

    using Array1 = Type[n][c / 16][h][w][16];
    using Array2 = Type[n][c][h][w];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_C, c / 16) {
        for_each (_h, h) {
          for_each (_w, w) {
            for_each (_v, 16) {
              (*adst)[_n][_C * 16 + _v][_h][_w] = (*asrc)[_n][_C][_h][_w][_v];
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, nChw16c, nchw>::reorder(
      Type *dst, Type *src, int n, int c, int h, int w)
  {
    using Array1 = Type[n][c][h][w];
    using Array2 = Type[n][c / 16][h][w][16];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_C, c / 16) {
        for_each (_h, h) {
          for_each (_w, w) {
            for_each (_v, 16) {
              (*adst)[_n][_C][_h][_w][_v] = (*asrc)[_n][_C * 16 + _v][_h][_w];
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, OIhw16i16o, oihw>::reorder(
      Type *dst, Type *src, int o, int i, int h, int w)
  {
    using Array1 = Type[o][i][h][w];
    using Array2 = Type[o / 16][i / 16][h][w][16][16];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(4)
    for_each (_O, o / 16) {
      for_each (_I, i / 16) {
        for_each (_h, h) {
          for_each (_w, w) {
            for_each (_iv, 16) {
              for_each (_ov, 16) {
                (*adst)[_O][_I][_h][_w][_iv][_ov]
                    = (*asrc)[_O * 16 + _ov][_I * 16 + _iv][_h][_w];
              }
            }
          }
        }
      }
    }
  }

  template <typename Type>
  reorder<Type, oihw, OIhw16i16o>::reorder(
      Type *dst, Type *src, int o, int i, int h, int w)
  {
    using Array1 = Type[o / 16][i / 16][h][w][16][16];
    using Array2 = Type[o][i][h][w];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(4)
    for_each (_O, o / 16) {
      for_each (_I, i / 16) {
        for_each (_h, h) {
          for_each (_w, w) {
            for_each (_iv, 16) {
              for_each (_ov, 16) {
                (*adst)[_O * 16 + _ov][_I * 16 + _iv][_h][_w]
                    = (*asrc)[_O][_I][_h][_w][_iv][_ov];
              }
            }
          }
        }
      }
    }
  }

  template <typename Type>
  void ref_convolution2d(eld_conv_t<Type> &desc, Type *output, Type *input,
      Type *weights, Type *bias)
  {
    int n = desc.dims.input.n;
    int ic = desc.dims.input.c;
    int oc = desc.dims.output.c;
    int ih = desc.dims.input.h;
    int iw = desc.dims.input.w;
    int oh = desc.dims.output.h;
    int ow = desc.dims.output.w;
    int kh = desc.dims.weights.h;
    int kw = desc.dims.weights.w;

    using Array1 = Type[n][ic][ih][iw];
    using Array2 = Type[oc][ic][kh][kw];
    using Array3 = Type[n][oc][oh][ow];

    Array1 *ainput = (Array1 *)input;
    Array2 *aweights = (Array2 *)weights;
    Array3 *aoutput = (Array3 *)output;

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_oc, oc) {
        for_each (_oh, oh) {
          for_each (_ow, ow) {
            (*aoutput)[_n][_oc][_oh][_ow] = desc.with_bias ? bias[_oc] : 0.0f;
            for_each (_ic, ic) {
              for_each (_kh, kh) {
                int _ih = _oh * desc.strides.h - desc.pads.t
                    + _kh * desc.dilations.h;
                if (_ih < 0 || _ih > ih)
                  continue;

                for_each (_kw, kw) {
                  int _iw = _ow * desc.strides.w - desc.pads.l
                      + _kw * desc.dilations.w;
                  if (_iw < 0 || _iw > iw)
                    continue;

                  (*aoutput)[_n][_oc][_oh][_ow]
                      += (*ainput)[_n][_ic][_ih][_iw]
                      * (*aweights)[_oc][_ic][_kh][_kw];
                }
              }
            }
          }
        }
      }
    }
  }

  template <typename Type>
  void ref_convolution2d_block16(eld_conv_t<Type> &desc, Type *output,
      Type *input, Type *weights, Type *bias)
  {
    int n = desc.dims.input.n;
    int ic = desc.dims.input.c;
    int oc = desc.dims.output.c;
    int ih = desc.dims.input.h;
    int iw = desc.dims.input.w;
    int oh = desc.dims.output.h;
    int ow = desc.dims.output.w;
    int kh = desc.dims.weights.h;
    int kw = desc.dims.weights.w;

    Type *in = (Type *)malloc(sizeof(Type) * n * ic * ih * iw);
    Type *out = (Type *)malloc(sizeof(Type) * n * oc * oh * ow);
    Type *wei = (Type *)malloc(sizeof(Type) * oc * ic * kh * kw);

    reorder<Type, nchw, nChw16c>(in, input, n, ic, ih, iw);
    reorder<Type, oihw, OIhw16i16o>(wei, weights, oc, ic, kh, kw);

    ref_convolution2d(desc, out, in, wei, bias);

    reorder<Type, nChw16c, nchw>(output, out, n, oc, oh, ow);

    free(in);
    free(out);
    free(wei);
  }

  template void ref_convolution2d<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);

  template void ref_convolution2d_block16<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);
}
}
