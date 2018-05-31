#include <math.h>
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

  template <typename Type>
  int compare_conv_results_block16(eld_conv_t<Type> &, Type *, Type *)
  {
    return -1;
  }

  template <>
  int compare_conv_results_block16<float>(
      eld_conv_t<float> &desc, float *out, float *ref)
  {
    auto dims = desc.dims.output;
    using Array = float[dims.n][dims.c / 16][dims.h][dims.w][16];
    Array *aout = (Array *)out;
    Array *aref = (Array *)ref;

#define MAX_PRINT_ERRORS (20)
    int errors = 0;

#pragma omp parallel for collapse(3)
    for_each (_n, dims.n) {
      for_each (_C, dims.c / 16) {
        for_each (_h, dims.h) {
          for_each (_w, dims.w) {
            for_each (_v, 16) {
              double delta = fabs(
                  (*aout)[_n][_C][_h][_w][_v] - (*aref)[_n][_C][_h][_w][_v]);
              double rel_diff = delta / fabs((*aref)[_n][_C][_h][_w][_v]);
              if (rel_diff > 1e-6) {
                if (errors < MAX_PRINT_ERRORS) {
                  printf("Not equal!: [%d][%d][%d][%d][%d]: %f != %f (ref), "
                         "delta=%g, rel_diff=%g\n",
                      _n, _C, _h, _w, _v, (*aout)[_n][_C][_h][_w][_v],
                      (*aref)[_n][_C][_h][_w][_v], delta, rel_diff);
                }
                errors++;
              }
            }
          }
        }
      }
    }

    if (errors > 0) {
      printf("Error: number of errors: %d/%d, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
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

#pragma omp parallel for collapse(3)
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

#pragma omp parallel for collapse(3)
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

#pragma omp parallel for collapse(3)
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

#pragma omp parallel for collapse(3)
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
  int ref_convolution2d(eld_conv_t<Type> &desc, Type *output, Type *input,
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
    int sh = desc.strides.h;
    int sw = desc.strides.w;
    int pt = desc.pads.t;
    int pl = desc.pads.l;
    int dh = desc.dilations.h;
    int dw = desc.dilations.w;

    using Array1 = Type[n][ic][ih][iw];
    using Array2 = Type[oc][ic][kh][kw];
    using Array3 = Type[n][oc][oh][ow];

    Array1 *ainput = (Array1 *)input;
    Array2 *aweights = (Array2 *)weights;
    Array3 *aoutput = (Array3 *)output;

    if (desc.dims.input.n != desc.dims.output.n
        || desc.dims.input.c != desc.dims.weights.i
        || desc.dims.output.c != desc.dims.weights.o
        || desc.dims.output.c != desc.dims.bias.c) {
      printf("Dimension error!");
      return -1;
    }

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_oc, oc) {
        for_each (_oh, oh) {
          for_each (_ow, ow) {
            (*aoutput)[_n][_oc][_oh][_ow] = desc.with_bias ? bias[_oc] : 0.0f;
            for_each (_ic, ic) {
              for_each (_kh, kh) {
                int _ih = _oh * sh - pt + _kh * dh;
                if (_ih < 0 || _ih >= ih)
                  continue;
                for_each (_kw, kw) {
                  int _iw = _ow * sw - pl + _kw * dw;
                  if (_iw < 0 || _iw >= iw)
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
    return 0;
  }

  template <typename Type>
  int ref_convolution2d_block16(eld_conv_t<Type> &desc, Type *output,
      Type *input, Type *weights, Type *bias)
  {
    int n = desc.dims.input.n;
    int IC = desc.dims.input.c / 16;
    int OC = desc.dims.output.c / 16;
    int ih = desc.dims.input.h;
    int iw = desc.dims.input.w;
    int oh = desc.dims.output.h;
    int ow = desc.dims.output.w;
    int kh = desc.dims.weights.h;
    int kw = desc.dims.weights.w;
    int sh = desc.strides.h;
    int sw = desc.strides.w;
    int pt = desc.pads.t;
    int pl = desc.pads.l;
    int dh = desc.dilations.h;
    int dw = desc.dilations.w;

    using Array1 = Type[n][IC][ih][iw][16];
    using Array2 = Type[OC][IC][kh][kw][16][16];
    using Array3 = Type[n][OC][oh][ow][16];

    Array1 *ainput = (Array1 *)input;
    Array2 *aweights = (Array2 *)weights;
    Array3 *aoutput = (Array3 *)output;

    if (desc.dims.input.n != desc.dims.output.n
        || desc.dims.input.c != desc.dims.weights.i
        || desc.dims.output.c != desc.dims.weights.o
        || desc.dims.output.c != desc.dims.bias.c) {
      printf("Dimension error!");
      return -1;
    }

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_OC, OC) {
        for_each (_oh, oh) {
          for_each (_ow, ow) {
            for_each (_ov, 16) {
              (*aoutput)[_n][_OC][_oh][_ow][_ov]
                  = desc.with_bias ? bias[_OC * 16 + _ov] : 0.0f;
              for_each (_IC, IC) {
                for_each (_iv, 16) {
                  for_each (_kh, kh) {
                    int _ih = _oh * sh - pt + _kh * dh;
                    if (_ih < 0 || _ih >= ih)
                      continue;
                    for_each (_kw, kw) {
                      int _iw = _ow * sw - pl + _kw * dw;
                      if (_iw < 0 || _iw >= iw)
                        continue;
                      (*aoutput)[_n][_OC][_oh][_ow][_ov]
                          += (*ainput)[_n][_IC][_ih][_iw][_iv]
                          * (*aweights)[_OC][_IC][_kh][_kw][_iv][_ov];
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }

  template int ref_convolution2d<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);

  template int ref_convolution2d_block16<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);
}
}
