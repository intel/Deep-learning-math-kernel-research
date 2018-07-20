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
    MEMALIGN64(input, desc.byte_sizes.input);
    MEMALIGN64(weights, desc.byte_sizes.weights);
    MEMALIGN64(output, desc.byte_sizes.output);
    MEMALIGN64(bias, desc.byte_sizes.bias);

#pragma omp parallel for
    for (size_t i = 0; i < desc.sizes.input; i++) {
      (*input)[i] = i % 15;
    }
#pragma omp parallel for
    for (size_t i = 0; i < desc.sizes.weights; i++) {
      (*weights)[i] = i % 31;
    }
#pragma omp parallel for
    for (size_t i = 0; i < desc.sizes.bias; i++) {
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
  int __compare_conv_results_plain(eld_conv_t<Type> &, Type *, Type *)
  {
    return -1;
  }

  template <typename Type>
  int __compare_conv_results_blocked(eld_conv_t<Type> &, Type *, Type *)
  {
    return -1;
  }

  template <typename Type>
  int compare_conv_results(eld_conv_t<Type> &desc, Type *out, Type *ref)
  {
    if (desc.formats.output == nchw)
      return __compare_conv_results_plain(desc, out, ref);
    else
      return __compare_conv_results_blocked(desc, out, ref);
  }

  template <>
  int __compare_conv_results_blocked<float>(
      eld_conv_t<float> &desc, float *out, float *ref)
  {
    const int V = 16;
    auto dims = desc.dims.output;
    int C = ALIGNUP(dims.c, V) / V;
    int Or = dims.c % V ? dims.c % V: V;

    using Array = float[dims.n][C][dims.h][dims.w][V];
    Array *aout = (Array *)out;
    Array *aref = (Array *)ref;

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;

#pragma omp parallel for collapse(3)
    for_each (_n, dims.n) {
      for_each (_C, C) {
        for_each (_h, dims.h) {
          for_each (_w, dims.w) {
            int v = _C == C - 1 ? Or : V;
            for_each (_v, v) {
              double delta = fabs(
                  (*aout)[_n][_C][_h][_w][_v] - (*aref)[_n][_C][_h][_w][_v]);
              double rel_diff = delta / fabs((*aref)[_n][_C][_h][_w][_v]);
              if (rel_diff > 5e-6) {
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
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
          desc.sizes.output, ((errors * 1.0) / desc.sizes.output) * 100.0);
      return -1;
    }
    return 0;
  }

  template <>
  int __compare_conv_results_plain<float>(
      eld_conv_t<float> &desc, float *out, float *ref)
  {
    auto dims = desc.dims.output;
    using Array = float[dims.n][dims.c][dims.h][dims.w];
    Array *aout = (Array *)out;
    Array *aref = (Array *)ref;

#define MAX_PRINT_ERRORS (20)
    size_t errors = 0;

#pragma omp parallel for collapse(3)
    for_each (_n, dims.n) {
      for_each (_c, dims.c) {
        for_each (_h, dims.h) {
          for_each (_w, dims.w) {
            double delta = fabs(
                (*aout)[_n][_c][_h][_w] - (*aref)[_n][_c][_h][_w]);
            double rel_diff = delta / fabs((*aref)[_n][_c][_h][_w]);
            if (rel_diff > 5e-6) {
              if (errors < MAX_PRINT_ERRORS) {
                printf("Not equal!: [%d][%d][%d][%d]: %f != %f (ref), "
                       "delta=%g, rel_diff=%g\n",
                    _n, _c, _h, _w, (*aout)[_n][_c][_h][_w],
                    (*aref)[_n][_c][_h][_w], delta, rel_diff);
              }
              errors++;
            }
          }
        }
      }
    }

    if (errors > 0) {
      printf("Error: number of errors: %ld/%ld, percentage: %f%%\n", errors,
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
    int C = ALIGNUP(c, 16) / 16; // padding
    int Vr = c % 16 ? c % 16 : 16;

    using Array1 = Type[n][C][h][w][16];
    using Array2 = Type[n][c][h][w];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(3)
    for_each (_n, n) {
      for_each (_C, C) {
        for_each (_h, h) {
          for_each (_w, w) {
            int v = (_C == C - 1) ? Vr : 16;
            for_each (_v, v) {
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
    int C = ALIGNUP(c, 16) / 16; // padding
    int Vr = c % 16 ? c % 16 : 16;

    using Array1 = Type[n][c][h][w];
    using Array2 = Type[n][C][h][w][16];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(3)
    for_each (_n, n) {
      for_each (_C, C) {
        for_each (_h, h) {
          for_each (_w, w) {
            int v = (_C == C - 1) ? Vr : 16;
            for_each (_v, 16) {
              if (_v < v)
                (*adst)[_n][_C][_h][_w][_v]
                    = (*asrc)[_n][_C * 16 + _v][_h][_w];
              else
                (*adst)[_n][_C][_h][_w][_v] = 0;
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
    int O = ALIGNUP(o, 16) / 16; // padding
    int I = ALIGNUP(i, 16) / 16; // padding
    int Or = o % 16 ? o % 16 : 16;
    int Ir = i % 16 ? i % 16 : 16;

    using Array1 = Type[o][i][h][w];
    using Array2 = Type[O][I][h][w][16][16];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(3)
    for_each (_O, O) {
      for_each (_I, I) {
        for_each (_h, h) {
          for_each (_w, w) {
            int ov = (_O == O - 1) ? Or : 16;
            int iv = (_I == I - 1) ? Ir : 16;
            for_each (_iv, 16) {
              for_each (_ov, 16) {
                if (_iv < iv && _ov < ov)
                  (*adst)[_O][_I][_h][_w][_iv][_ov]
                      = (*asrc)[_O * 16 + _ov][_I * 16 + _iv][_h][_w];
                else
                  (*adst)[_O][_I][_h][_w][_iv][_ov] = 0;
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
    int O = ALIGNUP(o, 16) / 16; // padding
    int I = ALIGNUP(i, 16) / 16; // padding
    int Or = o % 16 ? o % 16 : 16;
    int Ir = i % 16 ? i % 16 : 16;

    using Array1 = Type[O][I][h][w][16][16];
    using Array2 = Type[o][i][h][w];
    Array1 *asrc = (Array1 *)src;
    Array2 *adst = (Array2 *)dst;

#pragma omp parallel for collapse(3)
    for_each (_O, O) {
      for_each (_I, I) {
        for_each (_h, h) {
          for_each (_w, w) {
            int ov = _O == O - 1 ? Or : 16;
            int iv = _I == I - 1 ? Ir : 16;
            for_each (_iv, iv) {
              for_each (_ov, ov) {
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

    if (desc.dims.input.n != desc.dims.output.n
        || desc.dims.input.c != desc.dims.weights.i
        || desc.dims.output.c != desc.dims.weights.o
        || desc.dims.output.c != desc.dims.bias.c) {
      printf("Dimension error!");
      return -1;
    }

    Type *tinput = nullptr, *tweights = nullptr, *toutput = nullptr;
    if (desc.formats.input == nChw16c) {
      tinput = (Type *)malloc(desc.byte_sizes.input);
      reorder<Type, nchw, nChw16c>(tinput, input, n, ic, ih, iw);
    }
    if (desc.formats.weights == OIhw16i16o) {
      tweights = (Type *)malloc(desc.byte_sizes.weights);
      reorder<Type, oihw, OIhw16i16o>(tweights, weights, oc, ic, kh, kw);
    }
    if (desc.formats.output == nChw16c) {
      toutput = (Type *)malloc(desc.byte_sizes.output);
    }

    using Array1 = Type[n][ic][ih][iw];
    using Array2 = Type[oc][ic][kh][kw];
    using Array3 = Type[n][oc][oh][ow];

    Array1 *ainput
        = desc.formats.input == nchw ? (Array1 *)input : (Array1 *)tinput;
    Array2 *aweights
        = desc.formats.weights == oihw ? (Array2 *)weights : (Array2 *)tweights;
    Array3 *aoutput
        = desc.formats.output == nchw ? (Array3 *)output : (Array3 *)toutput;

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

    if (desc.formats.output == nChw16c)
      reorder<Type, nChw16c, nchw>(output, toutput, n, oc, oh, ow);

    if (tinput != nullptr)
      free(tinput);
    if (tweights != nullptr)
      free(tweights);
    if (toutput != nullptr)
      free(toutput);

    return 0;
  }

  template <typename Type>
  int ref_convolution2d_block16(eld_conv_t<Type> &desc, Type *output,
      Type *input, Type *weights, Type *bias)
  {
    int n = desc.dims.input.n;
    int IC = ALIGNUP(desc.dims.input.c, 16) / 16;
    int OC = ALIGNUP(desc.dims.output.c, 16) / 16;
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

    if (desc.formats.input != nChw16c || desc.formats.weights != OIhw16i16o
        || desc.formats.output != nChw16c) {
      printf("Format error!");
      return -1;
    }
    int Or = desc.dims.output.c % 16 ?  desc.dims.output.c % 16 : 16;
    int Ir = desc.dims.input.c % 16 ? desc.dims.input.c % 16 : 16;

#pragma omp parallel for collapse(4)
    for_each (_n, n) {
      for_each (_OC, OC) {
        for_each (_oh, oh) {
          for_each (_ow, ow) {
            int ov = _OC == OC - 1 ? Or : 16;
            for_each (_ov, ov) {
              (*aoutput)[_n][_OC][_oh][_ow][_ov]
                  = desc.with_bias ? bias[_OC * 16 + _ov] : 0.0f;
              for_each (_IC, IC) {
                int iv = _IC == IC - 1 ? Ir : 16;
                for_each (_iv, iv) {
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

  template int compare_conv_results<float>(
      eld_conv_t<float> &, float *, float *);

  template int ref_convolution2d<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);

  template int ref_convolution2d_block16<float>(
      eld_conv_t<float> &, float *, float *, float *, float *);
}
}
