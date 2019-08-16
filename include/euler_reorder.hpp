#pragma once

#include <stdlib.h>
#include "euler.hpp"

namespace euler {

// Reorder
template <typename Type, const int dst_fmt, const int src_fmt, typename... Args>
struct EULER_API reorder {
  reorder(Type *dst, Type *src, Args...) {
    assert(!!"reorder not implemented\n");
    abort();
  }
};

template <typename Type> struct EULER_API reorder<Type, nchw, nhwc> {
  reorder(Type *dst, Type *src, int n, int c, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, nhwc, nchw> {
  reorder(Type *dst, Type *src, int n, int c, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, nchw, nChw16c> {
  reorder(Type *dst, Type *src, int n, int c, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, nChw16c, nchw> {
  reorder(Type *dst, Type *src, int n, int c, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, oihw, OIhw16i16o> {
  reorder(Type *dst, Type *src, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, goihw, gOIhw16i16o> {
  reorder(Type *dst, Type *src, int g, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, OIhw16i16o, oihw> {
  reorder(Type *dst, Type *src, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, gOIhw16i16o, goihw> {
  reorder(Type *dst, Type *src, int g, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, oihw, hwio> {
  reorder(Type *dst, Type *src, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, goihw, ghwio> {
  reorder(Type *dst, Type *src, int g, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, hwio, oihw> {
  reorder(Type *dst, Type *src, int o, int i, int h, int w);
};

template <typename Type> struct EULER_API reorder<Type, ghwio, goihw> {
  reorder(Type *dst, Type *src, int g, int o, int i, int h, int w);
};

template struct reorder<float,   nchw, nhwc>;
template struct reorder<float,   nhwc, nchw>;
template struct reorder<float,   nchw, nChw16c>;
template struct reorder<float,   nChw16c, nchw>;
template struct reorder<float,   oihw, OIhw16i16o>;
template struct reorder<float,   goihw, gOIhw16i16o>;
template struct reorder<float,   OIhw16i16o, oihw>;
template struct reorder<float,   gOIhw16i16o, goihw>;
template struct reorder<float,   oihw, hwio>;
template struct reorder<float,   goihw, ghwio>;
template struct reorder<float,   hwio, oihw>;
template struct reorder<float,   ghwio, goihw>;
template struct reorder<uint8_t, nchw, nhwc>;
template struct reorder<uint8_t, nhwc, nchw>;
template struct reorder<uint8_t, nchw, nChw16c>;
template struct reorder<uint8_t, nChw16c, nchw>;
template struct reorder<uint8_t, oihw, OIhw16i16o>;
template struct reorder<uint8_t, goihw, gOIhw16i16o>;
template struct reorder<uint8_t, OIhw16i16o, oihw>;
template struct reorder<uint8_t, gOIhw16i16o, goihw>;
template struct reorder<uint8_t, oihw, hwio>;
template struct reorder<uint8_t, goihw, ghwio>;
template struct reorder<uint8_t, hwio, oihw>;
template struct reorder<uint8_t, ghwio, goihw>;

} // namespace euler
