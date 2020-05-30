#pragma once

#include <stdlib.h>
#include <cassert>
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

} // namespace euler
