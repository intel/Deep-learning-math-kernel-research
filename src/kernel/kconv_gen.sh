#!/bin/bash

# Build conv kernel instantiation
#

src_file=$1; dst_dir=$2; cc=$3; enable_user_fp16=$4

if [ ! -f $src_file ] || [ ! -d $dst_dir ]; then
  "Invalid src_file=$src_file or dst_dir=$dst_dir"
  exit -1
fi

__kconv_generate_inst__() {
  ktype=$1; dtype=$2; V=$3; Vx=$4; I=$5; S=$6; F=$7;

  cat <<@ > $dst_dir/elk_${ktype}_otj_${dtype}_${V}_${Vx}_${I}_${S}_${F}.cpp
// _generated_kernel_file_
//
#include "$src_file"

using namespace euler;

namespace euler {

#undef E
#define E(O, T, K) \\
  ${ktype}_kernel_binder::conv_ker_cls<conv_impl::$dtype, \\
      $V, $Vx, $I, $S, $F, O, T, K>::conv
  ${ktype}_kernel_binder::kconv<conv_impl::$dtype>
      *${ktype}_kernel_binder::kconv_${dtype}_${V}_${Vx}_${I}_${S}_${F}[8][32][3] =
  { // 8
    { // 32
      { E(1, 1,  3), E(1, 1,  5), E(1, 1,  7) },
      { E(1, 2,  3), E(1, 2,  5), E(1, 2,  7) },
      { E(1, 3,  3), E(1, 3,  5), E(1, 3,  7) },
      { E(1, 4,  3), E(1, 4,  5), E(1, 4,  7) },
      { E(1, 5,  3), E(1, 5,  5), E(1, 5,  7) },
      { E(1, 6,  3), E(1, 6,  5), E(1, 6,  7) },
      { E(1, 7,  3), E(1, 7,  5), E(1, 7,  7) },
      { E(1, 8,  3), E(1, 8,  5), E(1, 8,  7) },
      { E(1, 9,  3), E(1, 9,  5), E(1, 9,  7) },
      { E(1, 10, 3), E(1, 10, 5), E(1, 10, 7) },
      { E(1, 11, 3), E(1, 11, 5), E(1, 11, 7) },
      { E(1, 12, 3), E(1, 12, 5), E(1, 12, 7) },
      { E(1, 13, 3), E(1, 13, 5), E(1, 13, 7) },
      { E(1, 14, 3), E(1, 14, 5), E(1, 14, 7) },
    },
    { // 32
      { E(2,  1, 3), E(2,  1, 5), E(2,  1, 7) },
      { E(2,  2, 3), E(2,  2, 5), E(2,  2, 7) },
      { E(2,  3, 3), E(2,  3, 5), E(2,  3, 7) },
      { E(2,  4, 3), E(2,  4, 5), E(2,  4, 7) },
      { E(2,  5, 3), E(2,  5, 5), E(2,  5, 7) },
      { E(2,  6, 3), E(2,  6, 5), E(2,  6, 7) },
      { E(2,  7, 3), E(2,  7, 5), E(2,  7, 7) },
      { E(2,  8, 3), E(2,  8, 5), E(2,  8, 7) },
      { E(2,  9, 3), E(2,  9, 5), E(2,  9, 7) },
      { E(2, 10, 3), E(2, 10, 5), E(2, 10, 7) },
      { E(2, 11, 3), E(2, 11, 5), E(2, 11, 7) },
      { E(2, 12, 3), E(2, 12, 5), E(2, 12, 7) },
      { E(2, 13, 3), E(2, 13, 5), E(2, 13, 7) },
      { E(2, 14, 3), E(2, 14, 5), E(2, 14, 7) },
    },
  };

} // namespace
@
}

if [ $enable_user_fp16 == "ON" ]; then
  eval $($cc -DENABLE_USER_FP16 -DBUILD_OTJ_TBL -E $src_file 2>&1 | grep _generate_inst_)
else
  eval $($cc -DBUILD_OTJ_TBL -E $src_file 2>&1 | grep _generate_inst_)
fi
