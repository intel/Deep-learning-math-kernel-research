#!/bin/bash

# Build conv kernel instantiation
#

src_file=$1; dst_dir=$2; cc=$3; enable_user_fp16=$4

if [ ! -f $src_file ] || [ ! -d $dst_dir ]; then
  "Invalid src_file=$src_file or dst_dir=$dst_dir"
  exit -1
fi

__u8s8_depthwise_kconv_generate_inst__() {
  ktype=$1; dtype=$2; otype=$3; V=$4; Vx=$5; I=$6; S=$7; F=$8;

  cat <<@ > $dst_dir/elk_${ktype}_${dtype}_${otype}_${V}_${Vx}_${I}_${S}_${F}.cpp
// _generated_kernel_file_
//
#include "$src_file"

using namespace euler;

namespace euler {

#undef E
#define E(O, T, K) \\
  ${ktype}_kernel_binder::conv_ker_cls<conv_impl::$dtype, \\
      $otype, $V, $Vx, $I, $S, $F, O, T, K>::conv
  ${ktype}_kernel_binder::kconv<conv_impl::$dtype, $otype>
      *${ktype}_kernel_binder::kconv_${dtype}_${otype}_${V}_${Vx}_${I}_${S}_${F}[1][32][1] =
  { // 8
    { // 32
      { E(1, 1,  3) },
      { E(1, 2,  3) },
      { E(1, 3,  3) },
      { E(1, 4,  3) },
      { E(1, 5,  3) },
      { E(1, 6,  3) },
      { E(1, 7,  3) },
      { E(1, 8,  3) },
      { E(1, 9,  3) },
      { E(1, 10, 3) },
      { E(1, 11, 3) },
      { E(1, 12, 3) },
      { E(1, 13, 3) },
      { E(1, 14, 3) },
      { E(1, 15, 3) },
      { E(1, 16, 3) },
      { E(1, 17, 3) },
      { E(1, 18, 3) },
      { E(1, 19, 3) },
      { E(1, 20, 3) },
      { E(1, 21, 3) },
      { E(1, 22, 3) },
      { E(1, 23, 3) },
      { E(1, 24, 3) },
      { E(1, 25, 3) },
      { E(1, 26, 3) },
      { E(1, 27, 3) },
      { E(1, 28, 3) },
      { E(1, 29, 3) },
      { E(1, 30, 3) },
      { E(1, 31, 3) },
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
