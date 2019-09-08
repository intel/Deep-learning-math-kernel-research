#!/bin/bash

ROOT_DIR="$(dirname $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd))"

echo Root dir: $ROOT_DIR
echo

nsockets=$NSOCKETS
ncores_per_socket=
nthreads_per_core=1
nthreads=$NTHREADS
build_dir=build

nsockets=${nsockets:=$( lscpu | grep 'Socket(s)' | cut -d: -f2 )}
ncores_per_socket=${ncores_per_socket:=$( lscpu | grep 'Core(s) per socket' | cut -d: -f2 )}
nthreads=${nthreads:=$(( nsockets  * ncores_per_socket * nthreads_per_core ))}

euler_mt_runtime="OMP"
if EULER_VERBOSE=1 build/tests/elt_conv -version | grep -i 'MT_RUNTIME: TBB' >& /dev/null; then
  euler_mt_runtime="TBB"
fi

if [ $euler_mt_runtime = "OMP" ]; then
OMP_ENV="OMP_NUM_THREADS=$(( nthreads )) \
  KMP_HW_SUBSET=$(( nsockets ))s,$(( ncores_per_socket ))c,$(( nthreads_per_core ))t \
  KMP_AFFINITY=compact,granularity=fine \
  KMP_BLOCKTIME=infinite"
else # TBB
  if [ "x$NTHREADS" != "x" ] || [ "x$NSOCKETS" != "x" ]; then
    echo -e "Warning: \$NSOCKETS or \$NTHREADS is not supported in TBB mode\n"
  fi
  OMP_ENV="KMP_BLOCKTIME=0"
fi
echo OMP Environment: $OMP_ENV

function conv_test() {
  # Default
  n=1; g=1; i=0; o=0; h=0; w=0; H=0; W=0; k=3; K=3; p=1; P=1; s=1; S=1
  b=1; r=0; v=1; a=wino; l=16; B=0; A=0; T=0
  flt_o=0; flt_t=0; blk_i=0; blk_o=0; pat_i=1; pat_o=1
  tile_size=5; nthreads=0; execution_mode=0
  streaming_input=0; streaming_output=0
  input_format=nChw16c; weights_format=OIhw16i16o; output_format=nChw16c
  input_as_blocked=0; weights_as_blocked=0; output_as_blocked=0
  with_ip_sum=0; with_argmax=0; f16c_opt=0; data_type_cfg=0
  input_file=""; weights_file=""; bias_file=""
  sampling_kind=2; tinput_cali_s=0; tinput_cali_z=0
  disable_autoparam=1

  OPTIND=1
  while getopts ":n:g:i:o:h:w:H:W:k:K:p:P:s:S:b:r:v:f:l:B:A:a:-:" opt; do
    case "$opt" in
      g) g=$OPTARG ;;
      n) n=$OPTARG ;;
      i) i=$OPTARG ;;
      o) o=$OPTARG ;;
      h) h=$OPTARG ;;
      w) w=$OPTARG ;;
      H) H=$OPTARG ;;
      W) W=$OPTARG ;;
      k) k=$OPTARG ;;
      K) K=$OPTARG ;;
      p) p=$OPTARG ;;
      P) P=$OPTARG ;;
      s) s=$OPTARG ;;
      S) S=$OPTARG ;;
      b) b=$OPTARG ;;
      r) r=$OPTARG ;;
      v) v=$OPTARG ;;
      a) a=$OPTARG ;;
      l) l=$OPTARG ;;
      B) B=$OPTARG ;;
      A) A=$OPTARG ;;
      -)
        case "${OPTARG}" in
          alg) a="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          alg=*) a=${OPTARG#*=}
            ;;
          nthreads) nthreads="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          nthreads=*) nthreads=${OPTARG#*=}
            ;;
          execution-mode) execution_mode="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          execution-mode=*) execution_mode=${OPTARG#*=}
            ;;
          flt-o) flt_o="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          flt-o=*) flt_o=${OPTARG#*=}
            ;;
          flt-t) flt_t="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          flt-t=*) flt_t=${OPTARG#*=}
            ;;
          blk-i) blk_i="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          blk-i=*) blk_i=${OPTARG#*=}
            ;;
          blk-o) blk_o="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          blk-o=*) blk_o=${OPTARG#*=}
            ;;
          pat-i) pat_i="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          pat-i=*) pat_i=${OPTARG#*=}
            ;;
          pat-o) pat_o="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          pat-o=*) pat_o=${OPTARG#*=}
            ;;
          tile-size) tile_size="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          tile-size=*) tile_size=${OPTARG#*=}
            ;;
          streaming-input) streaming_input="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          streaming-input=*) streaming_input=${OPTARG#*=}
            ;;
          streaming-output) streaming_output="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          streaming-output=*) streaming_output=${OPTARG#*=}
            ;;
          input-format) input_format="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          input-format=*) input_format=${OPTARG#*=}
            ;;
          weights-format) weights_format="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          weights-format=*) weights_format=${OPTARG#*=}
            ;;
          output-format) output_format="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          output-format=*) output_format=${OPTARG#*=}
            ;;
          input-as-blocked) input_as_blocked="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          input-as-blocked=*) input_as_blocked=${OPTARG#*=}
            ;;
          weights-as-blocked) weights_as_blocked="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          weights-as-blocked=*) weights_as_blocked=${OPTARG#*=}
            ;;
          output-as-blocked) output_as_blocked="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          output-as-blocked=*) output_as_blocked=${OPTARG#*=}
            ;;
          with-ip-sum) with_ip_sum="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          with-ip-sum=*) with_ip_sum=${OPTARG#*=}
            ;;
          with-argmax) with_argmax="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          with-argmax=*) with_argmax=${OPTARG#*=}
            ;;
          f16c-opt) f16c_opt="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          f16c-opt=*) f16c_opt=${OPTARG#*=}
            ;;
          data-type-cfg) data_type_cfg="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          data-type-cfg=*) data_type_cfg=${OPTARG#*=}
            ;;
          input-data-file) input_file="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          input-data-file=*) input_file=${OPTARG#*=}
            ;;
          weights-data-file) weights_file="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          weights-data-file=*) weights_file=${OPTARG#*=}
            ;;
          bias-data-file) bias_file="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          bias-data-file=*) bias_file=${OPTARG#*=}
            ;;
          sampling-kind) sampling_kind="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          sampling-kind=*) sampling_kind=${OPTARG#*=}
            ;;
          tinput-cali-s) tinput_cali_s="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          tinput-cali-s=*) tinput_cali_s=${OPTARG#*=}
            ;;
          tinput-cali-z) tinput_cali_z="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          tinput-cali-z=*) tinput_cali_z=${OPTARG#*=}
            ;;
          disable-autoparam) disable_autoparam="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
            ;;
          disable-autoparam=*) disable_autoparam=${OPTARG#*=}
            ;;
       esac
       ;;
    esac
  done
  shift $((OPTIND-1))
  input_file_opt=""
  weights_file_opt=""
  bias_file_opt=""
  if [ "x$input_file" != "x" ]; then input_file_opt="--input-data-file=$input_file"; fi
  if [ "x$weights_file" != "x" ]; then weights_file_opt="--weights-data-file=$weights_file"; fi
  if [ "x$bias_file" != "x" ]; then bias_file_opt="--bias-data-file=$bias_file"; fi
  set -v
  eval $OMP_ENV $ROOT_DIR/$build_dir/tests/elt_conv \
    -mb=$n -g=$g -ic=$i -oc=$o -ih=$h -iw=$w -oh=$H -ow=$W -kh=$k -kw=$K -ph=$p -pw=$P -sh=$s -sw=$S \
    -with_bias=$b -with_relu=$r -validate_results=$v -alg=$a -repeated_layer=$l -dbuffering=$B -output_as_input=$A \
    -flt_o=$flt_o -flt_t=$flt_t -blk_i=$blk_i -blk_o=$blk_o \
    -pat_i=$pat_i -pat_o=$pat_o -tile_size=$tile_size \
    -nthreads=$nthreads -execution_mode=$execution_mode \
    -streaming_input=$streaming_input \
    -streaming_output=$streaming_output \
    -input_format=$input_format \
    -weights_format=$weights_format \
    -output_format=$output_format \
    -input_as_blocked=$input_as_blocked \
    -weights_as_blocked=$weights_as_blocked \
    -output_as_blocked=$output_as_blocked   \
    -with_ip_sum=$with_ip_sum \
    -with_argmax=$with_argmax \
    -f16c_opt=$f16c_opt \
    -data_type_cfg=$data_type_cfg \
    -sampling_kind=$sampling_kind \
    -tinput_cali_s=$tinput_cali_s \
    -tinput_cali_z=$tinput_cali_z \
    -disable_autoparam=$disable_autoparam \
    $input_file_opt \
    $weights_file_opt \
    $bias_file_opt
  set +v
}

function show_help() {
cat <<@

Euler test script:
  -h        display this help and exit.
  -c        convolution test

@
}

OPTIND=1
while getopts ":hc" opt; do
  case "$opt" in
    h)
      show_help
      exit 0
      ;;
    c)
      echo Run convolution test...
      shift
      conv_test $@
      exit 0
      ;;
    \?)
      show_help
      exit 0
      ;;
  esac
done
shift $((OPTIND-1))
