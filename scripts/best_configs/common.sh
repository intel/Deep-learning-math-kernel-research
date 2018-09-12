#!/bin/bash

function usage() {
cat <<!
  -v   Validation on.
  -p   Use plain format.
  -r   ReLU fusion.
  -l   Flush loop.
  -f   Flush cache.
  -s   Inplace sum.
  -h   This page.
!
}

v=0; r=0; s=0; l=1; f=0
input_format=nChw16c
weights_format=OIhw16i16o
output_format=nChw16c
tile_size=5

OPTIND=1
while getopts "vprsft:l:h" opt; do
  case $opt in
    v)
      v=1
      ;;
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    r)
      r=1
      ;;
    s)
      s=1
      ;;
    l)
      l=$OPTARG
      ;;
    f)
      f=1
      ;;
    h)
      usage
      ;;
    ?)
      usage
      ;;
  esac
done
shift $((OPTIND-1))

COMMON="-v$v --input-format=$input_format --weights-format=$weights_format --output-format=$output_format -r$r --with-ip-sum=$s -l$l -f$f"

echo "Common option:" $COMMON
echo

