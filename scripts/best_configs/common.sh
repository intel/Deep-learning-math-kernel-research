#!/bin/bash

function usage() {
cat <<!
  -v   Validation on.
  -p   Use plain format.
  -t   Tile size
  -h   This page.
!
}

v=0
input_format=nChw16c
weights_format=OIhw16i16o
output_format=nChw16c
tile_size=5

OPTIND=1
while getopts "vpt:h" opt; do
  case $opt in
    v)
      v=1
      ;;
    p)
      input_format=nchw
      weights_format=oihw
      output_format=nchw
      ;;
    t)
      tile_size=$OPTARG
      ;;
    h)
      usage
      ;;
    ?)
      usage
      ;;
  esac
done

COMMON="-v$v --input-format=$input_format --weights-format=$weights_format --output-format=$output_format --tile-size=$tile_size "

echo "Common option:" $COMMON
echo

