#!/bin/bash

source ./scripts/best_configs/common.sh $@
COMMON="$COMMON --f16c-opt=1"

echo "Plain: nhwc"
# nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h42 -w256 -H42 -W256 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=4 --flt-o=8 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h32 -w32 -H32 -W32 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=1 --flt-o=2 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o3072 -h42 -w128 -H42 -W128 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=1 --flt-o=2 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o4096 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=4 --blk-o=1 --flt-o=2 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc
NSOCKETS=1 ./scripts/run.sh -c -n1 -i4096 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=16 --blk-o=1 --flt-o=1 --flt-t=7 --execution-mode=0xa060 --pat-o=1 $COMMON --input-format=nhwc --output-format=nhwc

echo "Blocked: nChw16c"
# blocked
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h21 -w256 -H21 -W256 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=8 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h4 -w256 -H4 -W256 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=2 --flt-t=14 --execution-mode=0xa060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o3072 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1024 -o4096 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=32 --pat-i=2  --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-o=1 $COMMON
NSOCKETS=1 ./scripts/run.sh -c -n1 -i4096 -o1024 -h1 -w42 -H1 -W42 -k1 -K1 -p0 -P0 -s1 -S1 -b1 -adirect --blk-i=16 --pat-i=16 --blk-o=1 --flt-o=1 --flt-t=28 --execution-mode=0xa060 --pat-o=1 $COMMON
