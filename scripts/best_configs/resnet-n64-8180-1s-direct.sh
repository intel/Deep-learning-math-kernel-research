#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o64 -H112 -k7 -K7 -s2 -S2 -p3 -P3 -n64 -adirect --execution-mode=0xc060 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON --input-format=nchw --weights-format=hwio --f16c-opt=1
sleep 1
# resnet50_res2a_branch2b, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n64 -adirect --execution-mode=0xc060 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 --pat-o=1 $COMMON
sleep 1
# resnet50_res3a_branch2b, 7.0 - 7.5T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n64 -adirect --execution-mode=0xc060 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 $COMMON
sleep 1
# resnet50_res4a_branch2b, 4.2T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n64 -adirect --execution-mode=0xc060 --blk-i=16 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 $COMMON --f16c-opt=1
sleep 1
# resnet50_res5a_branch2b, 2.5T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n64 -adirect --execution-mode=0xc060 --blk-i=16 --blk-o=1 --flt-o=2 --flt-t=7 --pat-i=2 $COMMON --f16c-opt=1
sleep 1
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H28 -k3 -s2 -S2 -n64 -adirect --execution-mode=0xc060 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 --pat-i=1 $COMMON --f16c-opt=1
sleep 1
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H14 -k3 -s2 -S2 -n64 -adirect --execution-mode=0xc060 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 --pat-i=1 $COMMON --f16c-opt=1
sleep 1
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H7 -k3 -s2 -S2 -n64 -adirect --execution-mode=0xc060 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=7 --pat-i=2 $COMMON --f16c-opt=1
