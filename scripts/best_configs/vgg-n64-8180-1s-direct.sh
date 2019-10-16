#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

# vgg19_conv1_1,
NSOCKETS=1 ./scripts/run.sh -c -i3 -h224 -o64 -H224 -n64 --blk-i=1 --blk-o=2 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON --input-format=nchw --weights-format=hwio

sleep 1
# vgg19_conv1_2,
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n64 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON

sleep 1
# vgg19_conv2_1,
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n64 --blk-i=4 --blk-o=2 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-o=1 $COMMON

sleep 1
# vgg19_conv2_2,
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n64 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-i=1 --pat-o=1 $COMMON

sleep 1
# vgg19_conv3_1,
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n64 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 -adirect --execution-mode=0xc060 $COMMON

sleep 1
# vgg19_conv3_2,
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n64 --blk-i=8 --blk-o=2 --flt-o=2 --flt-t=14 --pat-i=2 --pat-o=1 -adirect --execution-mode=0xc060 $COMMON

sleep 1
# vgg19_conv4_1, 6.6T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n64 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-i=2 --pat-o=1 $COMMON

sleep 1
# vgg19_conv4_2, 6.4T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n64 --blk-i=16 --blk-o=1 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-i=2 --pat-o=1 $COMMON

sleep 1
# vgg19_conv5_1, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n64 --blk-i=16 --blk-o=2 --flt-o=2 --flt-t=14 -adirect --execution-mode=0xc060 --pat-i=2 --pat-o=1 $COMMON --f16c-opt=1
