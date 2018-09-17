#/bin/bash

# VGG
# batch-size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n64 --tile-size=4 --blk-i=4 --blk-o=4 --flt-t=28 --tile-size=4 --execution-mode=0xa061 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n64 --tile-size=4 --blk-i=4 --blk-o=8 --flt-t=30 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv2_2
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n64 --tile-size=4 --blk-i=8 --blk-o=8 --flt-t=27 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv3_1
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n64 --tile-size=4 --blk-i=8 --blk-o=16 --flt-t=29 --pat-o=1 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv3_2
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n64 --tile-size=4 --blk-i=8 --blk-o=8 --flt-t=29 --pat-o=1 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv4_1
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n64 --tile-size=4 --blk-i=8 --blk-o=16 --flt-t=22 --tile-size=4 --execution-mode=0xa000 --streaming-input=2 $COMMON
sleep 1
# vgg19_conv4_2
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n64 --tile-size=4 --blk-i=8 --blk-o=32 --flt-t=23 --tile-size=4 --execution-mode=0xa000 --streaming-input=2 $COMMON
sleep 1
# vgg19_conv5_1
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n64 --tile-size=4 --blk-i=8 --blk-o=8 --flt-t=25 --tile-size=4 --execution-mode=0xa000 --streaming-input=2 $COMMON
