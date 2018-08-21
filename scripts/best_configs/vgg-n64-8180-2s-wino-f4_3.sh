#/bin/bash

# VGG
# batch-size: 64
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 12.2T
NSOCKETS=2 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n64 --tile-size=6 --blk-i=4 --blk-o=4 --blk-t=28 --execution-mode=0xa061 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1, 13.3T
NSOCKETS=2 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n64 --tile-size=6 --blk-i=4 --blk-o=8 --blk-t=28 --execution-mode=0xa040 $COMMON 
sleep 1
# vgg19_conv2_2, 14.2T
NSOCKETS=2 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n64 --tile-size=6 --blk-i=8 --blk-o=8 --blk-t=28 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv3_1, 13.9T
NSOCKETS=2 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n64 --tile-size=6 --blk-i=8 --blk-o=8 --blk-t=28 --pat-o=2 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv3_2, 11.2T
NSOCKETS=2 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n64 --tile-size=6 --blk-i=8 --blk-o=8 --blk-t=28 --pat-o=2 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv4_1, 10.0T
NSOCKETS=2 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n64 --tile-size=6 --blk-i=8 --blk-o=8 --blk-t=31 --execution-mode=0xa000 --streaming-input=2 $COMMON
sleep 1
# vgg19_conv4_2, 10.8T
NSOCKETS=2 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n64 --tile-size=6 --blk-i=8 --blk-o=4 --blk-t=31 --execution-mode=0xa000 --streaming-input=2 $COMMON 
sleep 1
# vgg19_conv5_1, 10.6T
NSOCKETS=2 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n64 --tile-size=6 --blk-i=8 --blk-o=4 --blk-t=28 --execution-mode=0xa000 --streaming-input=2 $COMMON 
