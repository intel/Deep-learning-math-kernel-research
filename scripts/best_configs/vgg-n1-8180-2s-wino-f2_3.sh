#/bin/bash

# VGG
# batch-size: 1
# SKX 8180 2S

source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 12.8T tflops=10.5969
NSOCKETS=2 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n1 --blk-i=4 --blk-o=2 --blk-t=28 --tile-size=4 --execution-mode=0xa040 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1, 13.5T tflops=11.1854
NSOCKETS=2 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n1 --blk-i=4 --blk-o=8 --blk-t=28 --tile-size=4 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv2_2, 14.3T tflops=11.7138
NSOCKETS=2 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n1 --blk-i=8 --blk-o=8 --blk-t=28 --tile-size=4 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv3_1, 13.0T tflops=11.5115
NSOCKETS=2 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=4 --blk-t=28 --pat-o=2 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv3_2, 12.6T tflops=11.1579
NSOCKETS=2 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=8 --blk-t=28 --pat-o=2 --tile-size=4 --execution-mode=0xa061 $COMMON
sleep 1
# vgg19_conv4_1, 10.2T   tflops=10.2682
NSOCKETS=2 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=16 --blk-t=28 --tile-size=4 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv4_2, 10.5T tflops= 11.6569
NSOCKETS=2 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=32 --blk-t=28 --tile-size=4 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv5_1, 7.3T tflops=8.14171
NSOCKETS=2 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n1 --blk-i=16 --blk-o=1 --blk-t=25 --tile-size=4 --execution-mode=0xa000 $COMMON
