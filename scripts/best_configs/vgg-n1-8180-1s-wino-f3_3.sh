#/bin/bash

# VGG19
# batch-size: 1
# SKX 8180 1S


source ./scripts/best_configs/common.sh $@

# vgg19_conv1_2, 7.0T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h224 -o64 -H224 -n1 --blk-i=4 --blk-o=4 --flt-t=29 --tile-size=5 --execution-mode=0xa040 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv2_1, 7.3T
NSOCKETS=1 ./scripts/run.sh -c -i64 -h112 -o128 -H112 -n1 --blk-i=4 --blk-o=8 --flt-t=26 --tile-size=5 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv2_2, 7.4T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h112 -o128 -H112 -n1 --blk-i=8 --blk-o=8 --flt-t=26 --tile-size=5 --execution-mode=0xa040 $COMMON
sleep 1
# vgg19_conv3_1, 7.3T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=8 --flt-t=26 --pat-o=2 --tile-size=5 --execution-mode=0xa061 --output-as-blocked=true $COMMON
sleep 1
# vgg19_conv3_2, 7.3T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h56 -o256 -H56 -n1 --blk-i=8 --blk-o=8 --flt-t=26 --pat-o=2 --tile-size=5 --execution-mode=0xa060 $COMMON
sleep 1
# vgg19_conv4_1, 5.6T
# NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=15 --pat-o=4 --tile-size=5 --execution-mode=0xa061 $COMMON
# vgg19_conv4_1, 5.3T
# NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=28 --pat-o=4 --tile-size=5 --execution-mode=0xa061 --tile-size=4 $COMMON
# vgg19_conv4_1, 5.5T
# NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=20 --pat-i=1 --pat-o=1 --tile-size=5 --execution-mode=0xa0e0 $COMMON
# vgg19_conv4_1, 6.6T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=4 --flt-t=25 --tile-size=5 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv4_2, 6.4T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h28 -o512 -H28 -n1 --blk-i=8 --blk-o=4 --flt-t=26 --tile-size=5 --execution-mode=0xa000 --streaming-input=1 $COMMON
sleep 1
# vgg19_conv5_1, 5.2T
NSOCKETS=1 ./scripts/run.sh -c -i512 -h14 -o512 -H14 -n1 --blk-i=8 --blk-o=4 --flt-t=25 --tile-size=5 --execution-mode=0xa000 $COMMON

