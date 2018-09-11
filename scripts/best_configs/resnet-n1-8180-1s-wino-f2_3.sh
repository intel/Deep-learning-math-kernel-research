#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@
export NSOCKETS=1
# resnet50_res2a_branch2b, 5.2T   tflops=4.83548
NSOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=4 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=2 --output-as-blocked=true $COMMON
sleep 1
# resnet50_res3a_branch2b, 4.4T   tflops:3.68377
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n1 --tile-size=4 --execution-mode=0xa061 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=7 --pat-o=4 --output-as-blocked=true $COMMON
sleep 1
# resnet50_res4a_branch2b, 4.2T   tplops:3.18039
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=16 --blk-o=16 --flt-o=1 --flt-t=25 $COMMON
sleep 1
# resnet50_res5a_branch2b, 2.5T   tflops:2.27461
NSOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n1 --tile-size=4 --execution-mode=0xa000 --blk-i=32 --blk-o=32 --flt-o=1 --flt-t=16 $COMMON
