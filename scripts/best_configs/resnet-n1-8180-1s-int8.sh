#/bin/bash

# Resnet50
# batch-size: 1
# SKX 8180 1S

source ./scripts/best_configs/common.sh $@

# resnet50_res2a_branch2b 6.75T
NOCKETS=1 ./scripts/run.sh -c -i64 -h56 -o64 -H56 -n1 --tile-size=6 --execution-mode=0xa161 --blk-i=1 --blk-o=1 --flt-o=4 --flt-t=7 --pat-o=1 --output-as-blocked=true -v0 $COMMON
# resnet50_res3a_branch2b 5.62T
NSOCKETS=1 ./scripts/run.sh -c -i128 -h28 -o128 -H28 -n1 --tile-size=4 --execution-mode=0xa161 --blk-i=2 --blk-o=2 --flt-o=4 --flt-t=7 --pat-o=1 -v0 $COMMON
# resnet50_res4a_branch2b 4.95T
NSOCKETS=1 ./scripts/run.sh -c -i256 -h14 -o256 -H14 -n1 --tile-size=5 --execution-mode=0xa133 --blk-i=4 --blk-o=2 --flt-o=4 --flt-t=5 -v0 $COMMON
# resnet50_res5a_branch2b 3.19T
NOCKETS=1 ./scripts/run.sh -c -i512 -h7 -o512 -H7 -n1 --tile-size=4 --execution-mode=0xa133 --blk-i=8 --blk-o=1 --flt-o=1 --flt-t=26 -v0 $COMMON
