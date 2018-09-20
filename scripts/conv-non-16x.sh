#/bin/bash

# conv: wino:
# a061
# with_relu=0 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -i127 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 --tile-size=6 --execution-mode=0xa061 --input-format=nchw --weights-format=oihw --output-format=nchw -v1
# with_relu=1 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -i127 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 --tile-size=6 --execution-mode=0xa061 --input-format=nchw --weights-format=oihw --output-format=nchw -r1 -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=1 ./scripts/run.sh -c -i127 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-o=1 --tile-size=6 --execution-mode=0xa061 --input-format=nchw --weights-format=oihw --output-format=nchw --with-ip-sum=1 -v1

# a0e1
# with_relu=0 with_ip_sum=0
NSOCKETS=2 ./scripts/run.sh -c -i255 -h28 -o511 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=20 --pat-i=1 --pat-o=1 --tile-size=5 --execution-mode=0xa0e1 --input-format=nchw --weights-format=oihw --output-format=nchw -v1
# with_relu=1 with_ip_sum=0
NSOCKETS=2 ./scripts/run.sh -c -i255 -h28 -o511 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=20 --pat-i=1 --pat-o=1 --tile-size=5 --execution-mode=0xa0e1 --input-format=nchw --weights-format=oihw --output-format=nchw -r1 -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=2 ./scripts/run.sh -c -i255 -h28 -o511 -H28 -n1 --blk-i=8 --blk-o=8 --flt-t=20 --pat-i=1 --pat-o=1 --tile-size=5 --execution-mode=0xa0e1 --input-format=nchw --weights-format=oihw --output-format=nchw --with-ip-sum=1 -v1

# a073
# with_relu=0 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -i255 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa073 --input-format=nchw --weights-format=oihw --output-format=nchw -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=1 ./scripts/run.sh -c -i255 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa073 --input-format=nchw --weights-format=oihw --output-format=nchw --with-ip-sum=1 -v1

# a07b
# with_relu=0 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -i255 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa07b --input-format=nchw --weights-format=oihw --output-format=nchw -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=1 ./scripts/run.sh -c -i255 -h56 -o255 -H56 -n1 --blk-i=8 --blk-o=1 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=1 --tile-size=6 --execution-mode=0xa07b --input-format=nchw --weights-format=oihw --output-format=nchw --with-ip-sum=1 -v1



# conv: direct 1x1:
# a061
# with_relu=0 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -n1 -i255  -h56 -o511  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 -v0 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=16 --execution-mode=0xa061 -v1
# with_relu=1 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -n1 -i255  -h56 -o511  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 -v0 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=16 --execution-mode=0xa061 -r1 -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i255  -h56 -o511  -H28 -k1 -K1 -s2 -S2 -p0 -P0 -b1 -adirect_1x1 -v0 --blk-i=16 --flt-o=2 --flt-t=14 --pat-i=1 --pat-o=16 --execution-mode=0xa061 --with-ip-sum=1 -v1

# f061
# with_relu=0 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1023 -o1023 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=13 --execution-mode=0xf061 --pat-o=32 -v1
# with_relu=1 with_ip_sum=0
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1023 -o1023 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=13 --execution-mode=0xf061 --pat-o=32 -r1 -v1
# with_relu=0 with_ip_sum=1
NSOCKETS=1 ./scripts/run.sh -c -n1 -i1023 -o1023 -h19 -k1 -K1 -H19 -p0 -P0 -s1 -b1 -r0 -v0 -adirect_1x1 --blk-i=32 --flt-o=2 --flt-t=13 --execution-mode=0xf061 --pat-o=32 --with-ip-sum=1 -v1
