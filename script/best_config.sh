#/bin/bash

m=$1
# batch-size: 1, 8180-2s

if [ "x$m" = "xvgg-n1-8180-1s" ]; then
  # vgg19_conv1_2, 7.0T
  ./script/run.sh -c -i64 -h224 -o64 -H224 -v0 -n1 --blk-i=4 --blk-o=4 --blk-t=29 --execution-mode=0xa040
  # vgg19_conv2_1, 7.3T
  ./script/run.sh -c -i64 -h112 -o128 -H112 -v0 -n1 --blk-i=4 --blk-o=8 --blk-t=26 --execution-mode=0xa040
  # vgg19_conv2_2, 7.4T
  ./script/run.sh -c -i128 -h112 -o128 -H112 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=26 --execution-mode=0xa040
  # vgg19_conv3_1, 7.3T
  ./script/run.sh -c -i128 -h56 -o256 -H56 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=26 --pat-o=2 --execution-mode=0xa061
  # vgg19_conv3_2, 7.3T
  ./script/run.sh -c -i256 -h56 -o256 -H56 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=26 --pat-o=2 --execution-mode=0xa061
  # vgg19_conv4_1, 5.6T
  #./script/run.sh -c -i256 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=15 --pat-o=4 --execution-mode=0xa061
  # vgg19_conv4_1, 6.1T
  ./script/run.sh -c -i256 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=25 --execution-mode=0xa000
  # vgg19_conv4_2, 6.0T
  ./script/run.sh -c -i512 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=26 --execution-mode=0xa000
  # vgg19_conv5_1, 5.2T
  ./script/run.sh -c -i512 -h14 -o512 -H14 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=25 --execution-mode=0xa000
fi

if [ "x$m" = "xvgg-n1-8180-2s" ]; then
  # vgg19_conv1_2, 12.8T
  ./script/run.sh -c -i64 -h224 -o64 -H224 -v0 -n1 --blk-i=4 --blk-o=4 --blk-t=17 --execution-mode=0xa040
  # vgg19_conv2_1, 13.5T
  ./script/run.sh -c -i64 -h112 -o128 -H112 -v0 -n1 --blk-i=4 --blk-o=8 --blk-t=26 --execution-mode=0xa040
  # vgg19_conv2_2, 14.3T
  ./script/run.sh -c -i128 -h112 -o128 -H112 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=26 --execution-mode=0xa040
  # vgg19_conv3_1, 13.0T
  ./script/run.sh -c -i128 -h56 -o256 -H56 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=26 --pat-o=4 --execution-mode=0xa061
  # vgg19_conv3_2, 12.6T
  ./script/run.sh -c -i256 -h56 -o256 -H56 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=26 --pat-o=4 --execution-mode=0xa061
  # vgg19_conv4_1, 9.5T
  #./script/run.sh -c -i256 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=15 --pat-o=8 --execution-mode=0xa061
  # vgg19_conv4_1, 9.7T
  #./script/run.sh -c -i256 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=8 --blk-t=15 --pat-i=2 --pat-o=4 --execution-mode=0xa073
  # vgg19_conv4_1, 10.2T
  ./script/run.sh -c -i256 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=25 --execution-mode=0xa000
  # vgg19_conv4_2, 10.5T
  ./script/run.sh -c -i512 -h28 -o512 -H28 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=26 --execution-mode=0xa000
  # vgg19_conv5_1, 7.3T
  ./script/run.sh -c -i512 -h14 -o512 -H14 -v0 -n1 --blk-i=8 --blk-o=4 --blk-t=25 --execution-mode=0xa000
fi

# batch-size: 1, 8180-1s

if [ "x$m" = "xresnet-n1-8180-1s" ]; then
  # resnet50_res2a_branch2b, 5.2T
  ./script/run.sh -c -i64 -h56 -o64 -H56 -v0 -n1 --execution-mode=0xa061 --blk-i=4 --blk-o=2 --blk-t=26 --pat-o=2
  # resnet50_res3a_branch2b, 4.4T
  ./script/run.sh -c -i128 -h28 -o128 -H28 -v0 -n1 --execution-mode=0xa061 --blk-i=8 --blk-o=2 --blk-t=15 --pat-o=4
  # resnet50_res4a_branch2b, 4.2T
  ./script/run.sh -c -i256 -h14 -o256 -H14 -v0 -n1 --execution-mode=0xa000 --blk-i=8 --blk-o=4 --blk-t=25
  # resnet50_res5a_branch2b, 2.5T
  ./script/run.sh -c -i512 -h7 -o512 -H7 -v0 -n1 --execution-mode=0xa000 --blk-i=8 --blk-o=4 --blk-t=9
fi

if [ "x$m" = "xresnet-n1-8180-2s" ]; then
  # resnet50_res2a_branch2b, 7.7T
  ./script/run.sh -c -i64 -h56 -o64 -H56 -v0 -n1 --execution-mode=0xa061 --blk-i=4 --blk-o=2 --blk-t=13 --pat-o=1
  # resnet50_res3a_branch2b, 6.0T
  ./script/run.sh -c -i128 -h28 -o128 -H28 -v0 -n1 --execution-mode=0xa061 --blk-i=8 --blk-o=1 --blk-t=15 --pat-o=8
  # resnet50_res4a_branch2b, 5.1T
  ./script/run.sh -c -i256 -h14 -o256 -H14 -v0 -n1 --execution-mode=0xa000 --blk-i=8 --blk-o=1 --blk-t=9
  # resnet50_res5a_branch2b, 3.6T
  ./script/run.sh -c -i512 -h7 -o512 -H7 -v0 -n1 --execution-mode=0xa000 --blk-i=8 --blk-o=4 --blk-t=9
fi

if [ "x$m" = "xresnet-n64-8180-2s" ]; then
  # resnet50_res2a_branch2b, 11.2T
  ./script/run.sh -c -i64 -h56 -o64 -H56 -v0 -n64 --execution-mode=0xa040 --blk-i=4 --blk-o=4 --blk-t=28
  # resnet50_res3a_branch2b, 11.2T
  ./script/run.sh -c -i128 -h28 -o128 -H28 -v0 -n64 --execution-mode=0xa040 --blk-i=8 --blk-o=8 --blk-t=29
  # resnet50_res4a_branch2b, 9.7T
  ./script/run.sh -c -i256 -h14 -o256 -H14 -v0 -n64 --execution-mode=0xa061 --blk-i=8 --blk-o=4 --blk-t=23 --pat-o=4
  # resnet50_res5a_branch2b, 6.7T
  ./script/run.sh -c -i512 -h7 -o512 -H7 -v0 -n64 --execution-mode=0xa000 --blk-i=8 --blk-o=8 --blk-t=29
fi
