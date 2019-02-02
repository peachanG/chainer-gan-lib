#!/bin/bash

for i in `seq 1 9`; do
  echo "start train class= $i"
  python3 train.py --gpu 0 --algorithm stdgan --architecture sndcgan \
    --n_dis 1 --adam_beta1 0.5 --adam_beta2 0.999 \
    --out data/results/chainer/sn/$i --data_path data/datasets/cifar10/train \
    --class_name $i
done
