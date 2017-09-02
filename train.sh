#! /bin/bash
#LOG=/home/sensetime/detection/train.log
CAFFE=/home/sensetime/detection/tools/train_faster_rcnn_alt_opt.py
python $CAFFE --gpu 0 --net_name ZF --weights /home/sensetime/detection/data/imagenet_models/ZF.v2.caffemodel --imdb imagenet --cfg /home/sensetime/detection/experiments/cfgs/faster_rcnn_alt_opt.yml --pt_type imagenet
#| tee $LOG

