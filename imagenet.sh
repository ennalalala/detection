python tools/train_imagenet.py \
    --gpu 0 \
    --solver ./models/imagenet/VGG16/faster_rcnn_end2end/solver.prototxt \
    --weights data/imagenet_models/VGG16.v2.caffemodel \
    --imdb imagenet \
    --iters 15000 \
    --cfg ./experiments/cfgs/faster_rcnn_end2end.yml
