## end2end 15000千次迭代，学习率8000次时减半

## 显存不够，lib/fast_rcnn/config.py中缩放宽度 600设成300

## 训练数据集
- [imagenet n03126707](http://www.image-net.org/synset?wnid=n03126707)
- crane
- 目标在图片占比很高

##结果
- 有一定效果，学到了一些crane的特征
- loss见loss1.jpg

##问题
- 大量电线被错检
- 小目标基本没有被识别(训练集的分布决定)
