# install
## 按照caffe装好依赖库，配好CUDA
## git clone --recursive
## 根据自己的配置修改caffe-fast-rcnn/Makefile.config中未注释内容，若cudnn版本与faster-rcnn要求兼容，可取消注释第5行
## 在lib目录下 执行make
## 在caffe-fast-rcnn目录下 执行make -j   make pycaffe
## 编译完成，即可

# 数据准备
## 训练数据放在data/imagenet目录下
## 图片和标注各一个文件夹，命名为 Images Annotations 

# train 
## 运行detection目录下的end2end.sh，注意根据情况修改其中的参数，如模型文件路径、网络结构路径、迭代次数等，此脚本以end2end模式训练faster_rcnn
## 运行detection目录下的alt_opt.sh，注意同上，此脚本以alt_opt模式训练faster_rcnn，
## 目前看end2end比alternate optimize效果好
## 训练的注意事项：总的迭代次数最好>=10个epoch，根据情况修改运行参数或solver设置或网络结构设置
## 训练得到的模型会保存在output路径下
## 训练时运行脚本注意保存log， 如 sh end2end.sh > log 2&>1
## 需要画loss曲线可以借助detection目录下的plot_loss.py，使用前需将日志中的 包含" loss = "的行提取到一个文件中，可借助grep " loss = " filename > o,也可自己另写脚本

# test
## 图形化的测试，将测试图片置于data/demo路径下，运行tools/demo.py，画好框的测试结果将保存在demo_results路径下
## 计算mAP的测试，由于目前标注数据不足，尚未跑过，仍待开发
