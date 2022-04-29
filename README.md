# TDCUP2022

2022年泰迪杯数据挖掘比赛A题代码，大赛官网 [link](https://www.tipdm.org:10010/#/competition/1481159137780998144/introduce).

赛题任务为农田害虫检测识别，本质为计算机视觉领域的目标检测任务。我们结合Mask R-CNN^1^框架和最新Vision Transformer模型MPViT^2^，设计出适用于农田害虫检测的模型。


项目关键词：Detectron2、Mask R-CNN、MPViT、Transformer

## 项目结构
```
TDCUP2022/
|-- datasets/
|   |-- coco/
|   |
|   |-- to_coco.py
|
|-- mpvit/
|   |-- __init__.py
|   |-- backbone.py
|   |-- config.py
|   |-- dataset_mapper.py
|   |-- mpvit.py
|
|-- scripts/
|   |-- train.sh  # entrance for training
|   |-- evaluate.sh
|
|-- train_net.py
|-- predict.ipynb
|-- requirements.txt
|-- README
```

## 参考文献
[1] He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.

[2] Lee Y, Kim J, Willette J, et al. MPViT: Multi-Path Vision Transformer for Dense Prediction[J]. arXiv preprint arXiv:2112.11010, 2021.


