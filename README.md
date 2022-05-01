# TDCUP2022

2022年泰迪杯数据挖掘比赛A题代码，大赛官网 [link](https://www.tipdm.org:10010/#/competition/1481159137780998144/introduce).

赛题任务为农田害虫检测识别，本质为计算机视觉领域的目标检测任务。我们结合Mask R-CNN^1^框架和最新Vision Transformer模型MPViT^2^，设计出适用于农田害虫检测的模型。


**项目关键词**：Detectron2、Mask R-CNN、MPViT、Transformer

## 项目结构
```
TDCUP2022/
|-- datasets/
|   |-- coco/
|   |
|   |-- to_coco.py

|-- mpvit/
|   |-- __init__.py
|   |-- backbone.py
|   |-- config.py
|   |-- dataset_mapper.py
|   |-- mpvit.py
|
|-- output/
|-- pretrained/
|
|-- summmit/
|   |-- test1/
|   |-- test2/
|   |
|   |-- crop.py
|   |-- result2to3.py
|
|-- scripts/
|   |-- train.sh  # entrance for training
|   |-- evaluate.sh
|
|-- train_net.py
|-- predict1.ipynb
|-- predict2.ipynb
|-- requirements.txt
|-- README
```

- datasets：存放数据集及数据处理脚本
    - to_coco.py: 将csv格式的标准转换为coco格式

- mpvit：存放骨干模型
- output：训练输出路径
- pretrained：预训练模型存储路径
- summit
    - test1：测试集1预测结果路径
    - test2：测试集2预测结果路径
    - crop.py: 根据预测bbox分类裁剪图片，用于检查检测是否准确
    - result2to3.py: 根据result2.csv生成result3.csv
- predict1.ipynb: 用于测试集1的预测
- predict2.ipynb: 用于测试集2的预测

## 主要工作

- 数据处理
    - 大赛数据手动筛选
    - labelme手动标注数据
    - IP102数据集（未来得及使用）
- 模型部分
    - 二分类器判断图片是否包含目标：
    模型推理比较耗时，为加快速度，训练一个二分类器判断图像是否包含目标，是则检测；否则跳过。
    - 训练二分类器判断图片是否包含目标，加快测试集推理速度
    - 分类器纠正检测结果：有害虫外表极其相似，mask rcnn分类错误。重新训练一个分类模型，纠正mask rcnn对检测目标的识别结果。
    - 检测器：mask rcnn + mpvit

## 参考文献
[1] He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.

[2] Lee Y, Kim J, Willette J, et al. MPViT: Multi-Path Vision Transformer for Dense Prediction[J]. arXiv preprint arXiv:2112.11010, 2021.


