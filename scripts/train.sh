#!/bin/bash

config1=configs/maskrcnn/mask_rcnn_mpvit_small_ms_3x.yaml
config2=configs/maskrcnn/mask_rcnn_mpvit_base_ms_3x.yaml

python train_net.py --tag exp2-eval --num-gpus 2 \
    --config-file  ${config2}\
    SOLVER.IMS_PER_BATCH 2 SOLVER.BASE_LR 0.0001 MODEL.ROI_HEADS.NUM_CLASSES 28\
    SOLVER.MAX_ITER 1 SOLVER.CHECKPOINT_PERIOD 1 TEST.EVAL_PERIOD 1 \
    MODEL.WEIGHTS output/mask_rcnn_mpvit_base_ms_3x/exp2/model_0019999.pth