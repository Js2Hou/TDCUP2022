#!/usr/bin

python train_net.py --config-file configs/maskrcnn/mask_rcnn_mpvit_small_ms_3x.yaml  \
     --eval-only  --num-gpus 1 MODEL.WEIGHTS pretrained/mask_rcnn_mpvit_small_ms_3x.pth