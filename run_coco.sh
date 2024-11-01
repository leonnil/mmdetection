bash tools/dist_test.sh \
    configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py \
    checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    8

bash tools/dist_test.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8

bash tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50_fpn_2x_coco.py 8

bash tools/dist_train.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 8

bash tools/dist_train.sh \
    configs/dino/dino-4scale_r50_8xb2-24e_coco.py 8