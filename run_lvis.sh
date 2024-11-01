bash tools/dist_test.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_lvis.py \
    checkpoints/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1-aa78ac3d.pth \
    8

bash tools/dist_train.sh \
    configs/mask_rcnn/mask-rcnn_r50_fpn_1x_lvis.py 8

bash tools/dist_train.sh \
    configs/faster_rcnn/faster-rcnn_r50_fpn_2x_lvis.py 8