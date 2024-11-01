from pathlib import Path
from collections import defaultdict
import glob
import ndjson
import matplotlib.pyplot as plt


def load_scalars(scalar_path):
    scalar_dict = defaultdict(list)
    last_log_dirs = sorted([fn for fn in scalar_path.glob('*') if fn.is_dir()])
    chosen_dirs = last_log_dirs[-1:]
    for last_log_dir in chosen_dirs:
        scalars_file = last_log_dir / 'vis_data' / 'scalars.json'

        with open(scalars_file, 'r') as f:
            data = ndjson.load(f)
            for elem in data:
                for metric in metrics:
                    if metric in elem:
                        scalar_dict[metric].append(elem[metric])
    scalar_dict['loss'] = scalar_dict['loss'][::20]
    return scalar_dict

ORI_DIR = Path('work_dirs')
AUG_DIR = Path('work_dirs/synthetic_coco')

ori_model= 'faster-rcnn_r50_fpn_2x_coco'
aug_models = ['faster-rcnn_r50_fpn_2x_mdata_coco_stuff_bg_ignore_aug', 'faster-rcnn_r50_fpn_2x_coco_thing_aug']
aug_labels = ['aug_wo_small', 'aug']

metrics  = ['loss', 'coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75', 'coco/bbox_mAP_s', 'coco/bbox_mAP_m', 'coco/bbox_mAP_l']

ori_scalars = load_scalars(ORI_DIR / ori_model)
aug_scalars = [load_scalars(AUG_DIR / p) for p in aug_models]

fig, axes = plt.subplots(len(aug_models), len(metrics))
fig.set_figwidth(25)
fig.set_figheight(5)

for i, aug_model in enumerate(aug_models):
    for j, metric in enumerate(metrics):
        # axes[i][j].set_xlabel('iterations')
        axes[i][j].set_ylabel(metric)
        axes[i][j].plot(ori_scalars[metric], label='ori')
        axes[i][j].plot(aug_scalars[i][metric], label=aug_labels[i])
        axes[i][j].legend()

fig.tight_layout()
fig.savefig('synthetic4.png')