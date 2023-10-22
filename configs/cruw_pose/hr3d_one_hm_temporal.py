import itertools
import logging
from munch import DefaultMunch
from det3d.utils.config_tool import get_downsample_factor
from math import ceil
import numpy as np

BATCH_SIZE=16

tasks = [
    dict(num_class=1, class_names=["Pelvis"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

DATASET = dict(
  DIR=dict(
    ROOT_DIR='/mnt/nas_cruw_pose',
    META_FILE='file_meta.txt',
    KEYPOINT_META='Keypoints_meta.txt',
  ),
  LABEL= dict(
    IS_CONSIDER_ROI=True,
    ROI_TYPE='roi1',
    ROI_DEFAULT=[], # x_min_max, y_min_max, z_min_max / Dim: [m]
    IS_CHECK_VALID_WITH_AZIMUTH=False,
    MAX_AZIMUTH_DEGREE=[-50, 50],
    CONSIDER_RADAR_VISIBILITY=False,
  ),
  ROI = dict(
    roi1 = {'z': [-1.0875000000000021,  4.7125], 'y': [-5.0250000000000234, 5.024999999999931], 'x': [0.7703125, 8.0203125]}
  ),
  RDR_TYPE='zyx_real', # 'zyx_real', 'dzyx_real', 'zyx_complex', 'dzyx_complex'
  RDR_CUBE = dict(
      IS_CONSIDER_ROI=True,
      ROI_TYPE='roi1',
      # tensor zyx of shape 16, 64, 160
      GRID_SIZE=[0.0453125, 0.15703125, 0.3625], # [m], # [x, y, z]
      NORMALIZING_VALUE=(150000, 200000),

  ),
  DZYX = dict(
    REDUCE_TYPE='none', # 'none', 'avg', 'max'
    IS_CONSIDER_ROI=True,
    GRID_SIZE=[0.0453125, 0.15703125, 0.3625], # [m],
    NORMALIZING_VALUE=(100000, 9000000),
  ),
  ENABLE_SENSOR=['RADAR'],
  )

hr_final_conv_out = 128

# model settings
model = dict(
    type="RadarPoseNet",
    pretrained=None,
    reader=dict(
        type='RadarFeatureNet',
    ),
    backbone=dict(
        type="HRNet3D",
        backbone_cfg='hr_tiny_feat32_zyx_l4',
        final_conv_in = sum([32, 32, 64, 64]),
        final_conv_out = hr_final_conv_out,
        final_fuse = 'conat_conv',
        ds_factor=1,
    ),
    pose_head=dict(
      type='CenterHead',
      tasks=tasks,
      in_channels=hr_final_conv_out,
      share_conv_channel=128,
      dataset='cruw_pose',
      weight=0.1,
      code_weights=np.ones(45).tolist(), 
                    # weight of loss from common_heads (key point regression)
      common_heads={'reg': (45, 2)}, # ( 45  15 keypoints' (x, y, z), num of conv layers),
      dcn_head=False
    ),
    neck=None,
)






# dataset settings
dataset_type = "CRUW_POSE_Dataset"

# kradar data config


# todo: modify gussian map params

target_assigner = dict(
    tasks=tasks,
)

out_size_factor = [1, 1, 1]

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=out_size_factor, # TODO: check this
    gaussian_overlap=0.1,
    max_poses=1,
    min_radius=2,
    consider_radar_visibility=DATASET['LABEL']['CONSIDER_RADAR_VISIBILITY'],
)

train_cfg = dict(assigner=assigner)

test_cfg_range = DATASET['ROI'][DATASET['LABEL']['ROI_TYPE']]
test_cfg = dict(
    post_center_limit_range=[test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0], \
                             test_cfg_range['x'][1], test_cfg_range['y'][1], test_cfg_range['z'][1]], # [x_min, -y, -z, x_max, y, z] RoI
    circular_nms=True,
    nms=dict(
        use_rotate_nms=False,
        use_multi_class_nms=False,
        nms_pre_max_size=1, # select first nms_pre_max_size numnber of bbox to do nms
        nms_post_max_size=1, # select nms_post_max_size bbox after nms
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.0,
    pc_range=[test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0]],
    out_size_factor=out_size_factor,#get_downsample_factor(model)
    voxel_size=[0.0453125, 0.15703125, 0.3625],
    input_type='rdr_cube'
)



train_pipeline = [
    dict(type="AssignLabelPose2", cfg=train_cfg["assigner"]),
]

test_pipeline = [
    dict(type="AssignLabelPose2", cfg=train_cfg["assigner"]),
]


data = dict(
    samples_per_gpu=BATCH_SIZE,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Train.json',
        pipeline=train_pipeline,
        class_names=class_names,

    ),
    test=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Test.json',
        pipeline=test_pipeline,
        class_names=class_names,

    ),
    val=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        label_file='Train.json',
        pipeline=test_pipeline,
        class_names=class_names,

    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=5)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 100
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

cuda_device = '0'