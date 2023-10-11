import itertools
import logging
from munch import DefaultMunch
from det3d.utils.config_tool import get_downsample_factor
from math import ceil

BATCH_SIZE=16

tasks = [
    dict(num_class=15, class_names=["Pelvis", "Right_Hip", "Right_Knee", "Right_Ankle", "Left_Hip", \
                                    "Left_Knee", "Left_Ankle", "Thomx", "Head", "Left_Shoulder", "Left_Elbow", \
                                    "Left_Wrist", "Right_Shoulder", "Right_Elbow", "Right_Wrist"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

lidar_radar_voxel_ratio = [4, 16, 32] # x, y, z


DATASET = dict(
  DIR=dict(
    ROOT_DIR='/mnt/nas_cruw_pose',
    META_FILE='file_meta.txt',
    KEYPOINT_META='Keypoints_meta.txt',
    CALIB='calib.json',
    LIDAR='lidar_npy'
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
    roi1 = {'z': [-1.8125000000000018,  3.9875], 'y': [-5.0250000000000234, 5.024999999999931], 'x': [0.7703125, 8.0203125]}
  ),
  RDR_TYPE='zyx_real', # 'zyx_real', 'dzyx_real', 'zyx_complex', 'dzyx_complex'
  LIDAR = dict(
      IS_CONSIDER_ROI=True,
      ROI_TYPE='roi1',
      GRID_SIZE=[0.0453125/lidar_radar_voxel_ratio[0], 0.15703125/lidar_radar_voxel_ratio[1], 0.3625/lidar_radar_voxel_ratio[2]], # [m], # [x, y, z]
  ),
  ENABLE_SENSOR=['LIDAR'],
  )


lidar_feature_dim=4
# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=lidar_feature_dim,
    ),
    backbone=dict(
        type="SpMiddleResNetFHD", num_input_features=lidar_feature_dim, ds_factor=1
    ),
    neck=None,
    pose_head=dict(
      type='CenterHead',
      tasks=tasks,
      in_channels=64,
      share_conv_channel=32,
      dataset='cruw_pose',
      weight=1.0,
      code_weights=[1.0, 1.0, 1.0], 
                    # weight of loss from common_heads (key point regression)
      common_heads={'reg': (3, 2)}, # ( 3 (x, y, z), num of conv layers),
      dcn_head=False
    ),
    sensor_type='lidar'
)

dataset_type = "CRUW_POSE_Dataset"

target_assigner = dict(
    tasks=tasks,
)

# out size factor compared to the input voxel size
out_size_factor = [8, 8, 8]
# tensor zyx of shape 64, 128, 80
# radar heatmap zyx of shape  16, 64, 160


assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=out_size_factor, # TODO: check this
    gaussian_overlap=0.1,
    max_poses=1,
    min_radius=1,
    consider_radar_visibility=DATASET['LABEL']['CONSIDER_RADAR_VISIBILITY'],
)
train_cfg = dict(assigner=assigner)
test_cfg_range = DATASET['ROI'][DATASET['LABEL']['ROI_TYPE']]
test_cfg = dict(
    post_center_limit_range=[test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0], \
                             test_cfg_range['x'][1], test_cfg_range['y'][1], test_cfg_range['z'][1]], # [-x, -y, -z, x, y, z] RoI
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
    voxel_size=DATASET['LIDAR']['GRID_SIZE'],
    input_type='lidar_pc'
)


train_preprocessor = dict(
    mode="train",
    pc_type='lidar_pc',
    shuffle_points=True,
    global_translate_std=0.5,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    pc_type='lidar_pc',
    shuffle_points=False,
)

voxel_generator = dict(
    range=(test_cfg_range['x'][0], test_cfg_range['y'][0], test_cfg_range['z'][0], \
                             test_cfg_range['x'][1], test_cfg_range['y'][1], test_cfg_range['z'][1]),
    voxel_size=DATASET['LIDAR']['GRID_SIZE'],
    max_points_in_voxel=5,
    max_voxel_num=[16000, 16000],
)

train_pipeline = [
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabelPose", cfg=train_cfg["assigner"]),
]
test_pipeline = [
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabelPose", cfg=train_cfg["assigner"]),
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

checkpoint_config = dict(interval=2)
# yapf:disable
log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 30
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

cuda_device = '0'