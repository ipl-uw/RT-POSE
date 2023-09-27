import itertools
import logging
from munch import DefaultMunch

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=1, class_names=["Sedan"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)
        
# model settings
model = dict(
    type="RadarNetSingleStage",
    pretrained=None,
    reader=dict(
        type='RadarFeatureNet',
    ),
    backbone=dict(
        type="HRNet3D",
        backbone_cfg='hr_tiny',
        final_conv_in = 8,
        final_conv_out = 8,
        ds_factor=1
    ),
    # todo: modify neck and head config 
    neck=dict(
        type="RPN",
        layer_nums=[5, 5], # num of layer -1  per blocks
        ds_layer_strides=[1, 2], # strides of blocks
        ds_num_filters=[80, 160], # num of output channels for each block
        us_layer_strides=[1, 2], # upstrides of blocks
        us_num_filters=[160, 160], # num of output channels for each deblock
        num_input_features=192, # 
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([160, 160]), # sum of us_num_filters
        tasks=tasks,
        dataset='kradar',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], # weight of loss from common_heads
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2)}, # (num output feat maps, )
        share_conv_channel=64,
        dcn_head=False
    ),
)

# todo: modify gussian map params

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=30,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

# todo: modify test_cfg
test_cfg = dict(
    post_center_limit_range=[0., -6.4, -2, 71.6, 6., 5.6], # [x_min, -y, -z, x_max, y, z] RoI
    max_per_img=30,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=50, # select first nms_pre_max_size numnber of bbox to do nms
        nms_post_max_size=10, # select nms_post_max_size bbox after nms
        nms_iou_threshold=0.1,
    ),
    score_threshold=0.1,
    pc_range=[0., -6.4],
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.4, 0.4],
    input_type='rdr_cube'
)

# dataset settings
dataset_type = "KRadarDataset"

# kradar data config

### ----- Dataset ----- ###
DATASET = dict(
  TYPE_LOADING='path',
  DIR=dict(
    LIST_DIR=['/mnt/nas_kradar/kradar_dataset/dir_all']
  ),
  PATH_SPLIT={
    'train': 'configs/kradar/resources/split/train_clean.txt',
    'test':  'configs/kradar/resources/split/test.txt',
  },
  TYPE_COORD= 1, # 1: Radar, 2: Lidar, 3: Camera
  LABEL= dict(
    IS_CONSIDER_ROI=True,
    ROI_TYPE='cube', # in ['default', 'cube', 'sparse_cube', 'lpc']
    ROI_DEFAULT=[0,120,-100,100,-50,50], # x_min_max, y_min_max, z_min_max / Dim: [m]
    IS_CHECK_VALID_WITH_AZIMUTH=True,
    MAX_AZIMUTH_DEGREE=[-50, 50],
    TYPE_CHECK_AZIMUTH='center' # in ['center', 'apex']  
  ),
  RDR_CUBE = dict(
      DOPPLER=dict(
        IS_ANOTHER_DIR=True,
        OFFSET=1.9326 
      ),
      IS_COUNT_MINUS_ONE_FOR_BEV=True, # Null value = -1 for pw & -10 for Doppler
      IS_CONSIDER_ROI=True,
      ROI={ # each bin is 0.4 [m] (minimum range resolution)
      'z': [-2, 7.2],     # Dim: [m] / [-2.0, 5.6] for Driving corridor
      'y': [-6.4, 6.], # Dim: [m] / [-6.4, 6.0] for Driving corridor
      'x': [0., 70],     # Dim: [m] / [0.0, 71.6] for Driving corridor, arr_cube: (24, 32, 176)
      },
      GRID_SIZE=0.4, # [m],
      CONSIDER_ROI_ORDER='cube -> num', # in ['cube -> num', 'num -> cube']
      BEV_DIVIDE_WITH='bin_z', # in ['bin_z', 'none_minus_1'],
      NORMALIZING_VALUE=1e+13 # 'fixed' # todo: try other normalization strategies
  ),
  CLASS_INFO=dict(
    # If containing cls, make the ID as number
    # In this case, we consider ['Sedan', 'Bus or Truck'] as Sedan (Car)
    CLASS_ID={
      'Sedan': 1,
      'Bus or Truck': -1,
      'Motorcycle': -1,
      'Bicycle': -1,
      'Bicycle Group': -1,
      'Pedestrian': -1,
      'Pedestrian Group': -1,
      'Background': 0,
    },
    IS_CONSIDER_CLASS_NAME_CHANGE=False,
    CLASS_NAME_CHANGE={
      'Sedan': 'Car',
      # 'Bus or Truck': 'Sedan',
    },
    NUM_CLS= None,# automatically consider, just make this blank (not including background)
    SCALE_SMALL_CLS= 1.5),
  
    Z_OFFSET= 0.7, # Radar to Lidar [m] / prior value = 1.25
    # List of items to be returned by the dataloader
    GET_ITEM= {
      'rdr_sparse_cube'   : False,
      'rdr_tesseract'     : False,
      'rdr_cube'          : True,
      'rdr_cube_doppler'  : False,
      'ldr_pc_64'         : False,
      'cam_front_img'     : False,
    }
  )





train_preprocessor = dict(
)




train_pipeline = [
    dict(type="LoadRadarData"),  
    dict(type="AssignLabelRadar", cfg=train_cfg["assigner"]),
]

# val_preprocessor = dict(
# )
test_pipeline = [
    dict(type="LoadRadarData"),  
    dict(type="AssignLabelRadar", cfg=train_cfg["assigner"]),
]


data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        split='train',
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    # val=dict(
    #     type=dataset_type,
    #     root_path=data_root,
    #     class_names=class_names,
    #     pipeline=test_pipeline,
    # ),
    test=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        split='test', # todo: change
        class_names=class_names,
        pipeline=train_pipeline,
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

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(1)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]


# ROI: { # each bin is 0.4 [m] (minimum range resolution)
#     'z': [-2, 6.0],   # Dim: [m] / [-2.0, 6.0] for Driving corridor / None (erase)
#     'y': [-6.4, 6.4], # Dim: [m] / [-6.4, 6.4] for Driving corridor
#     'x': [0, 72.0],   # Dim: [m] / [0.0, 72.0] for Driving corridor
# } # Cartesian (+ 0.4m from setting of RDR_CUBE: Consider this as LPC)


if __name__ == '__main__':
  # test munch libarary to convert dict to object
  ds_cfg = DefaultMunch.fromDict(DATASET)
  print(ds_cfg)