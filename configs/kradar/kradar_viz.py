import itertools

tasks = [
  dict(num_class=2, class_names=["Sedan", "BusorTruck"]),
]
class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# kradar data config
dataset_type = "KRadarDataset"

### ----- Dataset ----- ###
DATASET = dict(
  TYPE_LOADING='path',
  DIR=dict(
    LIST_DIR=['/mnt/nas_kradar/kradar_dataset/dir_all']
  ),
  PATH_SPLIT={
    'train': 'configs/kradar/resources/split/train.txt',
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
      'z': [-2, 6.],     # Dim: [m] / [-2.0, 5.6] for Driving corridor
      'y': [-20., 20.], # Dim: [m] / [-6.4, 6.0] for Driving corridor
      'x': [0., 80.],     # Dim: [m] / [0.0, 71.6] for Driving corridssor
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
      'Bus or Truck': 1,
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
    },
    LIST_BAD = [] # seq not used in training
  )

data = dict(
    train=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        split='train',
        class_names=class_names,
        pipeline=[],
    ),
    test=dict(
        type=dataset_type,
        cfg=dict(DATASET=DATASET),
        split='test', # todo: change
        class_names=class_names,
        pipeline=[],
    ),
)

log_level = "INFO"