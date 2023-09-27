import os
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm
import numpy as np
from det3d.datasets.registry import DATASETS
from det3d.datasets.pipelines import Compose
from munch import DefaultMunch
import collections

@DATASETS.register_module
class KRadarDataset_OLD(Dataset):
    def __init__(self, cfg, split, class_names=None, pipeline=None):
        super().__init__()
        cfg = DefaultMunch.fromDict(cfg) # if cfg is dict
        self.class_names = class_names
        self.cfg = cfg
        self.cfg.update(class_names=class_names)
        if split == 'train':
            self.LIST_BAD = self.cfg.DATASET.LIST_BAD
        else:
            self.LIST_BAD = []
        ### Load label paths wrt split ###
        # load label paths
        self.split = split # in ['train', 'test']
        self.dict_split = self.get_split_dict(dict(self.cfg.DATASET.PATH_SPLIT)[self.split])
        self.list_path_label = [] # a list of dic
        for dir_seq in self.cfg.DATASET.DIR.LIST_DIR:
            list_seq = set(os.listdir(dir_seq))
            list_seq = list(list_seq.difference(set(self.LIST_BAD)))
            for seq in list_seq:
                seq_label_paths = sorted(glob(osp.join(dir_seq, seq, 'info_label', '*.txt')))
                seq_label_paths = list(filter(lambda x: (x.split('/')[-1].split('.')[0] in self.dict_split[seq]), seq_label_paths))
                self.list_path_label.extend(seq_label_paths)
        ### Load label paths wrt split ###
        
        ### Type of data for __getitem__ ###
        if self.cfg.DATASET.TYPE_LOADING == 'dict':
            self.type_item = 0
        elif self.cfg.DATASET.TYPE_LOADING == 'path':
            self.type_item = 1
        else:
            print('* Exception error (Dataset): check DATASET.TYPE_LOADING')
            
            
        ### Class info ###
        self.dict_cls_id = self.cfg.DATASET.CLASS_INFO.CLASS_ID
        self.cfg.DATASET.CLASS_INFO.NUM_CLS = len(list(set(list(self.dict_cls_id.values())).difference(set([0,-1]))))
        self.is_single_cls = True if self.cfg.DATASET.CLASS_INFO.NUM_CLS == 1 else False
        try:
            self.scale_small_cls = self.cfg.DATASET.CLASS_INFO.SCALE_SMALL_CLS
        except:
            self.scale_small_cls = 1.5
            print('* Exception error (Dataset): check DATASET.CLASS_INFO.SCALE_SMALL_CLS')
        self.is_consider_cls_name_change = self.cfg.DATASET.CLASS_INFO.IS_CONSIDER_CLASS_NAME_CHANGE
        if self.is_consider_cls_name_change:
            self.dict_cls_name_change = self.cfg.DATASET.CLASS_INFO.CLASS_NAME_CHANGE
        ### Class info ###

        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###
        self.type_coord = self.cfg.DATASET.TYPE_COORD # 1: Radar, 2: Lidar, 3: Camera
        ### Selecting Radar/Lidar/Camera (Unification of coordinate systems) ###

        ### Radar Sparse Cube ###
        self.is_get_sparse_cube = False
        if self.cfg.DATASET.GET_ITEM.rdr_sparse_cube:
            self.is_get_sparse_cube = True
            self.name_sp_cube = self.cfg.DATASET.RDR_SP_CUBE.NAME_RDR_SP_CUBE
            self.is_sp_another_dir = self.cfg.DATASET.RDR_SP_CUBE.IS_ANOTHER_DIR
            self.dir_sp = self.cfg.DATASET.DIR.DIR_SPARSE_CB
        ### Radar Sparse Cube ###

        ### Radar Tesseract###
        if self.cfg.DATASET.GET_ITEM.rdr_tesseract:
            # load physical values
            self.arr_range, self.arr_azimuth, self.arr_elevation = self.load_physical_values()
            # consider roi
            self.is_consider_roi_rdr = cfg.DATASET.RDR_TESSERACT.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr:
                self.consider_roi_tesseract(cfg.DATASET.RDR_TESSERACT.ROI)
        ### Radar Tesseract (Currently not used: TBD) ###

        ### Radar Cube ###
        self.is_get_cube_dop = False 
        if self.cfg.DATASET.GET_ITEM.rdr_cube:
            # dealing cube data
            # _, _, _, self.arr_doppler = self.load_physical_values(is_with_doppler=True)
            # To make BEV -> averaging power
            self.is_count_minus_1_for_bev = cfg.DATASET.RDR_CUBE.IS_COUNT_MINUS_ONE_FOR_BEV

            # Default ROI for CB (When generating CB from matlab applying interpolation)
            self.arr_bev_none_minus_1 = None
            self.arr_z_cb = np.arange(-30, 30, 0.4)
            self.arr_y_cb = np.arange(-80, 80, 0.4)
            self.arr_x_cb = np.arange(0, 100, 0.4)

            self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(cfg.DATASET.RDR_CUBE.ROI)
                if cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'cube -> num':
                    self.consider_roi_order = 1
                elif cfg.DATASET.RDR_CUBE.CONSIDER_ROI_ORDER == 'num -> cube':
                    self.consider_roi_order = 2
                else:
                    raise AttributeError('Check consider roi order in cfg')
                if cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'bin_z':
                    self.bev_divide_with = 1
                elif cfg.DATASET.RDR_CUBE.BEV_DIVIDE_WITH == 'none_minus_1':
                    self.bev_divide_with = 2
                else:
                    raise AttributeError('Check consider bev divide with in cfg')
            self.is_get_cube_dop = cfg.DATASET.GET_ITEM['rdr_cube_doppler']
            self.offset_doppler = cfg.DATASET.RDR_CUBE.DOPPLER.OFFSET
            self.is_dop_another_dir = cfg.DATASET.RDR_CUBE.DOPPLER.IS_ANOTHER_DIR
            self.dir_dop = cfg.DATASET.DIR.DIR_DOPPLER_CB
        ### Radar Cube (What we primarily utilize) ###

        ### Label ###
        self.roi_label = self.cfg.DATASET.LABEL.ROI_DEFAULT
        if self.cfg.DATASET.LABEL.IS_CONSIDER_ROI:
            if self.cfg.DATASET.LABEL.ROI_TYPE == 'sparse_cube':
                x_roi, y_roi, z_roi = cfg.DATASET.RDR_SP_CUBE.ROI.x, \
                                      cfg.DATASET.RDR_SP_CUBE.ROI.y, \
                                      cfg.DATASET.RDR_SP_CUBE.ROI.z
            elif self.cfg.DATASET.LABEL.ROI_TYPE == 'cube':
                x_roi, y_roi, z_roi = cfg.DATASET.RDR_CUBE.ROI.x, \
                                      cfg.DATASET.RDR_CUBE.ROI.y, \
                                      cfg.DATASET.RDR_CUBE.ROI.z
            elif self.cfg.DATASET.LABEL.ROI_TYPE == 'lpc':
                x_roi, y_roi, z_roi = cfg.DATASET.LPC.ROI.x, \
                                      cfg.DATASET.LPC.ROI.y, \
                                      cfg.DATASET.LPC.ROI.z
            else:
                x_roi, y_roi, z_roi = cfg.DATASET.ROI[cfg.DATASET.LABEL.ROI_TYPE].x, \
                            cfg.DATASET.ROI[cfg.DATASET.LABEL.ROI_TYPE].y, \
                            cfg.DATASET.ROI[cfg.DATASET.LABEL.ROI_TYPE].z    
            x_min, x_max = [0, 150]     if x_roi is None else x_roi
            y_min, y_max = [-160, 160]  if y_roi is None else y_roi
            z_min, z_max = [-150, 150]  if z_roi is None else z_roi
            self.roi_label = [x_min, y_min, z_min, x_max, y_max, z_max]
        else:
            print('* Exception error (Dataset): check DATASET.LABEL.ROI_TYPE')
            pass # using default
        self.is_roi_check_with_azimuth = self.cfg.DATASET.LABEL.IS_CHECK_VALID_WITH_AZIMUTH
        self.max_azimtuth_rad = self.cfg.DATASET.LABEL.MAX_AZIMUTH_DEGREE
        self.max_azimtuth_rad = [self.max_azimtuth_rad[0]*np.pi/180., self.max_azimtuth_rad[1]*np.pi/180.]
        if self.cfg.DATASET.LABEL.TYPE_CHECK_AZIMUTH == 'center':
            self.type_check_azimuth = 0
        elif self.cfg.DATASET.LABEL.TYPE_CHECK_AZIMUTH == 'apex':
            self.type_check_azimuth = 1
        # print(self.max_azimtuth_rad)
        ### Label ###

        ### V2: Load dictionary with paths (Saving interval of loading data) ###
        self.list_dict_item = []

        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)

    ### General functions ###
    def get_split_dict(self, path_split):
        # ./tools/train_test_splitter
        f = open(path_split, 'r')
        lines = f.readlines()
        f.close
        dict_seq = dict()
        for line in lines:
            seq = line.split(',')[0]
            label = line.split(',')[1].split('.')[0]
            if not (seq in list(dict_seq.keys())):
                dict_seq[seq] = []
            dict_seq[seq].append(label)
        return dict_seq

    def load_physical_values(self, is_in_rad=True, is_with_doppler=False):
        temp_values = loadmat('./resources/info_arr.mat')
        arr_range = temp_values['arrRange']
        if is_in_rad:
            deg2rad = np.pi/180.
            arr_azimuth = temp_values['arrAzimuth']*deg2rad
            arr_elevation = temp_values['arrElevation']*deg2rad
        else:
            arr_azimuth = temp_values['arrAzimuth']
            arr_elevation = temp_values['arrElevation']
        _, num_0 = arr_range.shape
        _, num_1 = arr_azimuth.shape
        _, num_2 = arr_elevation.shape
        arr_range = arr_range.reshape((num_0,))
        arr_azimuth = arr_azimuth.reshape((num_1,))
        arr_elevation = arr_elevation.reshape((num_2,))
        if is_with_doppler:
            arr_doppler = loadmat('./resources/arr_doppler.mat')['arr_doppler']
            _, num_3 = arr_doppler.shape
            arr_doppler = arr_doppler.reshape((num_3,))
            return arr_range, arr_azimuth, arr_elevation, arr_doppler
        else:
            return arr_range, arr_azimuth, arr_elevation

    def consider_roi_tesseract(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = [0, len(self.arr_range)-1, \
            0, len(self.arr_azimuth)-1, 0, len(self.arr_elevation)-1]

        idx_attr = 0
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi

        for k, v in roi_polar.items():
            if v is not None:
                min_max = (np.array(v)*deg2rad).tolist() if idx_attr > 0 else v
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx[idx_attr*2] = idx_min
                self.list_roi_idx[idx_attr*2+1] = idx_max
                
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  (np.array(v_new)*rad2deg) if idx_attr > 0 else v_new
                    self.cfg.DATASET.RDR.RDR_POLAR_ROI[k] = v_new
            idx_attr += 1


    def consider_roi_cube(self, roi_cart, is_reflect_to_cfg=True):
        # to get indices
        self.list_roi_idx_cb = [0, len(self.arr_z_cb)-1, \
            0, len(self.arr_y_cb)-1, 0, len(self.arr_x_cb)-1]
        idx_attr = 0
        for k, v in roi_cart.items():
            if v is not None:
                min_max = np.array(v).tolist()
                # print(min_max)
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}_cb'), min_max)
                setattr(self, f'arr_{k}_cb', arr_roi)
                self.list_roi_idx_cb[idx_attr*2] = idx_min
                self.list_roi_idx_cb[idx_attr*2+1] = idx_max
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new = np.array(v_new)
                    self.cfg.DATASET.RDR_CUBE.ROI[k] = v_new
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        return arr[idx_min:idx_max+1], idx_min, idx_max
    ### General functions ###

    ### Loading values from txt ###
    def get_calib_info(self, path_calib, is_z_offset_from_cfg=True):
        '''
        * return: [X, Y, Z]
        * if you want to get frame difference, get list_calib[0]
        '''
        if not (self.type_coord == 1): # not Rdr coordinate (labels are represented on Ldr coordinate)
            return None
        else:
            with open(path_calib) as f:
                lines = f.readlines()
                f.close()
            try:
                list_calib = list(map(lambda x: float(x), lines[1].split(',')))
                # list_calib[0] # frame difference
                list_values = [list_calib[1], list_calib[2]] # X, Y
                if is_z_offset_from_cfg:
                    list_values.append(self.cfg.DATASET.Z_OFFSET) # Z
                else:
                    list_values.append(0.)
                return np.array(list_values)
            except:
                print('* Exception error (Datatset): no calib info')

    # V2: Load this on cpu
    def get_tuple_object(self, line, calib_info, is_heading_in_rad=True, path_label=None):
        '''
        * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
        * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
        * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
        *       None if idx_cls == -1 or header != '*'
        '''
        list_values = line.split(',')

        if list_values[0] != '*':
            return None

        offset = 0
        if(len(list_values)) == 11:
            # print('* Exception error (Dataset): length of values is 11')
            offset = 1
        else:
            print('* Exception error (Dataset): length of values is 10')
            print(path_label)

        cls_name = list_values[2+offset][1:]

        idx_cls = self.dict_cls_id[cls_name]

        if idx_cls == -1: # Considering None as -1
            return None

        if self.is_consider_cls_name_change:
            if cls_name in self.dict_cls_name_change.keys():
                cls_name = self.dict_cls_name_change[cls_name]

        idx_obj = int(list_values[1+offset])
        x = float(list_values[3+offset])
        y = float(list_values[4+offset])
        z = float(list_values[5+offset])
        theta = float(list_values[6+offset])
        if is_heading_in_rad:
            theta = theta*np.pi/180.
        l = 2*float(list_values[7+offset])
        w = 2*float(list_values[8+offset])
        h = 2*float(list_values[9+offset])

        if self.type_coord == 1: # Radar coordinate
            # print('calib_info = ', calib_info)
            x = x+calib_info[0]
            y = y+calib_info[1]
            z = z+calib_info[2]

        if self.is_single_cls:
            pass
        else:
            if cls_name == 'Pedestrian':
                l = l*self.scale_small_cls
                w = w*self.scale_small_cls
                h = h*self.scale_small_cls

        # Check if the label is in roi (For Radar, checking azimuth angle)
        # print('* x, y, z: ', x, y, z)
        # print('* roi_label: ', self.roi_label)
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if ((x > x_min) and (x < x_max) and \
            (y > y_min) and (y < y_max) and \
            (z > z_min) and (z < z_max)):
            # print('* debug 1')

            ### RDR: Check azimuth angle if it is valid ###
            if self.is_roi_check_with_azimuth:
                min_azi, max_azi = self.max_azimtuth_rad
                # print('* min, max: ', min_azi, max_azi)
                azimuth_center = np.arctan2(y, x)
                if (azimuth_center < min_azi) or (azimuth_center > max_azi):
                    # print(f'* debug 2-1, azimuth = {azimuth_center}')
                    return None
            ### RDR: Check azimuth angle if it is valid ###

            # print('* debug 3')
            return (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        else:
            # print('* debug 4')
            return None

    def get_label_bboxes(self, path_label, calib_info):
        with open(path_label, 'r') as f:
            lines = f.readlines()
            f.close()
        # print('* lines : ', lines)
        line_objects = lines[1:]
        # print('* line_objs: ', line_objects)
        list_objects = []

        # print(dict_temp['meta']['path_label'])
        for line in line_objects:
            temp_tuple = self.get_tuple_object(line, calib_info, path_label=path_label)
            if temp_tuple is not None:
                obj = {
                    "obj_id": temp_tuple[-1],
                    "obj_type": temp_tuple[0],
                    "euler": [
                    0.,
                    0.,
                    temp_tuple[2][3]
                    ],
                    "xyz": temp_tuple[2][:3],
                    "lwh": temp_tuple[2][4:]
                }
                list_objects.append(obj)

        return list_objects

    def get_spcube(self, path_spcube):
        return np.load(path_spcube)

    def get_tesseract(self, path_tesseract, is_in_DRAE=True, is_in_3d=False, is_in_log=False):
        # Otherwise you make the input as 4D, you should not get the data as log scale
        arr_tesseract = np.load(path_tesseract)

        ### considering ROI ###
        # if self.is_consider_roi_rdr:
        #     # print(self.list_roi_idx)
        #     idx_r_0, idx_r_1, idx_a_0, idx_a_1, \
        #         idx_e_0, idx_e_1 = self.list_roi_idx
        #     # Python slicing grammar (+1)
        #     arr_tesseract = arr_tesseract[:,idx_r_0:idx_r_1+1,\
        #         idx_a_0:idx_a_1+1,idx_e_0:idx_e_1+1]
        ### considering ROI ###

        # Dimension reduction -> log operation


        return arr_tesseract

    def get_cube(self, path_cube, is_in_log=False, mode=0):
        '''
        * mode 0: arr_cube, mask, cnt
        * mode 1: arr_cube
        '''
        arr_cube = np.flip(np.load(path_cube), axis=0).astype(np.float32) # z-axis is flipped
        # arr_cube = np.flip(loadmat(path_cube)['arr_zyx'], axis=0).astype(np.float32) # z-axis is flipped
        norm_val = float(self.cfg.DATASET.RDR_CUBE.NORMALIZING_VALUE)
        

        # print(arr_cube.shape)
        # print(np.count_nonzero(arr_cube==-1.))

        # RoI selection
        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 1):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        
        # print(arr_cube.shape)
        
        if self.is_count_minus_1_for_bev:
            bin_z = len(self.arr_z_cb)
            if self.bev_divide_with == 1:
                bin_y = len(self.arr_y_cb)
                bin_x = len(self.arr_x_cb)
                # print(bin_z, bin_y, bin_x)
                arr_bev_none_minus_1 = np.full((bin_y, bin_x), bin_z)
            elif self.bev_divide_with == 2:
                arr_bev_none_minus_1 = bin_z-np.count_nonzero(arr_cube==-1., axis=0)
                arr_bev_none_minus_1 = np.maximum(arr_bev_none_minus_1, 1) # evade divide 0
            # print('* max: ', np.max(arr_bev_none_minus_1))
            # print('* min: ', np.min(arr_bev_none_minus_1))

        # print(arr_bev_none_minus_1.shape)

        if (self.is_consider_roi_rdr_cb) & (self.consider_roi_order == 2):
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            # print(idx_z_min, idx_z_max)
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
            if self.is_count_minus_1_for_bev:
                arr_bev_none_minus_1 = arr_bev_none_minus_1[idx_y_min:idx_y_max+1, idx_x_min:idx_x_max+1]

        # print(arr_bev_none_minus_1.shape)

        if is_in_log:
            arr_cube[np.where(arr_cube==-1.)]= 1.
            # arr_cube = np.maximum(arr_cube, 1.) # get rid of -1 before log
            arr_cube = 10*np.log10(arr_cube)
        else:
            arr_cube = np.maximum(arr_cube, 0.)

        none_zero_mask = np.nonzero(arr_cube)

        # print(arr_cube.shape)
        
        # normalize
        arr_cube = arr_cube / norm_val
        

        if mode == 0:
            return arr_cube, none_zero_mask, arr_bev_none_minus_1
        elif mode == 1:
            return arr_cube
        
    def get_cube_direct(self, path_cube):
        '''
        get the preprocessed cube and return it.
        '''
        arr_cube = np.load(path_cube).astype(np.float32)
        return arr_cube



    def get_cube_doppler(self, path_cube_doppler, dummy_value=100.):
        arr_cube = np.flip(loadmat(path_cube_doppler)['arr_zyx'], axis=0)
        # print(np.count_nonzero(arr_cube==-1.)) # no value -1. in doppler cube

        ### Null value is -10. for Doppler & -1. for pw (from matlab) ###
        arr_cube[np.where(arr_cube==-10.)] = dummy_value

        if self.is_consider_roi_rdr_cb:
            idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
            arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]

        arr_cube = arr_cube + self.offset_doppler # to send negative value as tensor

        return arr_cube

    def get_pc_lidar(self, path_lidar, calib_info=None):
        pc_lidar = []
        with open(path_lidar, 'r') as f:
            lines = [line.rstrip('\n') for line in f][13:]
            pc_lidar = [point.split() for point in lines]
            f.close()
        pc_lidar = np.array(pc_lidar, dtype = float).reshape(-1, 9)[:, :4]
        # 0.01: filter out missing values
        pc_lidar = pc_lidar[np.where(pc_lidar[:, 0] > 0.01)].reshape(-1, 4)

        if self.type_coord == 1: # Rdr coordinate
            if calib_info is None:
                raise AttributeError('* Exception error (Dataset): Insert calib info!')
            else:
                pc_lidar = np.array(list(map(lambda x: \
                    [x[0]+calib_info[0], x[1]+calib_info[1], x[2]+calib_info[2], x[3]],\
                    pc_lidar.tolist())))

        return pc_lidar
    
    def get_description(self, path_desc):
        # ./tools/tag_generator
        try:
            f = open(path_desc)
            line = f.readline()
            road_type, capture_time, climate = line.split(',')
            dict_desc = {
                'capture_time': capture_time,
                'road_type': road_type,
                'climate': climate,
            }
            f.close()
        except:
            raise FileNotFoundError(f'* Exception error (Dataset): check description {path_desc}')
        
        return dict_desc


    def __len__(self):
        if self.type_item == 0: # dict
            return len(self.list_dict_item)
        elif self.type_item == 1: # path
            return len(self.list_path_label)
        else:
            print('* Exception error (Dataset): check DATASET.TYPE_LOADING')

    def get_data_indices(self, path_label):
        f = open(path_label, 'r')
        line = f.readlines()[0]
        f.close()

        seq_id = path_label.split('/')[-3]
        rdr_idx, ldr_idx, camf_idx, _, _ = line.split(',')[0].split('=')[1].split('_')

        return seq_id, rdr_idx, ldr_idx, camf_idx

    def get_path_data_from_path_label(self, path_label):
        seq_id, radar_idx, lidar_idx, camf_idx = self.get_data_indices(path_label)
        path_header = path_label.split('/')[:-2]

        ### Sparse tensor
        path_radar_sparse_cube = None
        if self.is_get_sparse_cube:
            if self.is_sp_another_dir:
                path_radar_sparse_cube = os.path.join(self.dir_sp, path_header[-1], self.name_sp_cube, 'spcube_'+radar_idx+'.npy')
            else:
                path_radar_sparse_cube = '/'+os.path.join(*path_header, self.name_sp_cube, 'spcube_'+radar_idx+'.npy')

        ### Folders in seq.zip file
        path_radar_tesseract = '/'+os.path.join(*path_header, 'radar_tesseract_DEAR_npy', 'tesseract_'+radar_idx+'.npy')
        # path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube', 'cube_'+radar_idx+'.mat')
        path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube_npy_f32', 'cube_'+radar_idx+'.npy')
        # path_radar_cube = '/'+os.path.join(*path_header, 'radar_zyx_cube_npy_roi', 'cube_'+radar_idx+'.npy')

        path_lidar_pc_64 = '/'+os.path.join(*path_header, 'os2-64', 'os2-64_'+lidar_idx+'.pcd')
        path_cam_front = '/'+os.path.join(*path_header, 'cam-front', 'cam-front_'+camf_idx+'.png')
        path_calib = '/'+os.path.join(*path_header, 'info_calib', 'calib_radar_lidar.txt')
        path_desc = '/'+os.path.join(*path_header, 'description.txt')

        ### In different folder
        path_radar_cube_doppler = None
        if self.is_get_cube_dop:
            if self.is_dop_another_dir:
                path_radar_cube_doppler = os.path.join(self.dir_dop, path_header[-1], 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')
            else:
                path_radar_cube_doppler = '/'+os.path.join(*path_header, 'radar_cube_doppler', 'radar_cube_doppler_'+radar_idx+'.mat')

        ### Currently not used
        # path_radar_bev_img = '/'+os.path.join(*path_header, 'radar_bev_image', 'radar_bev_100_'+radar_idx+'.png')
        # path_lidar_bev_img = '/'+os.path.join(*path_header, 'lidar_bev_image', 'lidar_bev_100_'+lidar_idx+'.png')
        # path_lidar_pc_128 = '/'+os.path.join(*path_header, 'os1-128', 'os1-128_'+lidar_idx+'.pcd')

        dict_path = {
            'rdr_sparse_cube'   : path_radar_sparse_cube,
            'rdr_tesseract'     : path_radar_tesseract,
            'rdr_cube'          : path_radar_cube,
            'rdr_cube_doppler'  : path_radar_cube_doppler,
            'ldr_pc_64'         : path_lidar_pc_64,
            'cam_front_img'     : path_cam_front,
            'path_calib'        : path_calib,
            'path_desc'         : path_desc,
            'path_label'        : path_label,
        }

        return dict_path
    
    def __getitem__(self, idx):
        path_label = self.list_path_label[idx]
        dict_item = dict()
        dict_item['meta'] = dict()
        dict_path = self.get_path_data_from_path_label(path_label)
        dict_item['meta']['path'] = dict_path
        dict_item[f'meta']['desc'] = self.get_description(dict_path['path_desc'])
        calib_info = self.get_calib_info(dict_path['path_calib'])
        dict_item['calib'] = calib_info
        dict_item['objs'] = self.get_label_bboxes(path_label, calib_info)
        # try:
        ### Get only required data ###
        dict_path = dict_item['meta']['path']
        if self.cfg.DATASET.GET_ITEM['rdr_sparse_cube']:
            dict_item['rdr_sparse_cube'] = self.get_spcube(dict_path['rdr_sparse_cube'])
        if self.cfg.DATASET.GET_ITEM['rdr_tesseract']:
            dict_item['rdr_tesseract'] = self.get_tesseract(dict_path['rdr_tesseract'])
        if self.cfg.DATASET.GET_ITEM['rdr_cube']:
            rdr_cube = self.get_cube(dict_path['rdr_cube'], mode=1)
            # rdr_cube = self.get_cube_direct(dict_path['rdr_cube'])
            dict_item['rdr_cube'] = rdr_cube
        if self.cfg.DATASET.GET_ITEM['rdr_cube_doppler']:
            dict_item['rdr_cube_doppler'] = self.get_cube_doppler(dict_path['rdr_cube_doppler'])
        if self.cfg.DATASET.GET_ITEM['ldr_pc_64']:
            dict_item['ldr_pc_64'] = self.get_pc_lidar(dict_path['ldr_pc_64'], dict_item['calib'])
        ### Get only required data ###
        dict_item.update(mode=self.split)
        dict_item, _ = self.pipeline(dict_item, info=self.cfg)
        return dict_item
        # except:
        #     print('* Exception error (Dataset): __getitem__ error')
        #     return None
        
    def evaluation(self, detections, output_dir=None, testset=False):
        # TODO: implement the evaluation


        res = None

        return res, None
    
    @staticmethod
    def collate_fn(batch_list):
        if None in batch_list:
            print('* Exception error (Dataset): collate_fn')
            return None
        example_merged = collections.defaultdict(list)
        for example in batch_list:
            if type(example) is list:
                for subexample in example:
                    for k, v in subexample.items():
                        example_merged[k].append(v)
            else:
                for k, v in example.items():
                    example_merged[k].append(v)
        ret = {}
        for key, elems in example_merged.items():
            if key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_box",
                        "ind", "mask", "cat"]:
                ret[key] = collections.defaultdict(list)
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        ret[key][str(idx)].append(torch.tensor(ele))
                for kk, vv in ret[key].items():
                    res.append(torch.stack(vv))
                ret[key] = res  # [task], task: (batch, num_class_in_task, feat_shape_h, feat_shape_w)
            elif key in ['gt_boxes_and_cls', 'rdr_cube']:
                ret[key] = torch.tensor(np.stack(elems, axis=0))
            elif key in ['rdr_tensor']:
                elems = np.stack(elems, axis=0)
                ret[key] = torch.tensor(elems)
            elif key in ['meta', 'calib_kradar']:
                ret[key] = elems
            else:
                ret[key] = np.stack(elems, axis=0)

        return ret
    