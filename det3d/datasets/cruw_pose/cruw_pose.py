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
import json
from collections import defaultdict
import open3d as o3d
from eval_util import *

@DATASETS.register_module
class CRUW_POSE_Dataset(Dataset):
    def __init__(self, cfg, label_file, class_names=None, pipeline=None, split='train'):
        super().__init__()
        cfg = DefaultMunch.fromDict(cfg)
        self.cfg = cfg
        self.split = split
        self.class_names = class_names
        self.cfg.update(class_names=class_names)
        if self.cfg.DATASET.RDR_TYPE == 'zyx_real':
            # Default ROI for CB (When generating CB from matlab applying interpolation)
            self.arr_z_cb = np.arange(-5.8, 5.8, 11.6/32)
            self.arr_y_cb = np.arange(-10.05, 10.05, 20.1/128)
            self.arr_x_cb = np.arange(0, 11.6, 11.6/256)
            self.is_consider_roi_rdr_cb = cfg.DATASET.RDR_CUBE.IS_CONSIDER_ROI
            if self.is_consider_roi_rdr_cb:
                self.consider_roi_cube(cfg.DATASET.ROI[cfg.DATASET.LABEL['ROI_TYPE']])
        self.read_meta()
        self.label_file = os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, label_file)
        self.load_samples()
        if pipeline is None:
            self.pipeline = None
        else:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def _set_group_flag(self):
        self.flag = np.ones(len(self), dtype=np.uint8)

    def read_meta(self):
        with open(os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, self.cfg.DATASET.DIR.META_FILE), 'r') as f:
            lines = f.readlines()
        seq_id_to_name = {}
        for line in lines:
            seq_id, seq_name = line.strip().split(',')
            seq_id_to_name[seq_id] = seq_name
        self.seq_id_to_name = seq_id_to_name


    def load_samples(self):
        with open(self.label_file, 'r') as f:
            samples_by_seq = json.load(f)
        samples = []
        for seq, seq_frames in samples_by_seq.items():
            # TODO: remove the below line in the future
            if self.seq_id_to_name[seq] not in ['2023_0718_1642', '2023_0724_1553', '2023_0725_1559', '2023_0725_1600', \
                           '2023_0725_1602', '2023_0725_1603', '2023_0725_1604', '2023_0725_1714', \
                            '2023_0725_1716', '2023_0726_1602', '2023_0726_1619', '2023_0726_1620', \
                                '2023_0730_1240', '2023_0730_1242', '2023_0730_1245', '2023_0730_1316', \
                                    '2023_0730_1321', '2023_0730_1331', '2023_0730_1332']:
                continue
            for frame, frame_objs in seq_frames.items():
                sample = {}
                sample['seq'] = seq
                for obj in frame_objs:
                    sample['rdr_frame'] = obj['Radar_frameID']
                    sample['frame'] = frame
                    sample['poses'] = [obj['pose']]
                    samples.append(sample)
        self.samples = samples

            


    def consider_roi_tesseract(self, roi_polar, is_reflect_to_cfg=True):
        self.list_roi_idx = []
        deg2rad = np.pi/180.
        rad2deg = 180./np.pi
        for k, v in roi_polar.items():
            if v is not None:
                min_max = v if k == 'r' else (np.array(v) * deg2rad).tolist() 
                arr_roi, idx_min, idx_max = self.get_arr_in_roi(getattr(self, f'arr_{k}'), min_max)
                setattr(self, f'arr_{k}', arr_roi)
                self.list_roi_idx.append(idx_min)
                self.list_roi_idx.append(idx_max)
                if is_reflect_to_cfg:
                    v_new = [arr_roi[0], arr_roi[-1]]
                    v_new =  v_new if k == 'r' else (np.array(v_new) * rad2deg).tolist()
                    self.cfg.DATASET.DEAR.ROI[k] = v_new


    def consider_roi_cube(self, roi_cart):
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
            idx_attr += 1

    def get_arr_in_roi(self, arr, min_max):
        min_val, max_val = min_max
        idx_min = np.argmin(abs(arr-min_val))
        idx_max = np.argmin(abs(arr-max_val))
        if max_val > arr[-1]:
            return arr[idx_min:idx_max+1], idx_min, idx_max
        return arr[idx_min:idx_max], idx_min, idx_max-1

    def check_to_add_obj(self, object_xyz):
        x, y, z = object_xyz
        x_min, y_min, z_min, x_max, y_max, z_max = self.roi_label
        if self.is_roi_check_with_azimuth:
            min_azi, max_azi = self.max_azimtuth_rad
            azimuth_center = np.arctan2(y, x)
            if (azimuth_center < min_azi) or (azimuth_center > max_azi)\
                or (x < x_min) or (y < y_min) or (z < z_min)\
                or (x > x_max) or (y > y_max) or (z > z_max):
                return False
        return True


    def get_tesseract(self, seq, rdr_frame_id):
        # TODO: get the preprocessed tesseract and return it.
        # return arr_dear
        pass

        
    def get_cube(self, seq, rdr_frame_id):
        arr_cube = np.load(os.path.join(self.cfg.DATASET.DIR.ROOT_DIR, self.seq_id_to_name[seq], 'radar', 'npy', f'{rdr_frame_id}.npy'))
        norm_vals = [float(norm_val) for norm_val in self.cfg.DATASET.RDR_CUBE.NORMALIZING_VALUE]
        norm_start, norm_scale = norm_vals[0], norm_vals[1]-norm_vals[0]
        # RoI selection
        idx_z_min, idx_z_max, idx_y_min, idx_y_max, idx_x_min, idx_x_max = self.list_roi_idx_cb
        arr_cube = arr_cube[idx_z_min:idx_z_max+1,idx_y_min:idx_y_max+1,idx_x_min:idx_x_max+1]
        # normalize
        arr_cube = (arr_cube - norm_start) / norm_scale
        arr_cube[arr_cube < 0.] = 0.

        return arr_cube

    def __len__(self):
        return len(self.samples)


    def get_sample_by_idx(self, idx):
        sample = self.samples[idx]
        dict_item = {}
        dict_item['meta'] = {'seq': sample['seq'], 'frame': sample['frame'], 'rdr_frame': sample['rdr_frame']}
        dict_item['poses'] = sample['poses']
        dict_item['rdr_cube'] = self.get_cube(sample['seq'], sample['rdr_frame'])
        dict_item.update(mode=self.split)
        dict_item['hm_size'] = (len(self.arr_z_cb), len(self.arr_y_cb), len(self.arr_x_cb))
        dict_item, _ = self.pipeline(dict_item, info=self.cfg)
        return dict_item

    def __getitem__(self, idx):
        dict_item = self.get_sample_by_idx(idx)
        return dict_item
        
    def evaluation(self, detections, output_dir=None, testset=False):
        with open(self.label_file, 'r') as f:
            gt = json.load(f)
        seq_mpjpe = defaultdict(list)
        seq_abs_mpjpe = defaultdict(list)
        for seq_frame_rdr_frame, val in detections.items():
            seq, frame, rdr_frame = seq_frame_rdr_frame.split('/')
            gt_points = gt[seq][frame][0]['pose']
            keypoints = [point[1:4] for point in val['keypoints']]
            seq_mpjpe[seq].append(MPJPE(np.array(keypoints), np.array(gt_points)))
            seq_abs_mpjpe[seq].append(ABS_MPJPE(np.array(keypoints), np.array(gt_points)))
        seq_res = {}
        for seq, mpjpe_list in seq_mpjpe.items():
            seq_res[self.seq_id_to_name[seq]] = {}
            seq_res[self.seq_id_to_name[seq]]['MPJPE'] = np.mean(mpjpe_list)
            seq_res[self.seq_id_to_name[seq]]['ABS_MPJPE'] = np.mean(seq_abs_mpjpe[seq])
        res = {}
        total_results = {}
        total_results['MPJPE'] = np.mean([v['MPJPE'] for k, v in seq_res.items()])
        total_results['ABS_MPJPE'] = np.mean([v['ABS_MPJPE'] for k, v in seq_res.items()])
        res['results'] = total_results
        seq_res['ALL'] = total_results
        res['seq_results'] = seq_res
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
            if key in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", "hm", "anno_pose",
                        "ind", "mask", "cat", "obj_id"]:
                ret[key] = collections.defaultdict(list)
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        ret[key][str(idx)].append(torch.tensor(ele))
                for kk, vv in ret[key].items():
                    res.append(torch.stack(vv))
                ret[key] = res  # [task], task: (batch, num_class_in_task, feat_shape_h, feat_shape_w)
            elif key in ["voxels", "num_points", "num_gt", "voxel_labels", "num_voxels",
                   "cyv_voxels", "cyv_num_points", "cyv_num_voxels"]:
                ret[key] = torch.tensor(np.concatenate(elems, axis=0))
            elif key == "points":
                ret[key] = [torch.tensor(elem) for elem in elems]
            elif key in ["coordinates", "cyv_coordinates"]:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                    )
                    coors.append(coor_pad)
                ret[key] = torch.tensor(np.concatenate(coors, axis=0))
            elif key in ['gt_boxes_and_cls']:
                ret[key] = torch.tensor(np.stack(elems, axis=0))
            elif key in ['rdr_tensor']:
                elems = np.stack(elems, axis=0)
                ret[key] = torch.tensor(elems)
            elif key in ['meta', 'calib_kradar']:
                ret[key] = elems
            else:
                ret[key] = np.stack(elems, axis=0)

        return ret
