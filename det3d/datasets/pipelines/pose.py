import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_gaussian3D, gaussian_radius
)
from ..registry import PIPELINES

@PIPELINES.register_module
class AssignLabelPose(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_poses = assigner_cfg.max_poses
        self._min_radius = assigner_cfg.min_radius
        self._consider_radar_visibility = getattr(assigner_cfg, "consider_radar_visibility", False)
        self.cfg = assigner_cfg

    def __call__(self, res, info):
        max_poses = self._max_poses
        max_points = max_poses * 15
        class_names_by_task = [t.class_names for t in self.tasks]
        example = {}
        is_tesseract = True if 'rdr_tesseract' in res else False
        if is_tesseract:
            rdr_tensor = res['rdr_tesseract']
        else:
            rdr_tensor = np.expand_dims(res['rdr_cube'], axis=0) # (1, Z, Y, X)

        if res["mode"] == "train":

            # Calculate output featuremap size
            radar_range = np.array(list(dict(info.DATASET.ROI[info.DATASET.LABEL.ROI_TYPE]).values()), dtype=np.float32).transpose().flatten() # order: (zyx_min, zyx_max)
            voxel_size = info.DATASET.RDR_CUBE.GRID_SIZE
            feature_map_size = np.array(res['hm_size']) // self.out_size_factor # feature map is in 3D
            # prepare the gt by tasks
            gt_classnames = []
            gt_points_by_task = [[] for _ in range(len(self.tasks))] # [[[sub_class_id, x, y, z]], [],  ...]
            for pose in res['poses']:
                for pt_idx, pose_xyz in enumerate(pose):
                    for task_idx, class_names_task in enumerate(class_names_by_task):
                        for class_idx, class_name in enumerate(class_names_task):
                            if pt_idx == class_idx:
                                gt_points_by_task[task_idx].append([class_idx, *pose_xyz])
                                gt_classnames.append(class_name)
            
            draw_gaussian = draw_gaussian3D

            hms, anno_poses, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                gt_pose_task = gt_points_by_task[idx]
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[0], feature_map_size[1], feature_map_size[2])
                              ,dtype=np.float32)
                # [reg]
                anno_pose = np.zeros((max_points, 3), dtype=np.float32) # each pose has 15 keypoints
                ind = np.zeros((max_points), dtype=np.int64)
                mask = np.zeros((max_points), dtype=np.uint8)
                cat = np.zeros((max_points), dtype=np.int64)
                num_points = min(len(gt_pose_task)*15, max_points)  
                for k in range(num_points):
                    cls_id = gt_pose_task[k][0]
                    # hard code the l, w, h  by taking factor of voxel size closest to multiple of 2
                    # l, w, h are: , 1.25625, 1.45 in meters
                    # l, w, h =  16 / self.out_size_factor,  8 / self.out_size_factor, 4 / self.out_size_factor
                    # radius = gaussian_radius((w, l), min_overlap=self.gaussian_overlap)
                    # hard code the radius to be 2
                    radius = 1
                    radius = max(self._min_radius, int(radius))
                    # be really careful for the coordinate system of your box annotation. 
                    x, y, z = gt_pose_task[k][1:4]
                    coor_x, coor_y, coor_z = (x - radar_range[2]) / voxel_size[0] / self.out_size_factor, \
                                        (y - radar_range[1]) / voxel_size[1] / self.out_size_factor, \
                                         (z - radar_range[0]) / voxel_size[2] / self.out_size_factor
                    ct = np.array(
                        [coor_x, coor_y, coor_z], dtype=np.float32) 
                    ct_int = ct.astype(np.int32)
                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[2] and 0 <= ct_int[1] < feature_map_size[1]\
                             and 0 <= ct_int[2] < feature_map_size[0]):
                        continue 
                    draw_gaussian(hm[cls_id], ct_int, radius)
                    new_idx = k
                    x, y, z = ct_int[0], ct_int[1], ct_int[2]
                    cat[new_idx] = cls_id
                    ind[new_idx] = z*feature_map_size[1]*feature_map_size[2] + y * feature_map_size[2] + x
                    mask[new_idx] = 1
                    anno_pose[new_idx] = np.concatenate(
                        (ct - (x, y, z)), axis=None)

                hms.append(hm)
                anno_poses.append(anno_pose)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            example.update({'hm': hms, 'anno_pose': anno_poses, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass
        res_new = {}
        # add a new axis (channel) to rdr_cube
        res_new.update(
            example,
            meta=res['meta'],
            rdr_tensor=rdr_tensor
            )

        return res_new, info