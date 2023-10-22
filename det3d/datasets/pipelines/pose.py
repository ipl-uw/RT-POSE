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
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.pc_type = cfg.pc_type
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):
        points = res[self.pc_type]
        res["mode"] = self.mode
        res['metadata'] = {
            "num_point_features": points.shape[1],
        }
        res['lidar'] = {
                "type": "lidar",
                "points": None,
        }
        P_L2R = res.pop('P_L2R')
        points_xyz_homo_L = np.hstack((points[:, :3], np.ones(len(points)).reshape(-1, 1)))
        points_xyz_R = (P_L2R @ points_xyz_homo_L.T).T[:, :3]
        points[:, :3] = points_xyz_R
        # TODO: add y random flip
        # TODO: add global translate

        if self.shuffle_points:
            np.random.shuffle(points)
        
        res["lidar"]["points"] = points

        return res, info
    

@PIPELINES.register_module
class Voxelization(object):
    def __init__(self, **kwargs):
        cfg = kwargs.get("cfg", None)
        self.range = cfg.range
        self.voxel_size = cfg.voxel_size
        self.max_points_in_voxel = cfg.max_points_in_voxel
        self.max_voxel_num = [cfg.max_voxel_num, cfg.max_voxel_num] if isinstance(cfg.max_voxel_num, int) else cfg.max_voxel_num
        self.double_flip = cfg.get('double_flip', False)

        self.voxel_generator = VoxelGenerator(
            voxel_size=self.voxel_size,
            point_cloud_range=self.range,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num[0],
        )

    def __call__(self, res, info):
        voxel_size = self.voxel_generator.voxel_size
        pc_range = self.voxel_generator.point_cloud_range
        grid_size = self.voxel_generator.grid_size

        if res["mode"] == "train":
            max_voxels = self.max_voxel_num[0]
        else:
            max_voxels = self.max_voxel_num[1]

        voxels, coordinates, num_points = self.voxel_generator.generate(
            res["lidar"]["points"], max_voxels=max_voxels 
        )
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        res["lidar"]["voxels"] = dict(
            voxels=voxels,
            coordinates=coordinates,
            num_points=num_points,
            num_voxels=num_voxels,
            shape=grid_size,
            range=pc_range,
            size=voxel_size
        )

        double_flip = self.double_flip and (res["mode"] != 'train')

        if double_flip:
            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["yflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["yflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["xflip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["xflip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )

            flip_voxels, flip_coordinates, flip_num_points = self.voxel_generator.generate(
                res["lidar"]["double_flip_points"]
            )
            flip_num_voxels = np.array([flip_voxels.shape[0]], dtype=np.int64)

            res["lidar"]["double_flip_voxels"] = dict(
                voxels=flip_voxels,
                coordinates=flip_coordinates,
                num_points=flip_num_points,
                num_voxels=flip_num_voxels,
                shape=grid_size,
                range=pc_range,
                size=voxel_size
            )            

        return res, info
    

@PIPELINES.register_module
class AssignLabelPose(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = np.array(assigner_cfg.out_size_factor)
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
        rdr_res, lidar_res = None, None
        if 'rdr_cube' in res:
            rdr_res = {}
            rdr_tensor = res['rdr_cube']
            if len(rdr_tensor.shape) < 4:
                rdr_tensor = np.expand_dims(rdr_tensor, axis=0) # (1, Z, Y, X)
            rdr_res.update(rdr_tensor=rdr_tensor)
        if 'lidar' in res:
            lidar_res = {}
            lidar_res['points'] = res['lidar']['points']
            if 'voxels' in res["lidar"]:
                voxels = res["lidar"]["voxels"] 
                lidar_res.update(
                    voxels=voxels["voxels"],
                    shape=voxels["shape"],
                    num_points=voxels["num_points"],
                    num_voxels=voxels["num_voxels"],
                    coordinates=voxels["coordinates"],
                )

        if res["mode"] == "train":
            if 'rdr_cube' in res:
                example = {}
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
                        coor_x, coor_y, coor_z = (x - radar_range[2]) / voxel_size[0] / self.out_size_factor[2], \
                                            (y - radar_range[1]) / voxel_size[1] / self.out_size_factor[1], \
                                            (z - radar_range[0]) / voxel_size[2] / self.out_size_factor[0]
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
                rdr_res.update(example)
            if 'lidar' in res:
                example = {}
                # Calculate output featuremap size
                if 'voxels' in res['lidar']:
                    # Calculate output featuremap size
                    grid_size = res["lidar"]["voxels"]["shape"] 
                    pc_range = res["lidar"]["voxels"]["range"]
                    voxel_size = res["lidar"]["voxels"]["size"]
                else:
                    pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                    voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                    grid_size = np.round(grid_size).astype(np.int64)
                feature_map_size = grid_size[::-1] // self.out_size_factor # (z, y, x)
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
                        coor_x, coor_y, coor_z = (x - pc_range[2]) / voxel_size[0] / self.out_size_factor[2], \
                                            (y - pc_range[1]) / voxel_size[1] / self.out_size_factor[1], \
                                            (z - pc_range[0]) / voxel_size[2] / self.out_size_factor[0]
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
                lidar_res.update(example)
        else:
            pass

        res_new = {}
        res_new.update(
            meta=res['meta'],
            )
        if not rdr_res is None:
            res_new.update(rdr=rdr_res)
        if not lidar_res is None:
            res_new.update(lidar=lidar_res)

        return res_new, info
    
@PIPELINES.register_module
class AssignLabelPose2(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = np.array(assigner_cfg.out_size_factor)
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_poses = assigner_cfg.max_poses
        self._min_radius = assigner_cfg.min_radius
        self._consider_radar_visibility = getattr(assigner_cfg, "consider_radar_visibility", False)
        self.cfg = assigner_cfg

    def __call__(self, res, info):
        max_poses = self._max_poses
        max_points = max_poses 
        class_names_by_task = [t.class_names for t in self.tasks]
        rdr_res, lidar_res = None, None
        if 'rdr_cube' in res:
            rdr_res = {}
            rdr_tensor = res['rdr_cube']
            if len(rdr_tensor.shape) < 4:
                rdr_tensor = np.expand_dims(rdr_tensor, axis=0) # (1, Z, Y, X)
            rdr_res.update(rdr_tensor=rdr_tensor)
        if 'lidar' in res:
            lidar_res = {}
            lidar_res['points'] = res['lidar']['points']
            if 'voxels' in res["lidar"]:
                voxels = res["lidar"]["voxels"] 
                lidar_res.update(
                    voxels=voxels["voxels"],
                    shape=voxels["shape"],
                    num_points=voxels["num_points"],
                    num_voxels=voxels["num_voxels"],
                    coordinates=voxels["coordinates"],
                )

        if res["mode"] == "train":
            if 'rdr_cube' in res:
                example = {}
                # Calculate output featuremap size
                radar_range = np.array(list(dict(info.DATASET.ROI[info.DATASET.LABEL.ROI_TYPE]).values()), dtype=np.float32).transpose().flatten() # order: (zyx_min, zyx_max)
                voxel_size = info.DATASET.RDR_CUBE.GRID_SIZE
                feature_map_size = np.array(res['hm_size']) // self.out_size_factor # feature map is in 3D
                # prepare the gt by tasks
                gt_classnames = []
                gt_points_by_task = [[] for _ in range(len(self.tasks))] # [[[sub_class_id, 15 key points' xyz]], [],  ...]
                for pose in res['poses']:
                    pose_gt  =[]
                    for pt_idx, pose_xyz in enumerate(pose):
                        if pt_idx == 0:
                            gt_classnames.append(class_names_by_task[0][pt_idx])
                            pose_gt += [pt_idx] # hardcode the task type here
                        pose_gt += pose_xyz
                    gt_points_by_task[0].append(pose_gt) # hardcode the task index here, should remove this in the future version
                draw_gaussian = draw_gaussian3D

                hms, anno_poses, inds, masks, cats = [], [], [], [], []
                for idx, task in enumerate(self.tasks):
                    gt_pose_task = gt_points_by_task[idx]
                    hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[0], feature_map_size[1], feature_map_size[2])
                                ,dtype=np.float32)
                    # [reg]
                    anno_pose = np.zeros((max_points, 45), dtype=np.float32) # each pose has 15 keypoints
                    ind = np.zeros((max_points), dtype=np.int64)
                    mask = np.zeros((max_points), dtype=np.uint8)
                    cat = np.zeros((max_points), dtype=np.int64)
                    num_points = min(len(gt_pose_task), max_points)
                    for k in range(num_points):
                        cls_id = gt_pose_task[k][0]
                        radius = self._min_radius # todo: rename the config or use a function to geenrate radius
                        # radius = max(self._min_radius, int(radius))

                        keypoints_xyz = gt_pose_task[k][1:]
                        ct = []
                        for i in range(int(len(keypoints_xyz)/3)):
                            x, y, z = keypoints_xyz[i*3:3*(i+1)] # center's xyz
                            ct.append((x - radar_range[2]) / voxel_size[0] / self.out_size_factor[2])
                            ct.append((y - radar_range[1]) / voxel_size[1] / self.out_size_factor[1])
                            ct.append((z - radar_range[0]) / voxel_size[2] / self.out_size_factor[0])
                        ct = np.array(ct, dtype=np.float32)
                        ct_int = ct.astype(np.int32)[:3]
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
                        # anno pose stores the position offset for each keypoint the center's voxel
                        anno_pose[new_idx] = (ct.reshape((-1, 3)) - ct_int[None, :].astype(np.float32)).flatten()

                    hms.append(hm)
                    anno_poses.append(anno_pose)
                    masks.append(mask)
                    inds.append(ind)
                    cats.append(cat)

                example.update({'hm': hms, 'anno_pose': anno_poses, 'ind': inds, 'mask': masks, 'cat': cats})
                rdr_res.update(example)
            if 'lidar' in res:
                example = {}
                # Calculate output featuremap size
                if 'voxels' in res['lidar']:
                    # Calculate output featuremap size
                    grid_size = res["lidar"]["voxels"]["shape"] 
                    pc_range = res["lidar"]["voxels"]["range"]
                    voxel_size = res["lidar"]["voxels"]["size"]
                else:
                    pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                    voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                    grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                    grid_size = np.round(grid_size).astype(np.int64)
                feature_map_size = grid_size[::-1] // self.out_size_factor # (z, y, x)
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
                        coor_x, coor_y, coor_z = (x - pc_range[2]) / voxel_size[0] / self.out_size_factor[2], \
                                            (y - pc_range[1]) / voxel_size[1] / self.out_size_factor[1], \
                                            (z - pc_range[0]) / voxel_size[2] / self.out_size_factor[0]
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
                lidar_res.update(example)
        else:
            pass

        res_new = {}
        res_new.update(
            meta=res['meta'],
            )
        if not rdr_res is None:
            res_new.update(rdr=rdr_res)
        if not lidar_res is None:
            res_new.update(lidar=lidar_res)

        return res_new, info