import numpy as np

from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.builder import build_dbsampler

from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.core.utils.center_utils import (
    draw_umich_gaussian, gaussian_radius
)
from ..registry import PIPELINES


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

@PIPELINES.register_module
class Preprocess(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
            self.global_translate_std = cfg.get('global_translate_std', 0)
            self.class_names = cfg.class_names
            if cfg.db_sampler != None:
                self.db_sampler = build_dbsampler(cfg.db_sampler)
            else:
                self.db_sampler = None 
                
            self.npoints = cfg.get("npoints", -1)

        self.no_augmentation = cfg.get('no_augmentation', False)

    def __call__(self, res, info):

        res["mode"] = self.mode

        if res["type"] in ["WaymoDataset"]:
            if "combined" in res["lidar"]:
                points = res["lidar"]["combined"]
            else:
                points = res["lidar"]["points"]
        elif res["type"] in ["NuScenesDataset"]:
            points = res["lidar"]["combined"]
        else:
            raise NotImplementedError

        if self.mode == "train":
            anno_dict = res["lidar"]["annotations"]

            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.array(anno_dict["names"]).reshape(-1),
            }

        if self.mode == "train" and not self.no_augmentation:
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )

            _dict_select(gt_dict, selected)

            if self.min_points_in_gt > 0:
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"]
                )
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None
                )

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes]
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )


                    points = np.concatenate([sampled_points, points], axis=0)

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            gt_dict["gt_boxes"], points = prep.random_flip_both(gt_dict["gt_boxes"], points)
            
            gt_dict["gt_boxes"], points = prep.global_rotation(
                gt_dict["gt_boxes"], points, rotation=self.global_rotation_noise
            )
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise
            )
            gt_dict["gt_boxes"], points = prep.global_translate_(
                gt_dict["gt_boxes"], points, noise_translate_std=self.global_translate_std
            )
        elif self.no_augmentation:
            gt_boxes_mask = np.array(
                [n in self.class_names for n in gt_dict["gt_names"]], dtype=np.bool_
            )
            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes


        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

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
            gt_dict = res["lidar"]["annotations"]
            bv_range = pc_range[[0, 1, 3, 4]]
            mask = prep.filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            res["lidar"]["annotations"] = gt_dict
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

def flatten(box):
    return np.concatenate(box, axis=0)

def merge_multi_group_label(gt_classes, num_classes_by_task): 
    num_task = len(gt_classes)
    flag = 0 

    for i in range(num_task):
        gt_classes[i] += flag 
        flag += num_classes_by_task[i]

    return flatten(gt_classes)

@PIPELINES.register_module
class AssignLabel(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.cfg = assigner_cfg

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        # todo: check how to transform x and y

        if res["mode"] == "train":
            # Calculate output featuremap size
            if 'voxels' in res['lidar']:
                # Calculate output featuremap size
                grid_size = res["lidar"]["voxels"]["shape"] 
                pc_range = res["lidar"]["voxels"]["range"]
                voxel_size = res["lidar"]["voxels"]["size"]
                feature_map_size = grid_size[:2] // self.out_size_factor
            else:
                pc_range = np.array(self.cfg['pc_range'], dtype=np.float32)
                voxel_size = np.array(self.cfg['voxel_size'], dtype=np.float32)
                grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
                grid_size = np.round(grid_size).astype(np.int64)

            feature_map_size = grid_size[:2] // self.out_size_factor

            gt_dict = res["lidar"]["annotations"]

            # reorganize the gt_dict by tasks
            task_masks = []
            flag = 0
            for class_name in class_names_by_task:
                task_masks.append(
                    [
                        np.where(
                            gt_dict["gt_classes"] == class_name.index(i) + 1 + flag
                        )
                        for i in class_name
                    ]
                )
                flag += len(class_name)

            task_boxes = []
            task_classes = []
            task_names = []
            flag2 = 0
            for idx, mask in enumerate(task_masks):
                task_box = []
                task_class = []
                task_name = []
                for m in mask:
                    task_box.append(gt_dict["gt_boxes"][m])
                    task_class.append(gt_dict["gt_classes"][m] - flag2)
                    task_name.append(gt_dict["gt_names"][m])
                task_boxes.append(np.concatenate(task_box, axis=0))
                task_classes.append(np.concatenate(task_class))
                task_names.append(np.concatenate(task_name))
                flag2 += len(mask)

            for task_box in task_boxes:
                # limit rad to [-pi, pi]
                task_box[:, -1] = box_np_ops.limit_period(
                    task_box[:, -1], offset=0.5, period=np.pi * 2
                )

            # print(gt_dict.keys())
            gt_dict["gt_classes"] = task_classes
            gt_dict["gt_names"] = task_names
            gt_dict["gt_boxes"] = task_boxes

            res["lidar"]["annotations"] = gt_dict

            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]),
                              dtype=np.float32)

                if res['type'] == 'NuScenesDataset':
                    # [reg, hei, dim, vx, vy, rots, rotc]
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32)
                elif res['type'] == 'WaymoDataset':
                    anno_box = np.zeros((max_objs, 10), dtype=np.float32) 
                else:
                    raise NotImplementedError("Only Support nuScene for Now!")

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(gt_dict['gt_boxes'][idx].shape[0], max_objs)  

                for k in range(num_objs):
                    cls_id = gt_dict['gt_classes'][idx][k] - 1

                    w, l, h = gt_dict['gt_boxes'][idx][k][3], gt_dict['gt_boxes'][idx][k][4], \
                              gt_dict['gt_boxes'][idx][k][5]
                    w, l = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_dict['gt_boxes'][idx][k][0], gt_dict['gt_boxes'][idx][k][1], \
                                  gt_dict['gt_boxes'][idx][k][2]

                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor

                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32)  
                        ct_int = ct.astype(np.int32)

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue 

                        draw_gaussian(hm[cls_id], ct, radius)
                        new_idx = k
                        x, y = ct_int[0], ct_int[1]

                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[0] + x
                        mask[new_idx] = 1

                        if res['type'] == 'NuScenesDataset': 
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][8]
                            anno_box[new_idx] = np.concatenate(
                                (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                                np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        elif res['type'] == 'WaymoDataset':
                            vx, vy = gt_dict['gt_boxes'][idx][k][6:8]
                            rot = gt_dict['gt_boxes'][idx][k][-1]
                            anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_dict['gt_boxes'][idx][k][3:6]),
                            np.array(vx), np.array(vy), np.sin(rot), np.cos(rot)), axis=None)
                        else:
                            raise NotImplementedError("Only Support Waymo and nuScene for Now")

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = flatten(gt_dict['gt_boxes'])
            classes = merge_multi_group_label(gt_dict['gt_classes'], num_classes_by_task)

            if res["type"] == "NuScenesDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            elif res['type'] == "WaymoDataset":
                gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
            else:
                raise NotImplementedError()

            boxes_and_cls = np.concatenate((boxes, 
                classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
            boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
            gt_boxes_and_cls[:num_obj] = boxes_and_cls

            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass

        res["lidar"]["targets"] = example

        return res, info


def random_flip_y(gt_label, rdr_tensor, ROI, probability):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability]
    )
    if enable:
        rdr_tensor = np.flip(rdr_tensor, axis=2)
        gt_label_new = []
        for obj in gt_label:
            obj[2][1] = -obj[2][1] # flip y coordinate
            obj[2][3] = -obj[2][3] + 2 * np.pi # flip heading angle
            gt_label_new.append(obj)
        ROI.y = [-ROI.y[1], -ROI.y[0]]
        return gt_label_new, rdr_tensor, ROI
    return gt_label, rdr_tensor, ROI


@PIPELINES.register_module
class AssignLabelRadar(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self._consider_radar_visibility = getattr(assigner_cfg, "consider_radar_visibility", False)
        self.cfg = assigner_cfg
        if "flip_y_prob" in kwargs:
            self.flip_y_prob = kwargs["flip_y_prob"]
        else:
            self.flip_y_prob = 0.0
        if self._consider_radar_visibility:
            self.radar_visibility_bin = assigner_cfg.radar_visibility_cfg.bin
            self.radar_visibility_mod_coeff = assigner_cfg.radar_visibility_cfg.mod_coeff

    def get_mod_coef_by_points(self, num_of_points):
        bin_idx = 0
        for bin in self.radar_visibility_bin:
            if num_of_points < bin:
                break
            bin_idx += 1
        return self.radar_visibility_mod_coeff[bin_idx]


    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        example = {}
        is_tesseract = True if 'rdr_tesseract' in res else False
        if is_tesseract:
            rdr_tensor = res['rdr_tesseract']
        else:
            rdr_tensor = np.expand_dims(res['rdr_cube'], axis=0) # (1, Z, Y, X)

        if res["mode"] == "train":
            # Data augmentation
            # res['label'],  rdr_tensor, info.DATASET.RDR_CUBE.ROI  = random_flip_y(res['label'], rdr_tensor, \
            #                                                                            info.DATASET.RDR_CUBE.ROI, self.flip_y_prob)

            # Calculate output featuremap size
            radar_range = np.array(list(dict(info.DATASET.ROI[info.DATASET.LABEL.ROI_TYPE]).values()), dtype=np.float32).transpose().flatten() # order: (zyx_min, zyx_max)
            voxel_size = info.DATASET.RDR_CUBE.GRID_SIZE
            grid_size_y = len(np.arange(radar_range[1], radar_range[4], voxel_size))
            grid_size_x = len(np.arange(radar_range[2], radar_range[5], voxel_size))
            feature_map_size = np.array((grid_size_y, grid_size_x)) // self.out_size_factor # feature map is in BEV
            # prepare the gt by tasks
            gt_classnames = []
            gt_box_by_task = [[] for _ in range(len(self.tasks))] # [[[sub_class_id, x, y, z, l, w, h, theta]], [],  ...]
            for obj in res['objs']:
                for task_idx, class_names_task in enumerate(class_names_by_task):
                    for class_idx, class_name in enumerate(class_names_task):
                        if class_name == obj['obj_type']:
                            obj_theta = box_np_ops.limit_period(obj['euler'][2], offset=0.5, period=np.pi * 2)
                            gt_box_by_task[task_idx].append([class_idx, *obj['xyz'], *obj['lwh'], obj_theta, int(obj['obj_id']), int(obj['num_cfar_pts']) if 'num_cfar_pts' in obj else 0])
                            gt_classnames.append(class_name)
            
            draw_gaussian = draw_umich_gaussian

            hms, anno_boxs, inds, masks, cats, obj_ids = [], [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                gt_box_task = gt_box_by_task[idx]
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[0], feature_map_size[1]),
                              dtype=np.float32)
                # [reg, hei, dim, rotsin, rotcos]
                anno_box = np.zeros((max_objs, 8), dtype=np.float32)
                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)
                obj_id = -np.ones((max_objs), dtype=np.int64)
                num_objs = min(len(gt_box_task), max_objs)  
                for k in range(num_objs):
                    cls_id = gt_box_task[k][0]
                    l, w, h = gt_box_task[k][4], gt_box_task[k][5], \
                              gt_box_task[k][6]
                    l, w = w / voxel_size / self.out_size_factor, l / voxel_size / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((w, l), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))
                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_box_task[k][1:4]
                        coor_x, coor_y = (x - radar_range[2]) / voxel_size / self.out_size_factor, \
                                         (y - radar_range[1]) / voxel_size / self.out_size_factor
                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32) 
                        ct_int = ct.astype(np.int32)
                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[1] and 0 <= ct_int[1] < feature_map_size[0]):
                            continue 
                        mod_coef = self.get_mod_coef_by_points(gt_box_task[k][9]) if self._consider_radar_visibility else 1.
                        draw_gaussian(hm[cls_id], ct, radius, modulation_coef=mod_coef)
                        new_idx = k
                        x, y = ct_int[0], ct_int[1]
                        cat[new_idx] = cls_id
                        obj_id[new_idx] = gt_box_task[k][8]
                        ind[new_idx] = y * feature_map_size[1] + x
                        mask[new_idx] = 1
                        rot = gt_box_task[k][7]
                        anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_box_task[k][4:7]),
                             np.sin(rot), np.cos(rot)), axis=None)

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)
                obj_ids.append(obj_id)

            # used for two stage code 
            # boxes = []
            # for gt_boxes in gt_box_by_task:
            #     boxes.append(np.array(gt_boxes).reshape(-1, 8))
            # boxes = np.concatenate(boxes, axis=0)[:, 1:] if len(boxes) > 0 else np.array(boxes).reshape(-1, 7)# (x, y, z, l, w, h, rot_z)
            # classes = np.array([info['class_names'].index(class_name) for class_name in gt_classnames])
            # gt_boxes_and_cls = np.zeros((max_objs, 8), dtype=np.float32)
            # boxes_and_cls = np.concatenate((boxes, classes.reshape(-1, 1).astype(np.float32)), axis=1)
            # num_obj = len(boxes_and_cls)
            # assert num_obj <= max_objs
            # gt_boxes_and_cls[:num_obj] = boxes_and_cls
            # example.update({'gt_boxes_and_cls': gt_boxes_and_cls})
            # used for two stage code 

            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats, 'obj_id': obj_ids})
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
    

@PIPELINES.register_module
class PreprocessKradar(object):
    def __init__(self, cfg=None, **kwargs):
        self.shuffle_points = cfg.shuffle_points
        self.min_points_in_gt = cfg.get("min_points_in_gt", -1)
        self.pc_type = cfg.pc_type
        self.mode = cfg.mode
        if self.mode == "train":
            self.global_rotation_noise = cfg.global_rot_noise
            self.global_scaling_noise = cfg.global_scale_noise
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
                "annotations": None,
        }
        gt_dict = None
        # TODO: add data augmentation

        if self.shuffle_points:
            np.random.shuffle(points)
        
        res["lidar"]["points"] = points.astype(np.float32)

        if self.mode == "train":
            res["lidar"]["annotations"] = gt_dict

        return res, info
    

@PIPELINES.register_module
class VoxelizationKradar(object):
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
class AssignLabelLidar(object):
    def __init__(self, **kwargs):
        """Return CenterNet training labels like heatmap, height, offset"""
        assigner_cfg = kwargs["cfg"]
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.target_assigner.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.cfg = assigner_cfg

    def __call__(self, res, info):
        max_objs = self._max_objs
        class_names_by_task = [t.class_names for t in self.tasks]
        # num_classes_by_task = [t.num_class for t in self.tasks]

        example = {}

        if res["mode"] == "train":
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

            feature_map_size = grid_size[::-1][1:] // self.out_size_factor # (y, x)
            # prepare the gt by tasks
            gt_classnames = []
            gt_box_by_task = [[] for _ in range(len(self.tasks))] # [[[sub_class_id, x, y, z, l, w, h, theta]], [],  ...]
            for obj in res['objs']:
                for task_idx, class_names_task in enumerate(class_names_by_task):
                    for class_idx, class_name in enumerate(class_names_task):
                        if class_name == obj['obj_type']:
                            obj_theta = box_np_ops.limit_period(obj['euler'][2], offset=0.5, period=np.pi * 2)
                            gt_box_by_task[task_idx].append([class_idx, *obj['xyz'], *obj['lwh'], obj_theta])
                            gt_classnames.append(class_name)
                            
            draw_gaussian = draw_umich_gaussian
            hms, anno_boxs, inds, masks, cats = [], [], [], [], []

            for idx, task in enumerate(self.tasks):
                gt_box_task = gt_box_by_task[idx]
                hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[0], feature_map_size[1]),
                              dtype=np.float32)

                # [reg, hei, dim, rotsin, rotcos]
                anno_box = np.zeros((max_objs, 8), dtype=np.float32)

                ind = np.zeros((max_objs), dtype=np.int64)
                mask = np.zeros((max_objs), dtype=np.uint8)
                cat = np.zeros((max_objs), dtype=np.int64)

                num_objs = min(len(gt_box_task), max_objs)  

                for k in range(num_objs):
                    cls_id = gt_box_task[k][0]
                    l, w, h = gt_box_task[k][4], gt_box_task[k][5], \
                              gt_box_task[k][6]
                    l, w = w / voxel_size[0] / self.out_size_factor, l / voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = gaussian_radius((w, l), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))
                        # be really careful for the coordinate system of your box annotation. 
                        x, y, z = gt_box_task[k][1:4]
                        coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, \
                                         (y - pc_range[1]) / voxel_size[1] / self.out_size_factor
                        ct = np.array(
                            [coor_x, coor_y], dtype=np.float32) 
                        ct_int = ct.astype(np.int32)
                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[1] and 0 <= ct_int[1] < feature_map_size[0]):
                            continue 
                        draw_gaussian(hm[cls_id], ct, radius)
                        new_idx = k
                        x, y = ct_int[0], ct_int[1]
                        cat[new_idx] = cls_id
                        ind[new_idx] = y * feature_map_size[1] + x
                        mask[new_idx] = 1
                        rot = gt_box_task[k][7]
                        anno_box[new_idx] = np.concatenate(
                            (ct - (x, y), z, np.log(gt_box_task[k][4:7]),
                             np.sin(rot), np.cos(rot)), axis=None)

                hms.append(hm)
                anno_boxs.append(anno_box)
                masks.append(mask)
                inds.append(ind)
                cats.append(cat)

            # used for two stage code 
            boxes = []
            for gt_boxes in gt_box_by_task:
                boxes.append(np.array(gt_boxes).reshape(-1, 8))
            boxes = np.concatenate(boxes, axis=0)[:, 1:] if len(boxes) > 0 else np.array(boxes).reshape(-1, 7)# (x, y, z, l, w, h, rot_z)
            classes = np.array([info['class_names'].index(class_name) for class_name in gt_classnames])
            gt_boxes_and_cls = np.zeros((max_objs, 8), dtype=np.float32)
            boxes_and_cls = np.concatenate((boxes, classes.reshape(-1, 1).astype(np.float32)), axis=1)
            num_obj = len(boxes_and_cls)
            assert num_obj <= max_objs
            gt_boxes_and_cls[:num_obj] = boxes_and_cls
            example.update({'gt_boxes_and_cls': gt_boxes_and_cls})
            example.update({'hm': hms, 'anno_box': anno_boxs, 'ind': inds, 'mask': masks, 'cat': cats})
        else:
            pass
        res["lidar"]["targets"] = example

        return res, info