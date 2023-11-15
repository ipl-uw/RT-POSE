# This script evaluates localization performance in 3D object detection.
# Author: Andy Cheng, Yizhou Wang

import os
import argparse
from turtle import pos
import numpy as np
import json
import math
from collections import defaultdict
from tqdm import tqdm
from pytorch3d.transforms import transform3d as t3d
from pytorch3d.transforms.rotation_conversions import quaternion_to_matrix
import torch

BOX3D_CORNER_MAPPING = [
    [1, 1, 1, 1, -1, -1, -1, -1],
    [1, -1, -1, 1, 1, -1, -1, 1],
    [1, 1, -1, -1, 1, 1, -1, -1]
]

line_seg_idxs = [[0, 1], [1, 2], [2, 3], [3, 0]]


def evaluate_img(gts_dict, dts_dict, imgId, catId, olss_dict, olsThrs, recThrs, classes, log=False):
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    if len(gts) == 0 and len(dts) == 0:
        return None

    if log:
        olss_flatten = np.ravel(olss_dict[imgId, catId])
        print("Frame %d: %10s %s" % (imgId, classes[catId], list(olss_flatten)))

    dtind = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in dtind] # rearange the dt from conf. score high to low
    olss = olss_dict[imgId, catId]

    T = len(olsThrs)
    G = len(gts)
    D = len(dts)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))

    if not len(olss) == 0:
        for tind, t in enumerate(olsThrs):
            for dind, d in enumerate(dts):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gts):
                    # if this gt already matched, continue
                    if gtm[tind, gind] > 0:
                        continue
                    if olss[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = olss[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    # no gt matched
                    continue
                dtm[tind, dind] = gts[m]['id']
                gtm[tind, m] = d['id']
    # store results for given image and category
    return {
        'image_id': imgId,
        'category_id': catId,
        'dtIds': [d['id'] for d in dts],
        'gtIds': [g['id'] for g in gts],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dts],
    }

# recThrs: recall thresholds
def accumulate(evalImgs, start_end_frame, olsThrs, recThrs, classes, log=True):
    n_class = len(classes)

    T = len(olsThrs)
    R = len(recThrs)
    K = n_class
    precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
    recall = -np.ones((T, K))
    scores = -np.ones((T, R, K))
    n_objects = np.zeros((K,))

    for classid in range(n_class):
        E = [evalImgs[i * n_class + classid] for i in range(start_end_frame[1] - start_end_frame[0] + 1)]
        E = [e for e in E if not e is None] # filter out None
        if len(E) == 0:
            continue

        dtScores = np.concatenate([e['dtScores'] for e in E])
        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]

        dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds] # TxD
        gtm = np.concatenate([e['gtMatches'] for e in E], axis=1) # TXG
        nd = dtm.shape[1]  # number of detections
        ng = gtm.shape[1]  # number of ground truth
        n_objects[classid] = ng

        if log:
            print("%10s: %4d dets, %4d gts" % (classes[classid], dtm.shape[1], gtm.shape[1]))

        tps = np.array(dtm, dtype=bool)
        fps = np.logical_not(dtm)
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float32) # TXD
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float32) # TXD

        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp = np.array(tp)
            fp = np.array(fp)
            rc = tp / (ng + np.spacing(1)) # recall
            pr = tp / (fp + tp + np.spacing(1))
            q = np.zeros((R,)) #
            ss = np.zeros((R,)) # 

            if nd:
                recall[t, classid] = rc[-1] 
            else:
                recall[t, classid] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            pr = pr.tolist()
            q = q.tolist()

            for i in range(nd - 1, 0, -1): # make precision is desending
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            inds = np.searchsorted(rc, recThrs, side='left')
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
            except:
                pass
            precision[t, :, classid] = np.array(q)
            scores[t, :, classid] = np.array(ss)

    eval = {
        'counts': [T, R, K],
        'object_counts': n_objects, # different object class's object counts
        'precision': precision, # ols thresholds x recall thresholds x classid
        'recall': recall, # ols thresholds x classid, recall considering all the detections
        'scores': scores, # ols thresholds x recall thresholds x classid, detection score corresponds to the recall threshold
    }
    return eval


def summarize(eval, olsThrs, recThrs, classes, gl=True):
    n_class = len(classes)

    def _summarize(eval=eval, ap=1, olsThr=None):
        object_counts = eval['object_counts']
        n_objects = np.sum(object_counts)
        if ap == 1:
            # dimension of precision: [TxRxK]
            s = eval['precision']
            # IoU
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :, :] # let s remains 3 rank
        else:
            # dimension of recall: [TxK]
            s = eval['recall']
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :]
        # mean_s = np.mean(s[s>-1])
        mean_s = 0
        # weighted average the score by number of objects in different classes
        for classid in range(n_class):
            if ap == 1:
                s_class = s[:, :, classid]
                if len(s_class[s_class > -1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1]) # only average the non default values
            else:
                s_class = s[:, classid]
                if len(s_class[s_class > -1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1])
        return mean_s

    def _summarizeKps():
        stats = np.zeros((12,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=1, olsThr=.5)
        stats[2] = _summarize(ap=1, olsThr=.6)
        stats[3] = _summarize(ap=1, olsThr=.7)
        stats[4] = _summarize(ap=1, olsThr=.8)
        stats[5] = _summarize(ap=1, olsThr=.9)
        stats[6] = _summarize(ap=0)
        stats[7] = _summarize(ap=0, olsThr=.5)
        stats[8] = _summarize(ap=0, olsThr=.6)
        stats[9] = _summarize(ap=0, olsThr=.7)
        stats[10] = _summarize(ap=0, olsThr=.8)
        stats[11] = _summarize(ap=0, olsThr=.9)
        return stats

    def _summarizeKps_cur():
        stats = np.zeros((2,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=0)
        return stats

    if gl:
        summarize = _summarizeKps
    else:
        summarize = _summarizeKps_cur # only return average precision (AP) and average recall (AR)

    stats = summarize()
    return stats

def compute_ols_dts_gts(gts_dict, dts_dict, imgId, catId):
    """
    Compute OLS between detections and gts for a category in a frame.
    If the detected 3D bbox center falls inside gt 3D bbox, then the distance to gt is zero.
    """
    # todo: modify method for distance claculation
    gts = gts_dict[(imgId, catId)]
    dts = dts_dict[(imgId, catId)]
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    if len(gts) == 0 or len(dts) == 0:
        return []
    olss = np.zeros((len(dts), len(gts)))
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        for i, dt in enumerate(dts):
            olss[i, j] = get_ols_btw_objects(gt, dt)
    return olss



def get_3dbbox_corners(quat, tvec, size):
    translation = t3d.Translate(torch.tensor(tvec).to(torch.float32).reshape(-1, 3))
    quat = torch.tensor([quat[3], *quat[:3]]).to(torch.float32).reshape(-1, 4)
    R = quaternion_to_matrix(quat)
    rotation = t3d.Rotate(R=R.transpose(1, 2))  # Need to transpose to make it work.

    tfm = rotation.compose(translation)

    _corners = 0.5 * quat.new_tensor(BOX3D_CORNER_MAPPING).T # (8, 3)
    size = torch.tensor(size).to(torch.float32).reshape(-1, 3) #lwh
    corners_in_obj_frame = size.unsqueeze(1) * _corners.unsqueeze(0) # (1, 1, 3) * (1, 8, 3)

    corners3d = tfm.transform_points(corners_in_obj_frame)

    return corners3d.numpy()[0] # (8, 3)


# points: (N, 2), bboxes: (M, 4, 2)
def get_dist_to_rect_from_p(points, bboxes):
    dist_mat = np.zeros((points.shape[0], bboxes.shape[0]))

    for pid, p in enumerate(points):
        line_starts = bboxes.copy()
        line_ends = np.zeros_like(bboxes)
        line_ends[:, :3, :] = bboxes[:, 1:, :]
        line_ends[:, 3, :] = bboxes[:, 0, :]
        line_starts = np.reshape(line_starts, (-1, 2))
        line_ends = np.reshape(line_ends, (-1, 2))

        dists = lineseg_dists(p, line_starts, line_ends)
        dists = np.reshape(dists, (-1, 4))
        dists = np.min(dists, axis=1)

        dist_mat[pid] = dists

    return dist_mat


def lineseg_dists(p, a, b):
    """Cartesian distance from point to line segment
    Edited to support arguments as series, from:
    https://stackoverflow.com/a/54442561/11208892
    Args:
        - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
        - a: np.array of shape (x, 2)
        - b: np.array of shape (x, 2)
    """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

def get_ols_btw_objects(obj1, obj2):
    """
    Calculate OLS between two objects.
    :param obj1: object 1 dict (gt)
    :param obj2: object 2 dict (detection)
    :return: OLS value
    """
    if obj1['class_id'] != obj2['class_id']:
        print('Error: Computing OLS between different classes!')
        raise TypeError("OLS can only be compute between objects with same class.  ")
    if obj1['score'] < obj2['score']:
        raise TypeError("Confidence score of obj1 should not be smaller than obj2. "
                        "obj1['score'] = %s, obj2['score'] = %s" % (obj1['score'], obj2['score']))


    pos1 = obj1['pos']
    l, w, h = obj1['lwh']
    quat1 = obj1['quat']
    # project 3D bbox to get top 4 corners of bbox
    
    bbox3d_corners1 = get_3dbbox_corners(quat1, pos1, [l, w, h])
    bbox_bev_corners = bbox3d_corners1[[0, 1, 5, 4]][:, [0, 2]] # get top plane's x and z for bev which is [0, 2] in the 2nd dim, (4, 2)
    x1, z1 = pos1[0], pos1[2]
    x2 = obj2['x']
    z2 = obj2['z']
    point = np.array([x2, z2])
    # check if the points fall inside the bev_bbox
    inside_box = True
    for line_seg_idx in line_seg_idxs:
        line_start = bbox_bev_corners[line_seg_idx[0]]
        line_end = bbox_bev_corners[line_seg_idx[1]]
        pa_vec, ab_vec, bp_vec = line_start - point, line_end - line_start, point - line_end
        if np.dot(pa_vec, ab_vec) * np.dot(bp_vec, ab_vec) < 0:
            inside_box = False
            break

    dx = x1 - x2
    dz = z1 - z2
    dist = get_dist_to_rect_from_p(np.array([x2, z2]).reshape((-1, 2)), bbox_bev_corners[None, :, :])
    dist = 0. if inside_box else dist.flatten()[0]
    
    bbox_size_max = np.array([l, w]).max()
    return ols(dist, bbox_size_max)


# bbox_size: max(gt bbox w, gt bbox l), w: hyper parameter
def ols(dist, bbox_size, w=1.):
    """Calculate OLS based on distance, gt 3d bbox size"""
    e = dist ** 2 / bbox_size / w
    return math.exp(-e)

def evaluate_seq(pred, gt, start_end_frame):
    olss_all = {(imgId, catId): compute_ols_dts_gts(gt, pred, imgId, catId) \
                for imgId in range(start_end_frame[0], start_end_frame[1]+1)
                for catId in range(len(OBJ_CLASSES))}

    evalImgs = [evaluate_img(gt, pred, imgId, catId, olss_all, olsThrs, recThrs, OBJ_CLASSES)
                for imgId in range(start_end_frame[0], start_end_frame[1]+1)
                for catId in range(len(OBJ_CLASSES))]

    return evalImgs


def read_pred(pred_files):
    pred_files_str = ', '.join(pred_files)
    print(f'Read in prediction from {pred_files_str}')
    n_class = len(OBJ_CLASSES)
    seqs_to_preds = {}
    seq_to_start_end_frames = {}
    pred_file = {}
    for file in pred_files:
        with open(file, 'r') as file_obj:
            pred_file.update(json.load(file_obj))
    for seq_name, frames in tqdm(pred_file.items()):
        if seq_name not in ['2021_1120_1632', '2022_0203_1441']:
            continue
        end_frame_eval = len(os.listdir(os.path.join(cruw_root, seq_name, 'camera', 'left'))) - 1
        seq_to_start_end_frames[seq_name] = (start_frame_eval, end_frame_eval)
        dts = {(i, j): [] for i in range(start_frame_eval, end_frame_eval+1) for j in range(n_class)}
        id = 1
        for frame_name, objs in frames.items():
            frame_id = int(frame_name)
            for obj in objs:
                if obj['category'] in OBJ_CLASSES:
                    obj_tmp = {}
                    obj_tmp['id'] = id
                    obj_tmp['frame_id'] = frame_id
                    obj_tmp['x'] = obj['bbox3d'][4] * shfit_ratio_xyz[seq_name][0]
                    obj_tmp['z'] = obj['bbox3d'][6] * shfit_ratio_xyz[seq_name][2]
                    obj_tmp['class_id'] = obj['category_id']
                    obj_tmp['score'] = obj['score_3d']
                    dts[frame_id, obj['category_id']].append(obj_tmp)
        seqs_to_preds[seq_name] = dts
    return seqs_to_preds, seq_to_start_end_frames

# seq_to_start_end_frames: only evalute frames in-between (dict of tuple)
def read_gt(gt_file, seq_to_start_end_frames):
    n_class = len(OBJ_CLASSES) # todo: remove hardcoding
    gt_tmp, seq_to_frame_counts = defaultdict(dict), dict()
    with open(gt_file, 'r') as gt_file:
        gt_json = json.load(gt_file)['train']
    print('Generating gt tmp\n')
    for seq_frame in tqdm(gt_json):
        seq_name = seq_frame['seq_name']
        if seq_name not in ['2021_1120_1632', '2022_0203_1441']:
            continue
        frame_id = int(seq_frame['frame_name'])
        if seq_name not in seq_to_frame_counts:
            seq_to_frame_counts[seq_name] = len(os.listdir(os.path.join(cruw_root, seq_name, 'camera', 'left')))
        frame_info = {}
        objs = []
        for obj in seq_frame['objs']:
            obj_tmp = {}
            obj_tmp['frame_id'] = frame_id
            obj_tmp['pos'] = obj['position']
            obj_tmp['lwh'] = obj['scale']
            obj_tmp['quat'] = obj['quat']
            obj_tmp['class_name'] = obj['obj_type']
            obj_tmp['class_id'] = obj_type_to_obj_id[obj['obj_type']]

            objs.append(obj_tmp)

        frame_info['objs'] = objs
        gt_tmp[seq_name].update({frame_id: frame_info})
        
    # gt: {seq: [frame1_info, frame2_info, ...]} 
    seqs_to_gts = {}
    print('Generating gt \n')
    for seq_name, frame_counts in tqdm(seq_to_frame_counts.items()):
        if not seq_name in seq_to_start_end_frames:
            continue
        start_frame_eval, end_frame_eval = seq_to_start_end_frames[seq_name]
        gts = {(i, j): [] for i in range(start_frame_eval, end_frame_eval+1) for j in range(n_class)}
        id = 1
        for frame_id in range(start_frame_eval, end_frame_eval+1):
            frame_info = None
            if frame_id in gt_tmp[seq_name]:
                frame_info = gt_tmp[seq_name][frame_id]
            # for each frame
            if frame_info is None:
                continue
            for obj_dict in frame_info['objs']:
                class_id = obj_dict['class_id']
                # maybe can do the filtering by obj distance
                obj_dict_gt = obj_dict.copy()
                obj_dict_gt['id'] = id
                obj_dict_gt['score'] = 1.0
                gts[frame_id, class_id].append(obj_dict_gt)
                id += 1
        seqs_to_gts[seq_name] = gts
    
    return seqs_to_gts
    #gts: {(frame_id, class_id): [obj dict]}


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate CRUW localization performance')
    parser.add_argument('--pred_file', type=str, default='/mnt/disk1/DA/kitti_v99_cruw/bbox3d_predictions_3dnms_0.0.json', help='prediction json file') #/mnt/nas_cruw/neurips_exp/cam_detection_result/dd3d/dla34/day_night/bbox3d_predictions_3dnms_0.0.json
    parser.add_argument('--gt_file', type=str, default='/mnt/nas_cruw/data/Day_Night_all.json', help='ground truth json file')
    parser.add_argument('--save_dir', type=str, default='/home/andy/Desktop/v99_cruw.txt', help='directory to save testing results')
    args = parser.parse_args()
    return args

OBJ_CLASSES = ['Car', 'Pedestrian']
cruw_root = '/mnt/nas_cruw/CRUW_2022'
obj_type_to_obj_id = {obj_type: idx for idx, obj_type in enumerate(OBJ_CLASSES)}
start_frame_eval=1260
ols_threshold, recall_threshold = (0.5, 0.9, 0.05), (0.0, 1.0, 0.01) # (start, end, step)
# todo: change olsThrs if necessary
olsThrs = np.around(np.linspace(ols_threshold[0], ols_threshold[1], int(np.round((ols_threshold[1] - ols_threshold[0]) / ols_threshold[2]) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(recall_threshold[0], recall_threshold[1], int(np.round((recall_threshold[1] - recall_threshold[0]) / recall_threshold[2]) + 1), endpoint=True), decimals=2)

def read_shift_ratio():
    shift_ratio = {}
    with open('/home/andy/Downloads/ratios.txt', 'r') as f:
        for line in f.readlines():
            seq, ratios = line.split(',')
            ratios = ratios.split(' ')
            # shift_ratio[seq] = [float(ratio) for ratio in ratios]
            shift_ratio[seq] = [1, 1, 1]
    return shift_ratio

def main():
    args = parse_args()
    seqs_to_preds, seq_to_start_end_frames = read_pred([args.pred_file])
    seqs_to_gts = read_gt(args.gt_file, seq_to_start_end_frames)
    save_dir_root = os.path.split(args.save_dir)[0]
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)
    save_string = ''
    evalImgs_all = []
    n_frames_all = 0

    for seq_name, pred in seqs_to_preds.items():
        if seq_name not in ['2021_1120_1632', '2022_0203_1441']:
            continue
        evalImgs = evaluate_seq(pred, seqs_to_gts[seq_name], seq_to_start_end_frames[seq_name])
        eval = accumulate(evalImgs, seq_to_start_end_frames[seq_name], olsThrs, recThrs, OBJ_CLASSES, log=False)
        stats = summarize(eval, olsThrs, recThrs, OBJ_CLASSES, gl=False)
        log_string = "%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100)
        save_string += (log_string+'\n')
        print(log_string)

        n_frames_all += (seq_to_start_end_frames[seq_name][1] - seq_to_start_end_frames[seq_name][0] + 1)
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, (0, n_frames_all-1), olsThrs, recThrs, OBJ_CLASSES, log=False)
    stats = summarize(eval, olsThrs, recThrs, OBJ_CLASSES, gl=False)
    log_string = "%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100)
    save_string += (log_string + '\n\n')
    save_string += 'OlS threshold (start, end, step): {}, Recall threshold: {}\n'.format(ols_threshold, recall_threshold)
    save_string += 'Ground truth file: {}\n'.format(args.gt_file)
    save_string += 'Prediction file: {}\n'.format(args.pred_file)
    save_string += 'Object Classes: {}'.format(OBJ_CLASSES)
    print(log_string)
    with open(args.save_dir, 'w') as out_file:
        out_file.write(save_string)


if __name__ == '__main__':
    main()
    # quat = [0.56, -0.41, 0.42, 0.56]
    # tvec = [2.93, 1.60, 20.77]
    # size = [5.76, 2.12, 1.93]
    # bbox3d_corners1 = get_3dbbox_corners(quat, tvec, size)
    # bbox_bev_corners = bbox3d_corners1[[0, 1, 5, 4]][:, [0, 2]] # get top plane's x and z for bev which is [0, 2] in the 2nd dim, (4, 2)
    # print(bbox_bev_corners)
    # print(np.sqrt(np.power(bbox_bev_corners[0] - bbox_bev_corners[1], 2).sum(0))) 