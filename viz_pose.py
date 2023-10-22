from scipy import io
import numpy as np
import math
import re
import os
import json
import matplotlib.pylab as plt
from write_video import write_video

# Util Functions

def P_RC(results):
    R_mat = np.array([[0,0,1],[-1,0,0],[0,-1,0]])
    t_vec = np.array([0.3048, 0, 0])
    R_T = np.identity(4)
    R_T[:3,:3] = R_mat
    R_T[3,:3] = t_vec
    results = np.concatenate([results, np.ones([results.shape[0], 1])], axis=1)
    cal_targets = results @ R_T
    return cal_targets[:, :3]


def read_meta():
    with open('/mnt/nas_cruw_pose/file_meta.txt', 'r') as f:
        lines = f.readlines()
    seq_id_to_name = {}
    for line in lines:
        seq_id, seq_name = line.strip().split(',')
        seq_id_to_name[seq_id] = seq_name
    return seq_id_to_name    

def keypoint_vis(results, gt, ax, yaw, score):
    ax.clear()
    skeletons = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
                [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], 
                [12, 13], [13, 14]]
    pose = P_RC(results)
    gt_pose = P_RC(gt)
    max_range = np.array([6, 6, 3]).max() / 2.0
    mid_x = 0.0
    mid_y = 3.0
    mid_z = 1.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    for index, skeleton in enumerate(skeletons):
        ax.plot([pose[skeleton[0], 0], pose[skeleton[1], 0]],
                [pose[skeleton[0], 2], pose[skeleton[1], 2]],
                [-pose[skeleton[0], 1], -pose[skeleton[1], 1]], 'r')
    for index, skeleton in enumerate(skeletons):
        ax.plot([gt_pose[skeleton[0], 0], gt_pose[skeleton[1], 0]],
                [gt_pose[skeleton[0], 2], gt_pose[skeleton[1], 2]],
                [-gt_pose[skeleton[0], 1], -gt_pose[skeleton[1], 1]], 'b')                
    ax.view_init(0, yaw)


if __name__ == '__main__':
    seq_id_to_name = read_meta()
    
    # read in prediction results
    with open('work_dirs/hr3d_one_hm_doppler/20231017_144700/epoch_90/test_prediction.json', 'r') as f:
        results = json.load(f)

    # read in ground truth
    with open('/mnt/nas_cruw_pose/Test.json', 'r') as f:
        gt = json.load(f)

    save_root_viz_dir = 'work_dirs/hr3d_one_hm_doppler/20231017_144700/epoch_90/viz'    
    fig = plt.figure(figsize=[12, 8])
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("Front View")
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title("Side View")
    plt.tight_layout()

    for seq, frames in results.items():
        save_seq_viz_dir = os.path.join(save_root_viz_dir, seq_id_to_name[str(seq)])
        os.makedirs(save_seq_viz_dir, exist_ok=True)
        for frame_rdr_frame, val in sorted(frames.items(), key=lambda x: int(x[0])):
            frame, rdr_frame = frame_rdr_frame.split('_')
            gt_points = gt[seq][frame][0]['pose']
            keypoints = val['keypoints']
            point_class_to_keypoints = {}
            for keypoint in keypoints:
                point_class, x, y, z, score = keypoint
                point_class_to_keypoints[point_class] = [x, y, z]
            key_points_for_viz = []
            for i in range(15):
                if i in point_class_to_keypoints:
                    key_points_for_viz.append(point_class_to_keypoints[i])
                else:
                    key_points_for_viz.append([0, 0, 0])
            keypoint_vis(np.array(key_points_for_viz), np.array(gt_points), ax1, -90, score)
            keypoint_vis(np.array(key_points_for_viz), np.array(gt_points), ax2, 0, score)
            fig.suptitle(f' {seq}/{frame}/{rdr_frame}, {score:.2f}')
            plt.savefig(os.path.join(save_seq_viz_dir, f'{frame}_{rdr_frame}.png'))
        write_video(save_seq_viz_dir, save_root_viz_dir, f'{seq_id_to_name[str(seq)]}.mp4', 10)
    plt.close()
