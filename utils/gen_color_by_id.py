import os
import random
import json
from tqdm import tqdm


def rand_color():
    return '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])


tracking_id_to_color_seq = {}

seq_to_save = [9, 10, 23, 24, 28, 39, 43, 44, 45, 48, 50, 54]
seq_to_save = [str(seq) for seq in seq_to_save]

pred_files_root = '/mnt/nas_kradar/kradar_dataset/detection_result/icra/triplet_cosine_one_pos_hard_neg/SCT_radar_tracking_0914_icra_triplet_cosine_one_pos_hard_neg_diou'

for seq in tqdm(seq_to_save):
    tracking_id_to_color = {}
    pred_file = os.path.join(pred_files_root, f'{seq}_predbbx.txt')
    with open(pred_file, 'r') as f:
        tracklets_lines = f.readlines()
    for line in tracklets_lines:
        tracklet_id = line.strip().split(',')[1]
        if tracklet_id not in tracking_id_to_color:
            tracking_id_to_color[tracklet_id] = rand_color()
    tracking_id_to_color_seq[seq] = tracking_id_to_color

with open('/mnt/nas_kradar/kradar_dataset/detection_result/icra/triplet_cosine_one_pos_hard_neg/color_table.json', 'w') as f:
    json.dump(tracking_id_to_color_seq, f, indent=2)