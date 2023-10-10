import numpy as np
import os
from tqdm import tqdm
import json

if __name__ == '__main__':
    data_root = '/mnt/nas_cruw_pose'
    save_result_path = 'lidar_pc_analysis.json'
    seqs = list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root)))
    analysis_result = {}
    x_min_all, x_max_all = 1000, -1000
    y_min_all, y_max_all = 1000, -1000
    z_min_all, z_max_all = 1000, -1000
    for seq in tqdm(seqs):
        x_min, x_max = 1000, -1000
        y_min, y_max = 1000, -1000
        z_min, z_max = 1000, -1000
        seq_lidar_npy_dir = os.path.join(data_root, seq, 'lidar_npy')
        for npy_file in os.listdir(seq_lidar_npy_dir):
            lidar_pc = np.load(os.path.join(seq_lidar_npy_dir, npy_file))
            x_min = min(x_min, np.min(lidar_pc[:, 0]).item() )
            x_max = max(x_max, np.max(lidar_pc[:, 0]).item())
            y_min = min(y_min, np.min(lidar_pc[:, 1]).item())
            y_max = max(y_max, np.max(lidar_pc[:, 1]).item())
            z_min = min(z_min, np.min(lidar_pc[:, 2]).item())
            z_max = max(z_max, np.max(lidar_pc[:, 2]).item())
        analysis_result[seq] = {'x': [x_min, x_max], 'y': [y_min, y_max], 'z': [z_min, z_max]}
        x_min_all = min(x_min_all, x_min)
        x_max_all = max(x_max_all, x_max)
        y_min_all = min(y_min_all, y_min)
        y_max_all = max(y_max_all, y_max)
        z_min_all = min(z_min_all, z_min)
        z_max_all = max(z_max_all, z_max)
    analysis_result['all'] = {'x': [x_min_all, x_max_all], 'y': [y_min_all, y_max_all], 'z': [z_min_all, z_max_all]}
    with open(os.path.join(data_root, save_result_path), 'w') as f:
        json.dump(analysis_result, f, indent=2)