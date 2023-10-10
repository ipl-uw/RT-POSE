import os
from tqdm import tqdm
import shutil
import numpy as np

def copy_files(data_root, src_dirs, dst_dir):
    seqs = list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root)))
    for seq_name in tqdm(seqs):
        dst_seq_dir = os.path.join(dst_dir, seq_name)
        if not os.path.exists(dst_seq_dir):
            continue
        src_seq_dir = os.path.join(data_root, seq_name)
        for src_folder in src_dirs:
            src_dir = os.path.join(src_seq_dir, src_folder)
            if os.path.exists(src_dir):
                if os.path.exists(os.path.join(dst_seq_dir, src_folder)):
                    shutil.rmtree(os.path.join(dst_seq_dir, src_folder))
                shutil.copytree(src_dir, os.path.join(dst_seq_dir, src_folder), dirs_exist_ok=True)

if __name__ == '__main__':
    CRUW_POSE_Root = '/mnt/nas_cruw/CRUW3D_POSE'
    dst_dir = '/mnt/nas_cruw_pose'
    src_dirs = ['lidar_raw']
    copy_files(CRUW_POSE_Root, src_dirs, dst_dir)