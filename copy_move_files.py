import os
from tqdm import tqdm
import shutil
import numpy as np

def copy_files(data_root, src_dirs, dst_dir_seqs, seq_to_skip):
    for seq_name in tqdm(sorted(os.listdir(data_root))):
        if int(seq_name) in seq_to_skip:
            continue
        dst_dir = ''
        for k, v in dst_dir_seqs.items():
            if int(seq_name) in v:
                dst_dir = k
        seq_dst_dir = os.path.join(dst_dir, seq_name)
        if not os.path.exists(seq_dst_dir):
            os.makedirs(seq_dst_dir)
        seq_dir = os.path.join(data_root, seq_name)
        for src_folder in src_dirs:
            src_dir = os.path.join(seq_dir, src_folder)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, seq_dst_dir, dirs_exist_ok=True)

if __name__ == '__main__':
    kradar_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    seq_to_skip = [51, 52, 57, 58]
    dst_dir_seqs = {
        "/mnt/ssd1/kradar_dataset/radar_tensor_zyx": np.arange(1, 20).tolist(),
        "/mnt/ssd2/kradar_dataset/radar_tensor_zyx": np.arange(20, 59).tolist()
    }
    for k, v in dst_dir_seqs.items():
        for seq in seq_to_skip:
            if seq in v:
                v.remove(seq)
        dst_dir_seqs[k] = v
    src_dirs = ['radar_zyx_cube_npy_f32']
    copy_files(kradar_root, src_dirs, dst_dir_seqs, seq_to_skip)