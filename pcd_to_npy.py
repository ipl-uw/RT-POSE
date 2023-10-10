from pclpy import pcl
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    data_root = '/mnt/nas_cruw_pose'
    seqs = list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root)))
    cloud = pcl.PointCloud.PointXYZI()
    for seq in tqdm(seqs):
        seq_raw_lidar_dir = os.path.join(data_root, seq, 'lidar_raw')
        seq_lidar_npy_dir = os.path.join(data_root, seq, 'lidar_npy')
        os.makedirs(seq_lidar_npy_dir, exist_ok=True)
        for pcd_file in os.listdir(seq_raw_lidar_dir):
            pcl.io.loadPCDFile(os.path.join(seq_raw_lidar_dir, pcd_file), cloud)
            pc_np = np.zeros((len(cloud.x), 4), dtype=np.float32)
            pc_np[:, 0] = cloud.x
            pc_np[:, 1] = cloud.y
            pc_np[:, 2] = cloud.z
            pc_np[:, 3] = cloud.intensity
            np.save(os.path.join(seq_lidar_npy_dir, pcd_file[:-4] + '.npy'), pc_np)
            cloud.clear()
