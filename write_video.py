from logging import root
import os
# import imageio
import cv2
import glob
from tqdm import tqdm

def write_video(dir, save_dir, save_name='out.mp4', fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_list = sorted(glob.glob(os.path.join(dir, '*.[jp][pn]g')))
    shape = cv2.imread(frame_list[0]).shape # delete dimension 3
    size = (shape[1], shape[0])
    out = cv2.VideoWriter(os.path.join(save_dir, save_name), fourcc, fps, size)
    for idx, path in enumerate(frame_list):
        frame = cv2.imread(path)
        current_frame = idx+1
        total_frame_count = len(frame_list)
        percentage = int(current_frame*30 / (total_frame_count+1))
        print("\rProcess: [{}{}] {:06d} / {:06d}".format("#"*percentage, "."*(30-1-percentage), current_frame, total_frame_count), end ='')
        out.write(frame)
    out.release()


if __name__ == '__main__':
    # seq_dir = '/mnt/nas_kradar/kradar_dataset/dir_1to20/14/k_radar_3dResUnet_conf03'
    # write_video(seq_dir, seq_dir)    
    
    root_dir = '/mnt/nas_cruw_pose/pose_viz'
    save_video_dir = '/mnt/nas_cruw_pose/pose_viz_video'
    os.makedirs(save_video_dir, exist_ok=True)
    for dir in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, dir)
        print(f'Now processing :{dir}')
        write_video(img_dir, save_video_dir, f'{dir}.mp4')

    # image_path = '/mnt/nas/nas_cruw/CRUW_2022/2021_1120_1634/camera/left_rrdnet_seq'
    # write_video(image_path, '/mnt/nas/nas_cruw/CRUW_2022', save_name='2021_1120_1634_enhanced.mp4')

