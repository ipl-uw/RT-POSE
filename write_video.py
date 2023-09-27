from logging import root
import os
import imageio
from tqdm import tqdm

def write_video(image_path, save_dir, save_name='out.mp4', fps=20):
    writer = imageio.get_writer(os.path.join(save_dir, save_name), fps=fps)

    for image in tqdm(sorted(os.listdir(image_path))):
        if image.endswith(('.jpg', '.png')):
            img = imageio.imread(os.path.join(image_path, image))
            writer.append_data(img)
    writer.close()


if __name__ == '__main__':
    # seq_dir = '/mnt/nas_kradar/kradar_dataset/dir_1to20/14/k_radar_3dResUnet_conf03'
    # write_video(seq_dir, seq_dir)    
    
    root_dir = '/mnt/nas_kradar/kradar_dataset/dir_all'
    save_video_dir = '/home/andy/Desktop/neurlips_video'
    os.makedirs(save_video_dir, exist_ok=True)
    for dir in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, dir, 'inference_viz', 'HRTiny_feat16_small_clean_final_concat_epoch21_nomag_0.3')
        if not os.path.exists(img_dir):
            continue
        print(f'Now processing :{dir}')
        write_video(img_dir, save_video_dir, f'{dir}.mp4')

    # image_path = '/mnt/nas/nas_cruw/CRUW_2022/2021_1120_1634/camera/left_rrdnet_seq'
    # write_video(image_path, '/mnt/nas/nas_cruw/CRUW_2022', save_name='2021_1120_1634_enhanced.mp4')

