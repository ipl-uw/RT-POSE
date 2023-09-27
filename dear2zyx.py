import numpy as np
import torch
import os
from det3d.models.feat_transforms import PolarToCart
from utils.viz_kradar_funcs import viz_radar_tensor_bev
from scipy.io import loadmat

# physical valuses of each DEAR tensors' cell
def load_physical_values():
    temp_values = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'info_arr.mat'))
    arr_range = temp_values['arrRange']
    # deg2rad = np.pi/180.
    arr_azimuth = temp_values['arrAzimuth']
    arr_elevation = temp_values['arrElevation']
    arr_range = arr_range.flatten()
    arr_azimuth = arr_azimuth.flatten()
    arr_elevation = arr_elevation.flatten()
    arr_doppler = loadmat(os.path.join('/mnt/ssd1/kradar_dataset', 'resources', 'arr_doppler.mat'))['arr_doppler']
    arr_doppler = arr_doppler.flatten()
    return arr_range, arr_azimuth, arr_elevation, arr_doppler
        

if __name__ == '__main__':
    arr_r, arr_a, arr_e, arr_d = load_physical_values()
    dear = np.flip(np.load('/mnt/ssd1/kradar_dataset/radar_tensor/1/radar_DEAR_D_downsampled_2/tesseract_00102.npy'), axis=1).copy()
    dear[dear < 0.] = 1.
    dear = np.log(dear)
    cart_ROI = {'x': [0., 80.], 'y': [-30., 30.], 'z': [-2, 7.6]}
    polar_range = [0.,  118.03710938, -53, 53, -18, 18]
    voxel_size = 0.4

    ar = dear.max(axis=0).mean(axis=0)
    # dataset_cfg = {}
    # dataset_cfg['x_coord'] = arr_r
    # dataset_cfg['y_coord'] = arr_a
    # ar_bev_save_dir = './polar_bev.png'
    # viz_radar_tensor_bev(dataset_cfg, ar_bev_save_dir, ar, ('Range (m)', 'Azimuth (degree)'))

    polar_to_cart = PolarToCart(cart_ROI, voxel_size, polar_range, dimension='2')
    ar = torch.from_numpy(ar).unsqueeze(0).unsqueeze(0)
    bev_yx = polar_to_cart(ar)[0][0].numpy()
    print(bev_yx.shape)
    dataset_cfg = {}
    dataset_cfg['x_coord'] = np.arange(0, 80, 0.4)
    dataset_cfg['y_coord'] = np.arange(-30, 30, 0.4)
    yx_bev_save_dir = './cart_bev_p2d2d_avg.png'
    viz_radar_tensor_bev(dataset_cfg, yx_bev_save_dir, bev_yx, ('X(m)', 'Y(m)'))




    # dear = torch.tensor(dear).unsqueeze(0)
    
    # polar_to_cart = PolarToCart(cart_ROI, voxel_size, polar_range, dimension='3')
    # dzyx = polar_to_cart(dear)
    # dzyx = dzyx[0].numpy()
    # print(dzyx.shape)
    # radar_bev_yx = dzyx.max(axis=0).mean(axis=0)
    # dataset_cfg = {}
    # dataset_cfg['x_coord'] = np.arange(0, 80, 0.4)
    # dataset_cfg['y_coord'] = np.arange(-30, 30, 0.4)
    # yx_bev_save_dir = './cart_bev.png'
    # viz_radar_tensor_bev(dataset_cfg, yx_bev_save_dir, radar_bev_yx, ('X(m)', 'Y(m)'))
    

    # polar_to_cart = PolarToCart(cart_ROI, voxel_size, polar_range, dimension='2')
    # dar = dear.mean(axis=2)
    # dyx = polar_to_cart(dar)
    # dyx = dyx[0].numpy()
    # print(dyx.shape)
    # radar_bev_yx = dyx.max(axis=0)
    # dataset_cfg = {}
    # dataset_cfg['x_coord'] = np.arange(0, 80, 0.4)
    # dataset_cfg['y_coord'] = np.arange(-30, 30, 0.4)
    # yx_bev_save_dir = './cart_bev_p2d2d.png'
    # viz_radar_tensor_bev(dataset_cfg, yx_bev_save_dir, radar_bev_yx, ('X(m)', 'Y(m)'))
