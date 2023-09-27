
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path as osp
import pickle
import os

from utils.geometry import *
from utils.geometry import Object3D
from matplotlib.patches import Circle
from pathlib import Path
import json 
from PIL import ImageColor
import random

label_to_color = {
    0: '#06d6a0',
    1: '#ffd166'
}

label_to_text = {
    0: 'Sedan',
    1: 'Truck'
}

gt_label_to_color = {
    'Sedan': '#ef476f',
    'Bus or Truck': '#e147ef'
}

def viz_radar_tensor_bev(dataset, save_dir, rdr_tensor_bev, xy_labels):
    min_val = np.min(rdr_tensor_bev[rdr_tensor_bev!=-1.])
    max_val = np.max(rdr_tensor_bev[rdr_tensor_bev!=-1.])
    rdr_tensor_bev[rdr_tensor_bev!=-1.]= (rdr_tensor_bev[rdr_tensor_bev!=-1.]-min_val)/(max_val-min_val)

    arr_0, arr_1 = np.meshgrid(dataset['x_coord'], dataset['y_coord'])
    ### Jet map visualization ###
    rdr_tensor_bev[rdr_tensor_bev==0.] = -np.inf # for visualization
    fig, axes = plt.subplots(1, 1, figsize=(9, 9))
    # show corresponding image
    axes.set_title('Radar Tensor Power BEV')
    axes.set_xlabel(xy_labels[0])
    axes.set_ylabel(xy_labels[1])
    axes.set_aspect(1)
    # x_plot_loc = np.linspace(0, rdr_tensor_bev.shape[1], 11).astype(int)
    # y_plot_loc = np.linspace(0, rdr_tensor_bev.shape[0], 11).astype(int)
    # axes.set_xticks(x_plot_loc, -( x_plot_loc/rdr_tensor_bev.shape[1] * (dataset.arr_x_cb[-1] - dataset.arr_x_cb[0])\
    #     + dataset.arr_x_cb[0]).round(2))
    # axes.set_yticks(y_plot_loc, (-y_plot_loc/rdr_tensor_bev.shape[0] * (dataset.arr_y_cb[-1] - dataset.arr_y_cb[0])\
    #     + dataset.arr_y_cb[0]).round(2))
    mesh = axes.pcolormesh(arr_0, arr_1, rdr_tensor_bev, cmap='jet')
    fig.colorbar(mesh) # TODO: add this back
    
    fig.savefig(save_dir)
    plt.close()


def func_show_radar_cube_bev(dataset, meta, bboxes, save_dir, conf_thres, seq, magnifying=1., is_with_log = False):
    if meta['path']['rdr_cube'][-3:] == 'mat':
        dir_name = meta['path']['rdr_cube'].split('/')[-2]
        file_name = meta['path']['rdr_cube'].split('/')[-1]
        dir_name = f'{dir_name}_npy_f32'
        file_name = file_name[:-3] + 'npy'
        meta['path']['rdr_cube'] = os.path.join('/', *meta['path']['rdr_cube'].split('/')[:-2], dir_name,  file_name)
    # rdr_cube, rdr_cube_mask, rdr_cube_cnt = dataset.get_cube(meta['path']['rdr_cube'], mode=0)
    # if is_with_doppler:
    #     rdr_cube_doppler = dataset.get_cube_doppler(meta['path']['rdr_cube_doppler'])
    
    file_name = meta['path']['rdr_cube'].split('/')[-1]
    rdr_cube_path = meta['path']['rdr_cube'].split('/')[:-2]
    # rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_f32', file_name)
    # rdr_cube = dataset.get_cube(rdr_cube_path, mode=1)
    # rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_roi', file_name)
    rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_viz_roi', file_name)
    rdr_cube = dataset.get_cube_direct(rdr_cube_path)

    rdr_cube = np.mean(rdr_cube, axis=0)
    # log
    if is_with_log:
        rdr_cube = np.maximum(rdr_cube, 1.)
        rdr_cube_bev = 10*np.log10(rdr_cube)
    else:
        rdr_cube_bev = rdr_cube

    box3d, scores, pred_labels = bboxes

    normalizing = 'min_max'
    if normalizing == 'max':
        rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
    elif normalizing == 'fixed':
        rdr_cube_bev = rdr_cube_bev/dataset.cfg.DATASET.RDR_CUBE.NORMALIZING.VALUE
    elif normalizing == 'min_max':
        ### Care for Zero parts ###
        min_val = np.min(rdr_cube_bev[rdr_cube_bev!=0.])
        max_val = np.max(rdr_cube_bev[rdr_cube_bev!=0.])
        rdr_cube_bev[rdr_cube_bev!=0.]= (rdr_cube_bev[rdr_cube_bev!=0.]-min_val)/(max_val-min_val)

    arr_0, arr_1 = np.meshgrid(dataset.arr_x_cb, dataset.arr_y_cb)
    ### Jet map visualization ###
    rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    # show corresponding image
    cam_img = cv2.imread(meta['path']['cam_front_img'])[:, :, ::-1]
    axes[1].imshow(cam_img[:, :cam_img.shape[1]//2, :])
    axes[1].set_title('Cam Front Left')
    axes[0].set_title('Radar Cube Power BEV')
    axes[0].set_xlabel('X(m)')
    axes[0].set_ylabel('Y(m)')
    axes[0].set_aspect(1)
    # x_plot_loc = np.linspace(0, rdr_cube_bev.shape[1], 11).astype(int)
    # y_plot_loc = np.linspace(0, rdr_cube_bev.shape[0], 11).astype(int)
    # axes[0].set_xticks(x_plot_loc, -( x_plot_loc/rdr_cube_bev.shape[1] * (dataset.arr_x_cb[-1] - dataset.arr_x_cb[0])\
    #     + dataset.arr_x_cb[0]).round(2))
    # axes[0].set_yticks(y_plot_loc, (-y_plot_loc/rdr_cube_bev.shape[0] * (dataset.arr_y_cb[-1] - dataset.arr_y_cb[0])\
    #     + dataset.arr_y_cb[0]).round(2))
    mesh = axes[0].pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    fig.colorbar(mesh) # TODO: add this back
    
    for box_id, bbox in enumerate(box3d):
        if scores[box_id] < conf_thres:
            continue
        pred_label = pred_labels[box_id]
        x, y, z, xl, yl, zl, theta = bbox        
        xl, yl, zl = magnifying * xl, magnifying * yl, magnifying * zl
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)
        top_corners = obj3d.corners[[0, 4, 6, 2],:] # from the front left corner, counter-clocwise
        circle = Circle((x, y), radius=0, linewidth=4, edgecolor=label_to_color[pred_label], facecolor=label_to_color[pred_label])
        axes[0].add_patch(circle)
        draw_sequence = [1, 2, 3, 0]
        for begin_draw_idx, end_draw_idx in enumerate(draw_sequence):
            axes[0].plot((top_corners[begin_draw_idx][0], top_corners[end_draw_idx][0]), (top_corners[begin_draw_idx][1], top_corners[end_draw_idx][1])\
                , color=label_to_color[pred_label], linewidth=2.5)
        # Add car's front face            
        front_face = (top_corners[0] + top_corners[-1]) / 2
        axes[0].plot((x, front_face[0]), (y, front_face[1]), \
            color=label_to_color[pred_label], linewidth=2.5)
        box_conf_score = str(scores[box_id].round(2))
        # box_label = pred_labels[box_id] # TODO: add this to viz for multiclass prediction
        left_face = (top_corners[0] + top_corners[1]) / 2
        center_to_leftface = 0.5 * (left_face - bbox[:3]) / np.sqrt(np.square(left_face - bbox[:3]).sum()) # half normalized vector
        # axes[0].text(left_face[0]+center_to_leftface[0], left_face[1]+center_to_leftface[1], f'Sedan, {box_conf_score}', \
        #     rotation=theta/np.pi*180, fontsize=8, color='#FFFFFF', horizontalalignment='center')
        axes[0].text(left_face[0]+center_to_leftface[0], left_face[1]+center_to_leftface[1], f'{label_to_text[pred_label]}, {box_conf_score}', \
            rotation=theta/np.pi*180, fontsize=10, color='#FFFFFF', horizontalalignment='center')


    calib_info = dataset.get_calib_info(meta['path']['path_calib'])
    gt_objs = dataset.get_label_bboxes(meta['path']['path_label'], calib_info)
    for gt_obj in gt_objs: # (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
        cls_name, _, [x, y, z, theta, xl, yl, zl], _ = gt_obj
        xl, yl, zl = magnifying * xl, magnifying * yl, magnifying * zl# TODO: remvoe this, just for viz.
        if cls_name not in ['Sedan']  or x > 71.6 or y > 6 or y < -6.4 or z > 5.6 or z < -2:
            continue
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)
        top_corners = obj3d.corners[[0, 4, 6, 2],:] # from the front left corner, counter-clocwise
        circle = Circle((x, y), radius=0, linewidth=4, edgecolor=gt_label_to_color[cls_name], facecolor=gt_label_to_color[cls_name])
        axes[0].add_patch(circle)
        draw_sequence = [1, 2, 3, 0]
        for begin_draw_idx, end_draw_idx in enumerate(draw_sequence):
            axes[0].plot((top_corners[begin_draw_idx][0], top_corners[end_draw_idx][0]), (top_corners[begin_draw_idx][1], top_corners[end_draw_idx][1])\
                , color=gt_label_to_color[cls_name], linewidth=2.)
        # Add car's front face            
        front_face = (top_corners[0] + top_corners[-1]) / 2
        axes[0].plot((x, front_face[0]), (y, front_face[1]), \
            color=gt_label_to_color[cls_name], linewidth=2.)
        left_face = (top_corners[0] + top_corners[1]) / 2
        center_to_leftface = 0.5 * (left_face - np.array([x, y, z])) / np.sqrt(np.square(left_face - np.array([x, y, z])).sum()) # half normalized vector
    # TODO: show the bbox projection on image
    # plt.show()
    # img_name = os.path.split(meta['path']['path_label'])[1].split('.')[0] + '.jpg'
    img_name = seq + '_' + os.path.split(meta['path']['path_label'])[1].split('.')[0] + '.jpg'
    # plt.setp(axes[0].get_xticklabels()) #, visible=False
    # plt.setp(axes[0].get_yticklabels()) #, visible=False
    # plt.axis('off')
    fig.savefig(os.path.join(save_dir, img_name))
    # fig.savefig(os.path.join('/home/andy/Desktop/neurlips_viz', img_name))
    plt.close()
    

def rand_colors(num_of_objs=1):
    colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for i in range(num_of_objs)]
    colors = [ImageColor.getcolor(h, "RGB") for h in colors]
    return [value/255 for value in colors[0]]

def func_show_radar_cube_bev_tracking(dataset, frame_id, meta, tracklet_frame, viz_dir, tracking_id_to_color, used_colors, magnifying=1., is_with_log = False):


    if meta['path']['rdr_cube'][-3:] == 'mat':
        dir_name = meta['path']['rdr_cube'].split('/')[-2]
        file_name = meta['path']['rdr_cube'].split('/')[-1]
        dir_name = f'{dir_name}_npy_f32'
        file_name = file_name[:-3] + 'npy'
        meta['path']['rdr_cube'] = os.path.join('/', *meta['path']['rdr_cube'].split('/')[:-2], dir_name,  file_name)
    # rdr_cube, rdr_cube_mask, rdr_cube_cnt = dataset.get_cube(meta['path']['rdr_cube'], mode=0)
    # if is_with_doppler:
    #     rdr_cube_doppler = dataset.get_cube_doppler(meta['path']['rdr_cube_doppler'])
    
    file_name = meta['path']['rdr_cube'].split('/')[-1]
    rdr_cube_path = meta['path']['rdr_cube'].split('/')[:-2]
    rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_f32', file_name)
    rdr_cube = dataset.get_cube(rdr_cube_path, mode=1)
    # rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_roi', file_name)
    # rdr_cube_path = osp.join('/', *rdr_cube_path, 'radar_zyx_cube_npy_viz_roi', file_name)
    # rdr_cube = dataset.get_cube_direct(rdr_cube_path)

    rdr_cube = np.mean(rdr_cube, axis=0)
    # log
    if is_with_log:
        rdr_cube = np.maximum(rdr_cube, 1.)
        rdr_cube_bev = 10*np.log10(rdr_cube)
    else:
        rdr_cube_bev = rdr_cube


    normalizing = 'min_max'
    if normalizing == 'max':
        rdr_cube_bev = rdr_cube_bev/np.max(rdr_cube_bev)
    elif normalizing == 'fixed':
        rdr_cube_bev = rdr_cube_bev/dataset.cfg.DATASET.RDR_CUBE.NORMALIZING.VALUE
    elif normalizing == 'min_max':
        ### Care for Zero parts ###
        min_val = np.min(rdr_cube_bev[rdr_cube_bev!=0.])
        max_val = np.max(rdr_cube_bev[rdr_cube_bev!=0.])
        rdr_cube_bev[rdr_cube_bev!=0.]= (rdr_cube_bev[rdr_cube_bev!=0.]-min_val)/(max_val-min_val)

    arr_0, arr_1 = np.meshgrid(dataset.arr_x_cb, dataset.arr_y_cb)
    ### Jet map visualization ###
    rdr_cube_bev[np.where(rdr_cube_bev==0.)] = -np.inf # for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    # show corresponding image
    cam_img = cv2.imread(meta['path']['cam_front_img'])[:, :, ::-1]
    axes[1].imshow(cam_img[:, :cam_img.shape[1]//2, :])
    # axes[1].set_title('Cam Front Left')
    # axes[0].set_title('Radar Cube Power BEV')
    # axes[0].set_xlabel('X(m)')
    # axes[0].set_ylabel('Y(m)')
    axes[0].set_aspect(1)
    # x_plot_loc = np.linspace(0, rdr_cube_bev.shape[1], 11).astype(int)
    # y_plot_loc = np.linspace(0, rdr_cube_bev.shape[0], 11).astype(int)
    # axes[0].set_xticks(x_plot_loc, -( x_plot_loc/rdr_cube_bev.shape[1] * (dataset.arr_x_cb[-1] - dataset.arr_x_cb[0])\
    #     + dataset.arr_x_cb[0]).round(2))
    # axes[0].set_yticks(y_plot_loc, (-y_plot_loc/rdr_cube_bev.shape[0] * (dataset.arr_y_cb[-1] - dataset.arr_y_cb[0])\
    #     + dataset.arr_y_cb[0]).round(2))
    mesh = axes[0].pcolormesh(arr_0, arr_1, rdr_cube_bev, cmap='jet')
    fig.colorbar(mesh) # TODO: add this back
    
    for _, tracklet in enumerate(tracklet_frame):
        tracking_id, x, y, xl, yl, theta, score = tracklet
        x, y, xl, yl, theta = float(x), float(y), float(xl), float(yl), float(theta)

        if tracking_id in tracking_id_to_color:
            clr = tracking_id_to_color[tracking_id]
        else:
            clr = rand_colors()
            while clr in used_colors:
                clr = rand_colors()
            tracking_id_to_color[tracking_id] = clr

        z, zl = 0, 1.8
        xl, yl, zl = magnifying * xl, magnifying * yl, magnifying * zl
        obj3d = Object3D(x, y, z, xl, yl, zl, theta)
        top_corners = obj3d.corners[[0, 4, 6, 2],:] # from the front left corner, counter-clocwise
        circle = Circle((x, y), radius=0, linewidth=4, edgecolor=clr, facecolor=clr)
        axes[0].add_patch(circle)
        draw_sequence = [1, 2, 3, 0]
        for begin_draw_idx, end_draw_idx in enumerate(draw_sequence):
            axes[0].plot((top_corners[begin_draw_idx][0], top_corners[end_draw_idx][0]), (top_corners[begin_draw_idx][1], top_corners[end_draw_idx][1])\
                , color=clr, linewidth=2.5)
        # Add car's front face            
        front_face = (top_corners[0] + top_corners[-1]) / 2
        axes[0].plot((x, front_face[0]), (y, front_face[1]), \
            color=clr, linewidth=2.5)
        # box_label = pred_labels[box_id] # TODO: add this to viz for multiclass prediction
        left_face = (top_corners[0] + top_corners[1]) / 2
        left_face = left_face[:2]
        center_to_leftface = 0.5 * (left_face - np.array([x, y])) / np.sqrt(np.square(left_face - np.array([x, y])).sum()) # half normalized vector
        # axes[0].text(left_face[0]+center_to_leftface[0], left_face[1]+center_to_leftface[1], f'Sedan, {box_conf_score}', \
        #     rotation=theta/np.pi*180, fontsize=8, color='#FFFFFF', horizontalalignment='center')
        axes[0].text(left_face[0]+center_to_leftface[0], left_face[1]+center_to_leftface[1], f'{tracking_id}', \
            rotation=theta/np.pi*180, fontsize=16, color='#FFFFFF', horizontalalignment='center')


    calib_info = dataset.get_calib_info(meta['path']['path_calib'])
    # gt_objs = dataset.get_label_bboxes(meta['path']['path_label'], calib_info)
    # for gt_obj in gt_objs: # (cls_name, idx_cls, [x,y,z,theta,l,w,h], idx_obj)
    #     cls_name, _, [x, y, z, theta, xl, yl, zl], _ = gt_obj
    #     xl, yl, zl = magnifying * xl, magnifying * yl, magnifying * zl# TODO: remvoe this, just for viz.
    #     if cls_name not in ['Sedan', 'Bus or Truck']  or x > 71.6 or y > 6 or y < -6.4 or z > 5.6 or z < -2:
    #         continue
    #     obj3d = Object3D(x, y, z, xl, yl, zl, theta)
    #     top_corners = obj3d.corners[[0, 4, 6, 2],:] # from the front left corner, counter-clocwise
    #     circle = Circle((x, y), radius=0, linewidth=4, edgecolor=gt_label_to_color[cls_name], facecolor=gt_label_to_color[cls_name])
    #     axes[0].add_patch(circle)
    #     draw_sequence = [1, 2, 3, 0]
    #     for begin_draw_idx, end_draw_idx in enumerate(draw_sequence):
    #         axes[0].plot((top_corners[begin_draw_idx][0], top_corners[end_draw_idx][0]), (top_corners[begin_draw_idx][1], top_corners[end_draw_idx][1])\
    #             , color=gt_label_to_color[cls_name], linewidth=2.)
    #     # Add car's front face            
    #     front_face = (top_corners[0] + top_corners[-1]) / 2
    #     axes[0].plot((x, front_face[0]), (y, front_face[1]), \
    #         color=gt_label_to_color[cls_name], linewidth=2.)
    #     left_face = (top_corners[0] + top_corners[1]) / 2
    #     center_to_leftface = 0.5 * (left_face - np.array([x, y, z])) / np.sqrt(np.square(left_face - np.array([x, y, z])).sum()) # half normalized vector
    # TODO: show the bbox projection on image
    # plt.show()
    # img_name = os.path.split(meta['path']['path_label'])[1].split('.')[0] + '.jpg'
    img_name = frame_id + '.jpg'
    # plt.setp(axes[0].get_xticklabels(), visible=False)
    # plt.setp(axes[0].get_yticklabels(), visible=False)
    # plt.axis('off')
    fig.savefig(os.path.join(viz_dir, img_name))
    # fig.savefig(os.path.join('/home/andy/Desktop/neurlips_viz', img_name))
    plt.close()
    return  tracking_id_to_color, used_colors
    


    


def func_show_radar_tensor_bev(dataset, meta, bboxes=None, \
        roi_x = [0, 0.4, 100], roi_y = [-50, 0.4, 50], is_return_bbox_bev_tensor=False, alpha=0.9, lthick=1, infer=None, infer_gt=None, norm_img=None):
    rdr_tensor = dataset.get_tesseract(meta['path']['rdr_tesseract'])
    rdr_bev = np.mean(np.mean(rdr_tensor, axis=0), axis=2)

    arr_range = dataset.arr_range
    arr_azimuth = dataset.arr_azimuth
    arr_0, arr_1 = np.meshgrid(arr_azimuth, arr_range)
    height, width = np.shape(rdr_bev)
    # print(height, width)
    # figsize = (1, height/width) if height>=width else (width/height, 1)
    # plt.figure(figsize=figsize)
    plt.clf()
    plt.cla()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
    plt.axis('off'), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    
    temp_img = cv2.imread('./resources/imgs/img_tes_ra.png')
    temp_row, temp_col, _ = temp_img.shape
    if not (temp_row == height and temp_col == width):
        temp_img_new = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('./resources/imgs/img_tes_ra.png', temp_img_new)

    plt.close()
    plt.pcolormesh(arr_0, arr_1, 10*np.log10(rdr_bev), cmap='jet')
    plt.colorbar()
    plt.savefig('./resources/imgs/plot_tes_ra.png', dpi=300)

    # Polar to Cartesian (Should flip image)
    ra = cv2.imread('./resources/imgs/img_tes_ra.png')
    ra = np.flip(ra, axis=0)

    arr_yx, arr_y, arr_x  = get_xy_from_ra_color(ra, arr_range, arr_azimuth, \
        roi_x = roi_x, roi_y = roi_y, is_in_deg=False)

    ### Image processing ###
    if norm_img is None:
        pass
    elif norm_img.split('_')[0] == 'hist': # histogram stretching
        arr_yx = cv2.normalize(arr_yx, None, 0, 255, cv2.NORM_MINMAX)
    elif norm_img.split('_')[0] == 'alp':
        alp = float(norm_img.split('_')[1])
        arr_yx = np.clip((1+alp)*arr_yx - 128*alp, 0, 255).astype(np.uint8)

    if not (bboxes is None):
        ### Original ###
        # arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx, arr_y, arr_x, bboxes, lthick=lthick)
        # if is_return_bbox_bev_tensor:
        #     return arr_yx_bbox
        arr_yx_bbox = arr_yx.copy()

        ### inference ###
        if infer_gt is not None:
            bboxes_gt = []
            for idx_obj, obj in enumerate(infer_gt):
                bboxes_gt.append(['Gt Sedan', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
            arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_gt, lthick=lthick)
        
        if infer is not None:
            # print(infer)
            bboxes_infer = []
            for idx_obj, obj in enumerate(infer):
                bboxes_infer.append(['Infer', 0, [obj.xc, obj.yc, obj.zc, obj.rot_rad, obj.xl, obj.yl, obj.zl], idx_obj])
            arr_yx_bbox = draw_bbox_in_yx_bgr(arr_yx_bbox, arr_y, arr_x, bboxes_infer, lthick=lthick)

    # alpha = 0.9
    arr_yx_bbox = cv2.addWeighted(arr_yx_bbox, alpha, arr_yx, 1 - alpha, 0)

    # flip before show
    # arr_yx = arr_yx.transpose((1,0,2))
    # arr_yx = np.flip(arr_yx, axis=(0,1))

    # print(meta)
    # cv2.imshow('Cartesian', arr_yx)

    if not (bboxes is None):
        arr_yx_bbox = arr_yx_bbox.transpose((1,0,2))
        arr_yx_bbox = np.flip(arr_yx_bbox, axis=(0,1))
        cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx_bbox,(0,0),fx=4,fy=4))
    else:
        arr_yx = arr_yx.transpose((1,0,2))
        arr_yx = np.flip(arr_yx_bbox, axis=(0,1))
        cv2.imshow('Cartesian (bbox)', cv2.resize(arr_yx,(0,0),fx=2,fy=2))

    # cv2.imshow('Front image', cv2.imread(meta['path_cam_front'])[:,:1280,:])
    # plt.show()
    cv2.waitKey(0)


# import open3d as o3d

# os.environ['OPEN3D_CPU_RENDERING'] = 'true' 
# o3d_viz_cam_param_file = '/home/andy/Desktop/o3d_viz_cam_param/o3d_cam.json'
# def key_action_callback(vis, key, action):
#     # If the space key is pressed, save the camera parameters
#     if key == ord('p'):
#         param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#         # print or use the parameters
#         print(param)
#         # param.extrinsic is the camera pose you are looking for
#         print("Camera pose:")
#         print(param.extrinsic)
#         o3d.io.write_pinhole_camera_parameters(o3d_viz_cam_param_file, param)
#         print(f'Saved camera parameters to {o3d_viz_cam_param_file}')


# def func_show_pointcloud(dataset, meta, bboxes, save_dir, conf_thres, seq):
#     box3d, scores, pred_labels = bboxes

#     calib_rdr_to_lidar = dataset.get_calib_info(meta['path']['path_calib'])
#     pc_lidar = dataset.get_pc_lidar(meta['path']['ldr_pc_64'], calib_rdr_to_lidar)

#     lines = [ [0,1], [0,2], [1,3], [2,3], [0,4], [1,5],\
#                 [2,6], [3,7], [4,5], [4,6], [5,7], [6,7] ]
    
#     colors_label = [[1.0,0,0] for _ in range(len(lines))]
#     list_line_set_label = []
#     list_line_set_pred = []

#     list_obj_label = []
#     gt_objs = dataset.get_label_bboxes(meta['path']['path_label'], calib_rdr_to_lidar)

#     for label_obj in gt_objs:
#         cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
#         obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
#         list_obj_label.append(obj)

#     list_obj_pred = []
#     list_cls_pred = []
    
#     for box_id, bbox in enumerate(box3d):
#         if scores[box_id] < conf_thres:
#             continue
#         xc, yc, zc, xl, yl, zl, rot = bbox
#         obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
#         list_obj_pred.append(obj)
#         list_cls_pred.append('Sedan') # TODO: modify this

                
#     for label_obj in list_obj_label:
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors_label)
#         list_line_set_label.append(line_set)
    
#     for idx_pred, pred_obj in enumerate(list_obj_pred):
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         colors_pred = [[0.,1,0.] for _ in range(len(lines))]
#         line_set.colors = o3d.utility.Vector3dVector(colors_pred)
#         list_line_set_pred.append(line_set)
    
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
#     with open(o3d_viz_cam_param_file, 'r') as f:
#         json_file = json.load(f)
#         param = json_file['trajectory'][0]
#     # vis.create_window(window_name='o3d')
#     o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred, 
#                                       lookat=param['lookat'], up=param['up'], front=param['front'], zoom=param['zoom'],
#                                       window_name='o3d')


    # o3d_vis.clear_geometries()
    # for geom in [pcd] + list_line_set_label + list_line_set_pred:
    #     o3d_vis.add_geometry(geom)
    #     o3d_vis.get_view_control().set_front([ -0.90639256046206595, -0.028190540934627261, 0.42149474461828484 ])
    #     o3d_vis.get_view_control().set_lookat( [ 3.984188149813602, 0.26230989153228496, 10.563407826105824 ])
    #     o3d_vis.get_view_control().set_up( [ 0.4190981213585428, 0.065180980134815214, 0.90559825778454595 ])
    #     o3d_vis.get_view_control().set_zoom(0.080000000000000002)
    #     o3d_vis.update_renderer()

    # ctr = vis.get_view_control()
    # ctr.change_field_of_view(0.1)
    # ctr.set_lookat(np.array(param['lookat'], dtype=np.float64).reshape(3, 1))
    # ctr.set_up(np.array(param['up'], dtype=np.float64).reshape(3, 1))
    # ctr.set_zoom(param['zoom'])
    # ctr.set_front(np.array(param['front'], dtype=np.float64).reshape(3, 1))

    # ctr.set_lookat([0, 0, 0])  # Change to your desired lookat point
    # ctr.set_up([0, -1, 0])  # Change to your desired up vector
    # ctr.set_zoom(0.8)  # Change to your desired zoom level    
        
    # vis.capture_screen_image('/home/andy/Desktop/o3d_test.png')
    # vis.update_renderer()
    # vis.run()
    # vis.destroy_window()

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # for geom in [pcd] + list_line_set_label + list_line_set_pred:
    #     vis.add_geometry(geom)
    # if os.path.exists(o3d_viz_cam_param_file):
    #     with open(o3d_viz_cam_param_file, 'r') as f:
    #         json_file = json.load(f)
    #         param = json_file['trajectory'][0]
    #     ctr = vis.get_view_control()
    #     ctr.change_field_of_view(param['field_of_view'])
    #     ctr.set_lookat(param['lookat'])
    #     ctr.set_up(param['up'])
    #     ctr.set_zoom(10)
    #     ctr.set_front(param['front'])


    # o3d_vis.poll_events()
    # o3d_vis.update_renderer()
    # image = o3d_vis.capture_screen_float_buffer(do_render=True)

    # Convert to a numpy array and save as an image file
    # image_np = np.asarray(image)
    # o3d.io.write_image("/home/andy/Desktop/o3d_test.png", o3d.geometry.Image((image_np * 255).astype(np.uint8)))

    # vis.run()
    # vis.destroy_window()
