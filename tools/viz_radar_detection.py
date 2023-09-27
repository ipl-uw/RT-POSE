from det3d.datasets import build_dataset
import argparse
import os
import pickle
from det3d.torchie import Config
from det3d.torchie.apis import (
    get_root_logger,
)
from tqdm import tqdm
from utils.viz_kradar_funcs import *
from pathlib import Path
import torch
from collections import defaultdict

# scenario_frame_dict = {
#     '7': '00182_00149',
# 	'21': '00506_00502',
# 	'22': '00492_00490',
# 	'26': '00252_00247',
# 	'42': '00294_00285',
# 	'44': '00575_00566',
# 	'53': '00479_00475',
# 	'57': '00294_00289',
#     '34': '00171_00168',
# }


seq_to_viz = ["7", "54", "49", "41", "40" ] # '11', "25", "24", "23", "21", "20"

# scenario_frame_dict = {
#     '11': '00337_00304',
#     '53': '00479_00475',
# 	'57': '00294_00289',
# 	'42': '00294_00285',
#     '34': '00171_00168',
#     # '12': '00465_00429',
#     '22': '00538_00536',
# }

# scenario_frame_dict = {
#     '3': '00180_00150',
#     '12': '00310_00274',
# 	'27': '00302_00298',
# 	'28': '00242_00239',
# 	'48': '00192_00239',
#     '2': '00195_00165',
#     '12': '00300_00264',
#     '12': '00316_00280',
#     '12': '00817_00781',

# }

scenario_frame_dict = {
    '3': '00180_00150',
    '12': '00310_00274',
	'27': '00302_00298',
	'28': '00242_00239',
	'48': '00192_00239',
    '2': '00195_00165',
    '12': '00300_00264',
    '12': '00316_00280',
    '12': '00817_00781',

}




def parse_args():
    parser = argparse.ArgumentParser(description="Kradar detection visualization")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--pred_path", help="prediction pickle file")
    parser.add_argument("--checkpoint_name", help="checkpoint file generating the prediction")
    parser.add_argument("--viz_conf_thres", help="visualization confidence threshold")
    args = parser.parse_args()
    return args

# group prediction with sequence
def load_pred(pred_file):
    with open (pred_file, 'rb') as f:
        pred = pickle.load(f)
    if len(pred.items()) < 59:
        return pred
    print('Transforming prediction to dict indexed by sequence...')
    new_pred = defaultdict(dict)
    for seq_frame, v in tqdm(pred.items()):
        seq, frame = seq_frame.split('/')
        new_pred[seq].update({frame: v})
    with open(pred_file, "wb") as f:
        pickle.dump(new_pred, f)
    return new_pred


def viz_frame(dataset, pred_frame, save_dir_name, conf_thres, seq):
    if isinstance(pred_frame['box3d'], torch.Tensor):
        pred_frame['box3d'] = pred_frame['box3d'].numpy() 
        pred_frame['scores'] = pred_frame['scores'].numpy() 
        pred_frame['label_preds'] = pred_frame['label_preds'].numpy() 
    func_show_radar_cube_bev(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
        pred_frame['scores'], pred_frame['label_preds']), save_dir_name, conf_thres, seq)
    # func_show_pointcloud(dataset, pred_frame['metadata'], (pred_frame['box3d'], \
    # pred_frame['scores'], pred_frame['label_preds']), save_dir_name, conf_thres, seq)

'''
    save_dir_name: name of the model used for inference
'''

def viz_seq(kradar_dataset, pred_seq, checkpoint_name, conf_thres):
    pred_frame_tmp = list(pred_seq.values())[0]
    seq_root_seg = Path(pred_frame_tmp['metadata']['path']['path_calib']).parts[:-2]
    save_dir = os.path.join(*seq_root_seg, 'inference_viz', f'{checkpoint_name}_{conf_thres}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get_label_bboxes()
    conf_thres = float(conf_thres)
    for k, pred_frame in tqdm(sorted(pred_seq.items(), key=lambda x: x[0])):
        path_label_parts = pred_frame['metadata']['path']['path_label'].split('/')
        seq = path_label_parts[-3]
        frame = path_label_parts[-1].split('.')[0]
        # if (seq, frame) in scenario_frame_dict:
        if seq in seq_to_viz:
        # if seq == '11':
            # if draw_idx == 0:
            #     o3d_vis.run()
            viz_frame(kradar_dataset, pred_frame, save_dir, conf_thres, seq)
    # o3d_vis.destroy_window()
    

# Currently support single dataset visualization
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"Start visualization")
    dataset = build_dataset(cfg.data.test)
    pred = load_pred(args.pred_path)
    for seq, pred_seq in sorted(pred.items()):
        # if seq in ['7', '8', '10']:
        logger.info(f'Now visualizing sequence {seq}')
        viz_seq(dataset, pred_seq, args.checkpoint_name, args.viz_conf_thres)



def load_tracking_file(path):
    with open(path, 'r') as f:
        tracklet_file = f.readlines()
    tracklets = [x.strip() for x in tracklet_file]
    frame_to_dets = defaultdict(list)
    for tracklet in tracklets:
        frame_id, tracking_id, x, y, xl, yl, rotation, score, _, _ ,_ = tracklet.split(',')
        frame_to_dets[frame_id].append((tracking_id, x, y, xl, yl, rotation, score))

    return frame_to_dets



seq_to_offset = {
    '7': 33,
    '10': 26,
    '11': 33,
    '21': 4,
    '23': 2,
    '24': 3,
    '25': -1,
    '19': 37,
    '54': 6
}

def viz_tracking():
    root_dir = '/home/andy/Desktop/ee549 pred file kradar/SCT_with_rotation/SCT'
    viz_root = '/home/andy/Desktop/ee549 pred file kradar/SCT_with_rotation'
    detection_path = 'work_dirs_tmp/HRTiny_feat16_small_clean_final_concat_train_pred/epoch_40/train_prediction.pkl'
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # manually modify rdr cube ROI for viz
    cfg.data.test.cfg.DATASET.RDR_CUBE.ROI = { # each bin is 0.4 [m] (minimum range resolution)
      'z': [-2, 6.],     # Dim: [m] / [-2.0, 5.6] for Driving corridor
      'y': [-20., 20.], # Dim: [m] / [-6.4, 6.0] for Driving corridor
      'x': [0., 80.],     # Dim: [m] / [0.0, 71.6] for Driving corridssor
      }


    logger = get_root_logger(cfg.log_level)
    dataset = build_dataset(cfg.data.test)
    pure_detection = load_pred(detection_path)
    for seq_file in tqdm(os.listdir(root_dir)):
        frame_to_dets = load_tracking_file(os.path.join(root_dir, seq_file))
        viz_dir = seq_file.split('_')
        seq = viz_dir[0]
        viz_dir = f'{viz_dir[0]}_{viz_dir[1]}'
        os.makedirs(os.path.join(viz_root, viz_dir), exist_ok=True)
        viz_dir = os.path.join(viz_root, viz_dir)
        id_color, used_colors = {}, []
        logger.info(f"Start visualization {seq_file.split('.')[0]}")
        for frame, tracklet_frame in frame_to_dets.items():
            frame_id = "{:05d}_".format(int(frame)) + "{:05d}".format(int(frame) - seq_to_offset[seq])
            metadata = pure_detection[seq][frame_id]['metadata']
            id_color, used_colors = func_show_radar_cube_bev_tracking(dataset, frame_id, metadata, tracklet_frame, viz_dir, id_color, used_colors)
        


if __name__ == '__main__':
    # scenario_frame_dict = [(k, v) for k, v in scenario_frame_dict.items()]
    # scenario_frame_dict = [('12', '00310_00274'), ('12', '00300_00264'), ('12', '00316_00280'), ('12', '00817_00781')]
    # scenario_frame_dict = [[(k, v) for k, v in scenario_frame_dict.items()][0]]
    main()
    # viz_tracking()
