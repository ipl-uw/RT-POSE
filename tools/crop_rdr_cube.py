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
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Cropping ROI of Kradar radar")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    args = parser.parse_args()
    return args
    


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    
    cfg.data.train.cfg.DATASET.PATH_SPLIT.train = 'configs/kradar/resources/split/train.txt'
    cfg.data.train.pipeline = []
    cfg.data.test.pipeline = []
    cfg.data.test.cfg.DATASET = cfg.DATASET
    
    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info(f"Start cropping")
    dataset = build_dataset(cfg.data.train)
    for dict_item in tqdm(dataset):
        rdr_cube_dir_parts = Path(dict_item['meta']['path']['rdr_cube']).parts
        new_rdr_cube_root_dir = os.path.join(*rdr_cube_dir_parts[:-2], 'radar_zyx_cube_npy_viz_roi')
        rdr_cube = dict_item['rdr_cube']
        if not os.path.exists(new_rdr_cube_root_dir):
            os.makedirs(new_rdr_cube_root_dir)
        np.save(os.path.join(new_rdr_cube_root_dir, rdr_cube_dir_parts[-1]), rdr_cube, allow_pickle=False)
    

if __name__ == '__main__':
    main()