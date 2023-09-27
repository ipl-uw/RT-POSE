import json
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="combine train test viz file")
    parser.add_argument("train_file", help="path for json visualization file with prediction on train set")
    parser.add_argument("test_file", help="path for json visualization file with prediction on test set")
    parser.add_argument("--viz", action='store_true')
    args = parser.parse_args()
    return args


def combine_file(train_viz_path, test_viz_path):
    with open(train_viz_path, 'r') as f:
        combine_pred = json.load(f)
    with open(test_viz_path, 'r') as f:
        test_pred = json.load(f)
    for seq, frames in test_pred.items():
        combine_pred[seq].update(frames)
    return combine_pred



if __name__ == '__main__':
    args = parse_args()
    if args.viz:
        combine_pred = combine_file(args.train_file, args.test_file)
        save_path = os.path.join(os.path.dirname(args.train_file), 'train_test_prediction_viz_format.json')
        with open(save_path, 'w') as f:
            json.dump(combine_pred, f, indent=2)
    else:
        with open(args.train_file, 'r') as f:
            combine_gt = json.load(f)
        with open(args.test_file, 'r') as f:
            test_gt = json.load(f)
        combine_gt.update(test_gt)
        save_path = os.path.join(os.path.dirname(args.train_file), 'refined_v3numpoints_all_Radar_roi1_Sedan_BusorTruck.json')
        with open(save_path, 'w') as f:
            json.dump(combine_gt, f, indent=2)