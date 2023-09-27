from tqdm import tqdm
import pickle
import os
import numpy as np
from collections import defaultdict

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


if __name__ == '__main__':
    conf_thres = 0.5
    seq_to_save = ["7", "10", "11", "21", "54", "25", "24", "23", "19"]
    save_root = '/home/andy/Desktop/ee549'
    pred_file = 'work_dirs_tmp/HRTiny_feat16_small_clean_final_concat_train_pred/epoch_40/train_prediction.pkl'
    pred = load_pred(pred_file)
    for seq, pred_seq in sorted(pred.items()):
        if seq in seq_to_save:
            print(f'Now preparing sequence {seq}')
            last_frame = -10000
            preds_seg = []
            pred_seg_idx = 0
            preds = []
            for k, pred_frame in tqdm(sorted(pred_seq.items(), key=lambda x: x[0])):
                path_label_parts = pred_frame['metadata']['path']['path_label'].split('/')
                seq = path_label_parts[-3]
                frame = path_label_parts[-1].split('.')[0].split('_')[0]
                if int(frame) - last_frame > 1 and last_frame > 0:
                    preds_seg.append({str(pred_seg_idx): preds})
                    preds = []
                    pred_seg_idx += 1
                box3d = pred_frame['box3d']
                scores = pred_frame['scores']
                pred_labels = pred_frame['label_preds']
                for i in range(len(box3d)):
                    if scores[i] > conf_thres:
                        x, y, z, xl, yl, zl, theta = [round(x, 2) for x in box3d[i].tolist()] 
                        score = round(scores[i].item(), 2)
                        preds.append(f'{frame} Sedan {x} {y} {z} {xl} {yl} {zl} {theta} {score}\n')
                last_frame = int(frame)

            if len(preds) > 0:
                preds_seg.append({str(pred_seg_idx): preds})


            for pred_seg in preds_seg:
                pred_seg_idx, preds = list(pred_seg.items())[0]
                with open(os.path.join(save_root, f'{seq}_{pred_seg_idx}_pred.txt'), 'w') as outfile:
                    outfile.writelines(preds)
                
