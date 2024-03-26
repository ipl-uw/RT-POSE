import json 
from collections import defaultdict

occ_scenes = ["2023_1014_1356", "2023_1014_1358", "2023_1014_1359"]

with open('work_dirs/rf3d3/20240301_211544/epoch_45/epoch_45_seq_results_test.json') as f:
    results = json.load(f)

metrics = ['MPJPE', 'ABS_MPJPE', 'ABS_PJPE_0']
avg_metrics = defaultdict(dict)
normal_scene_count, occ_scene_count = 0, 0
for seq, seq_result in results.items():
    avg_key = ""
    if seq in occ_scenes:
        occ_scene_count += 1
        avg_key = "occ"

    else:
        normal_scene_count += 1
        avg_key = "normal"
    for metric in metrics:
        if metric not in avg_metrics[avg_key]:
            avg_metrics[avg_key][metric] = seq_result[metric]
        else:
            avg_metrics[avg_key][metric] += seq_result[metric]
    
for key in avg_metrics:
    for metric in metrics:
        avg_metrics[key][metric] /= normal_scene_count if key == "normal" else occ_scene_count

print(dict(avg_metrics))