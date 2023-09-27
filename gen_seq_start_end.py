import json
from collections import defaultdict

if __name__ == '__main__':
    seq_start_end = {}
    with open('/mnt/ssd1/kradar_dataset/labels/refined_v3.json', 'r')  as f:
        label = json.load(f)
    for split_type, data in label.items():
        seq_start_end[split_type] = defaultdict(list)
        start_idx, end_idx= None, None
        for idx, seq_frame in enumerate(data):
            if idx == 0:
                start_idx = idx
                continue
            seq, frame = seq_frame['seq'], int(seq_frame['frame'])
            last_seq, last_frame = data[idx-1]['seq'], int(data[idx-1]['frame'])
            if last_seq != seq or frame != last_frame + 1:
                end_idx = idx - 1
                seq_start_end[split_type][data[idx-1]['seq']].append([start_idx, end_idx])
                start_idx = idx
            
        end_idx = len(data) - 1
        seq_start_end[split_type][data[-1]['seq']].append([start_idx, end_idx])
    with open('/mnt/ssd1/kradar_dataset/labels/seq_start_end.json', 'w') as f:
        json.dump(seq_start_end, f, indent=2)