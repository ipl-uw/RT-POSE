import numpy as np
import os
import json

if __name__ == '__main__':
    data_root = '/mnt/nas_cruw_pose'
    processed_name = 'radar/npy_DZYX_complex'
    seqs = sorted(list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root))))
    max = []
    min = []
    result ={}

    for seq_name in seqs:
        print('Start analyzing seq: ', seq_name)
        seq_dir = os.path.join(data_root, seq_name)
        processed_name = os.path.join(seq_dir, processed_name)
        seq_max = []
        seq_min = []
        for file in os.listdir(processed_name):
            rad_tensor = np.abs(np.load(os.path.join(processed_name, file)))
            seq_max.append(np.max(rad_tensor))
            seq_min.append(np.min(rad_tensor))
        max.append(np.max(seq_max))
        min.append(np.min(seq_min))
        result[seq_name] = {'max': np.max(seq_max).item(), 'min': np.min(seq_min).item()}

    print('max: ', np.max(max))
    print('min: ', np.min(min))    
    result['all'] = {'max': np.max(max).item(), 'min': np.min(min).item()}
    print(result)
    with open('min_max_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
