import json 


if __name__ == '__main__':
    mpjpe_threshold = [0.1, 0.08]


    epoch_dir = 'work_dirs/hr3d_one_hm/20231013_000843/epoch_100'
    epoch_name = epoch_dir.split('/')[-1]
    with open(f'{epoch_dir}/{epoch_name}_seq_results_test.json', 'r') as f:
        test_seq_results = json.load(f)
    with open(f'{epoch_dir}/{epoch_name}_seq_results_train.json', 'r') as f:
        train_seq_results = json.load(f)


    unsatisfied_seq = {}
    for i in mpjpe_threshold:
        unsatisfied_seq[i] = []
    relative_error = {}
    for seq, result in test_seq_results.items():
        seq_relative_error = {}
        for metric, value in result.items():
            if metric == 'MPJPE':
                for i in mpjpe_threshold:
                    if value > i:
                        unsatisfied_seq[i].append(seq)
            seq_relative_error[metric] = value / train_seq_results[seq][metric]
        relative_error[seq] = seq_relative_error
    with open(f'{epoch_dir}/{epoch_name}_seq_results_relative_error.json', 'w') as f:
        json.dump(relative_error, f, indent=2)


    with open(f'{epoch_dir}/{epoch_name}_unsatisfied_seq.json', 'w') as f:
        json.dump(unsatisfied_seq, f, indent=2)

