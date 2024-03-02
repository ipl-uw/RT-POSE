import json

out_in = [('/mnt/nas_cruw_pose_2/Train_outdoor.json', '/mnt/nas_cruw_pose_2/Test_outdoor.json', '/mnt/nas_cruw_pose_2/file_meta_outdoor.txt'), \
 ('/mnt/nas_cruw_pose/Train.json', '/mnt/nas_cruw_pose/Test.json', '/mnt/nas_cruw_pose/file_meta.txt')
 ]

with open('/home/andy/ipl/CRUW-POSE/targetseqs.txt', 'r') as f:
    target_seqs = f.readlines()
target_seqs = [line.strip() for line in target_seqs]

train_all, test_all = {}, {}
for train_file, test_file, file_meta in out_in:
    with open(train_file, 'r') as f:
        train = json.load(f)
    with open(test_file, 'r') as f:
        test = json.load(f)
    with open(file_meta, 'r') as f:
        lines = f.readlines()
    seq_id_to_name, seq_name_to_id = {}, {}
    for line in lines:
        seq_id, seq_name = line.strip().split(',')
        seq_id_to_name[seq_id] = seq_name
        seq_name_to_id[seq_name] = seq_id
    for target_seq in target_seqs:
        if target_seq in seq_name_to_id:
            train_all[target_seq] = train[seq_name_to_id[target_seq]]
            test_all[target_seq] = test[seq_name_to_id[target_seq]]

# sort train and test by keys
train_all = dict(sorted(train_all.items()))
test_all = dict(sorted(test_all.items()))



new_seqid_to_name = {}
new_name_to_seqid = {}
# export to new meta.txt
with open('/mnt/ssd3/file_meta.txt', 'w') as f:
    for i, target_seq in enumerate(train_all.keys()):
        f.write(f'{i},{target_seq}\n')
        new_seqid_to_name[i] = target_seq
        new_name_to_seqid[target_seq] = i
    

# rename the keys in train_all and test_all
train_all_new = {}  
test_all_new = {}
for i, target_seq in enumerate(train_all.keys()):
    train_all_new[str(i)] = train_all[target_seq]
    test_all_new[str(i)] = test_all[target_seq]

with open('/mnt/ssd3/Train.json', 'w') as f:
    json.dump(train_all_new, f, indent=2)

with open('/mnt/ssd3/Test.json', 'w') as f:
    json.dump(test_all_new, f, indent=2)




    
