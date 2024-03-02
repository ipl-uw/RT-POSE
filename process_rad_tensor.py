import numpy as np
import os
import torch
import torch.nn.functional as F
import concurrent.futures


# donwsample doppler axis
def rad_preprocess(rad_tensor_path, to_abs=True, max=100000000, min=1., max_complex=1000000000, as_fp16=True, return_type='real'): 
    rad_tensor = np.load(rad_tensor_path)
    if return_type == 'real':
        if to_abs:
            rad_tensor = np.abs(rad_tensor)
        rad_tensor = torch.tensor(np.transpose(rad_tensor, (1, 2, 3, 0)))
        outout_shape = list(rad_tensor.shape)
        outout_shape[-1] = outout_shape[-1] // 2
        rad_tensor = rad_tensor.view(-1, rad_tensor.shape[-1]).unsqueeze(1) # make it 1 x (E x A x R) x D
        new_rad_tensor = F.max_pool1d(rad_tensor, 2, stride=2)
        new_rad_tensor = new_rad_tensor.squeeze()
        new_rad_tensor = new_rad_tensor.view(outout_shape).permute(3, 0, 1, 2).contiguous().numpy()
        new_rad_tensor = (new_rad_tensor - min) / (max - min)
        new_rad_tensor[new_rad_tensor<0.] = 0.
        if as_fp16:
            new_rad_tensor = new_rad_tensor.astype(np.float16)
        return new_rad_tensor
    elif return_type == 'complex':
        rad_tensor_abs = np.abs(rad_tensor)
        rad_tensor = torch.tensor(np.transpose(rad_tensor, (1, 2, 3, 0)))
        rad_tensor_abs = torch.tensor(np.transpose(rad_tensor_abs, (1, 2, 3, 0)))
        outout_shape = list(rad_tensor.shape)
        outout_shape[-1] = outout_shape[-1] // 2
        rad_tensor = rad_tensor.view(-1, rad_tensor.shape[-1]).unsqueeze(1) # make it 1 x (E x A x R) x D
        rad_tensor_abs = rad_tensor_abs.view(-1, rad_tensor_abs.shape[-1]).unsqueeze(1) # make it 1 x (E x A x R) x D
        _, indices = F.max_pool1d(rad_tensor_abs, 2, stride=2, return_indices=True)
        new_rad_tensor = torch.gather(rad_tensor, 2, indices).squeeze()
        new_rad_tensor = new_rad_tensor.view(outout_shape).permute(3, 0, 1, 2).contiguous().numpy()
        new_rad_tensor = np.stack([new_rad_tensor.real, new_rad_tensor.imag])
        new_rad_tensor = new_rad_tensor / max_complex
        if as_fp16:
            new_rad_tensor = new_rad_tensor.astype(np.float16)
        return new_rad_tensor
    
    
def process_seq(seq_name):
    # create a pytorch 1-D max pooling filter to downsample doppler axis
    print('Start processing seq: ', seq_name)
    target_seq_dir = os.path.join(target_dir, seq_name, stored_dir_name)
    os.makedirs(target_seq_dir, exist_ok=True)
    seq_dir = os.path.join(data_root, seq_name)
    processd_dir = os.path.join(seq_dir, processd_name)
    for file_name in sorted(os.listdir(processd_dir)):
        target_file_dir = os.path.join(target_seq_dir, file_name)
        if os.path.exists(target_file_dir):
            continue
        file_path = os.path.join(processd_dir, file_name)
        new_rad_tensor = rad_preprocess(file_path, return_type='real')
        np.save(target_file_dir, new_rad_tensor)



# conv_kernel = torch.nn.Conv1d(1, 1, 2, stride=2, bias=False)
# conv_kernel.weight.data = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
# conv_kernel.weight.requires_grad = False

if __name__ == '__main__':
    data_root = '/mnt/nas_cruw_pose'
    target_dir = '/mnt/ssd3/cruw_pose'
    processd_name = 'radar/npy_DZYX_complex'
    stored_dir_name = 'DZYX_npy_f16'
    # seqs = sorted(list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root))))[-8:]
    # seqs = ['2023_0724_1553', '2023_0725_1559', '2023_0725_1600', '2023_0725_1602', '2023_0725_1603', 
    #         '2023_0725_1604', '2023_0725_1714', '2023_0725_1716', '2023_0730_1240', '2023_0730_1242', 
    #         '2023_0730_1245', '2023_0730_1316', '2023_0730_1321', '2023_0730_1331', '2023_0730_1332', 
    #         '2023_0730_1333', '2023_0730_1343', '2023_0730_1346', '2023_0730_1347', '2023_0730_1349', 
    #         '2023_0730_1351', '2023_0730_1352', '2023_0730_1354', '2023_0730_1355', '2023_0730_1357', 
    #         '2023_0730_1359', '2023_0730_1401', '2023_0730_1402', '2023_0730_1404', '2023_0730_1406', 
    #         '2023_0730_1407', '2023_0730_1411', '2023_0730_1414', '2023_0730_1416', '2023_0730_1417', 
    #         '2023_0730_1418', '2023_0730_1425', '2023_0730_1426', '2023_0730_1432', '2023_0730_1433']
    # seqs = seqs[35:40]

    with open('targetseqs.txt', 'r') as f:
        seqs = f.readlines()
    # seqs = [line.strip() for line in seqs][:36]
    seqs = ['2023_0730_1359']
    print(seqs[0])
    print(seqs[-1])
    # seqs = [line.strip().split(',')[1] for line in seqs]


    # seqs = ['2023_1014_1359']

    multi_process = False
    
    if multi_process:
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(process_seq, seqs) 

    
    # create a pytorch 1-D max pooling filter to downsample doppler axis

    # read_err_list, reshape_err_list = [], []
    # for seq_name in seqs:
    #     print('Start processing seq: ', seq_name)
    #     target_seq_dir = os.path.join(target_dir, seq_name, stored_dir_name)
    #     os.makedirs(target_seq_dir, exist_ok=True)
    #     seq_dir = os.path.join(data_root, seq_name)
    #     processd_dir = os.path.join(seq_dir, processd_name)
    #     for file_name in sorted(os.listdir(processd_dir)):
    #         target_file_dir = os.path.join(target_seq_dir, file_name)
    #         if os.path.exists(target_file_dir):
    #             continue
    #         file_path = os.path.join(processd_dir, file_name)
    #         new_rad_tensor = rad_preprocess(file_path, return_type='complex')
    #         # if new_rad_tensor == 'read_err':
    #         #     print(f'read_err: {seq_name}/{file_name}')
    #         #     read_err_list.append(f'{seq_name}/{file_name}')
    #         #     continue
    #         # elif new_rad_tensor == 'reshape err':
    #         #     print(f'reshape_err: {seq_name}/{file_name}')
    #         #     reshape_err_list.append(f'{seq_name}/{file_name}')
    #         #     continue
    #         # else:
    #         np.save(target_file_dir, new_rad_tensor)
    # with open('./read_err.txt', 'w') as f:
    #     for line in read_err_list:
    #         f.write(line + '\n')
    # with open('./reshape_err.txt', 'w') as f:
    #     for line in reshape_err_list:
    #         f.write(line + '\n')
            
    else:
        for seq in seqs:
            process_seq(seq)