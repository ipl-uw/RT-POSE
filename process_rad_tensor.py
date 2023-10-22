import numpy as np
import os
import torch

# donwsample doppler axis
def rad_preprocess(rad_tensor_path, to_abs=True, filter=None, max=100000000, min=1., as_fp16=True): 
    # try:
    rad_tensor = np.load(rad_tensor_path)
    # except:
        # return 'read_err'
    if to_abs:
        rad_tensor = np.abs(rad_tensor)
    rad_tensor = torch.tensor(np.transpose(rad_tensor, (1, 2, 3, 0)))
    outout_shape = list(rad_tensor.shape)
    outout_shape[-1] = outout_shape[-1] // 2
    rad_tensor = rad_tensor.view(-1, rad_tensor.shape[-1]).unsqueeze(1) # make it 1 x (E x A x R) x D
    new_rad_tensor = filter(rad_tensor).squeeze()
    # try:
    new_rad_tensor = new_rad_tensor.view(outout_shape).permute(3, 0, 1, 2).contiguous().numpy()
    # except:
        # return 'reshape err'
    new_rad_tensor = (new_rad_tensor - min) / (max - min)
    new_rad_tensor[new_rad_tensor<0.] = 0.
    if as_fp16:
        new_rad_tensor = new_rad_tensor.astype(np.float16)
    return new_rad_tensor

# conv_kernel = torch.nn.Conv1d(1, 1, 2, stride=2, bias=False)
# conv_kernel.weight.data = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)
# conv_kernel.weight.requires_grad = False

if __name__ == '__main__':
    data_root = '/mnt/nas_cruw_pose'
    target_dir = '/mnt/ssd3/nas_cruw_pose'
    processd_name = 'radar/npy_DZYX_complex'
    stored_dir_name = 'DZYX_npy_f16'
    # seqs = sorted(list(filter(lambda file_name: file_name[:4] == '2023', os.listdir(data_root))))[-8:]
    seqs = ['2023_0730_1333']

    # create a pytorch 1-D max pooling filter to downsample doppler axis
    rad_filter = torch.nn.MaxPool1d(2, stride=2)

    read_err_list, reshape_err_list = [], []
    for seq_name in seqs:
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
            new_rad_tensor = rad_preprocess(file_path, filter=rad_filter)
            # if new_rad_tensor == 'read_err':
            #     print(f'read_err: {seq_name}/{file_name}')
            #     read_err_list.append(f'{seq_name}/{file_name}')
            #     continue
            # elif new_rad_tensor == 'reshape err':
            #     print(f'reshape_err: {seq_name}/{file_name}')
            #     reshape_err_list.append(f'{seq_name}/{file_name}')
            #     continue
            # else:
            np.save(target_file_dir, new_rad_tensor)
    # with open('./read_err.txt', 'w') as f:
    #     for line in read_err_list:
    #         f.write(line + '\n')
    # with open('./reshape_err.txt', 'w') as f:
    #     for line in reshape_err_list:
    #         f.write(line + '\n')