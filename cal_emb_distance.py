import torch
import torch.nn.functional as F
import numpy as np

# emb_batch1: NxD, emb_batch2: NxD
def L2_Norm(emb_batch1, emb_batch2, p=2):
    emb_batch1 = F.normalize(torch.from_numpy(emb_batch1), p=p, dim=1)
    emb_batch2 = F.normalize(torch.from_numpy(emb_batch2), p=p, dim=1)
    output = F.pairwise_distance(emb_batch1, emb_batch2, p)
    return output

if __name__ == '__main__':
    emb_batch1 = np.random.randn(3, 5) * 10
    emb_batch2 = np.random.randn(3, 5) * 10
    output = L2_Norm(emb_batch1, emb_batch2)
    print(output)