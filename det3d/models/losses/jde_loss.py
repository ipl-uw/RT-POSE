import torch
import torch.nn as nn
import torch.nn.functional as F
from det3d.core.utils.center_utils import _transpose_and_gather_feat
from pytorch_metric_learning import losses, miners, reducers, distances

class JDELoss(nn.Module):
    '''JDE loss for an output tensor
    Arguments:
        output (batch x embedding_dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        obj_id (batch x max_objects)
    '''
    def __init__(self, loss_fcn_cfg={}, reducer_cfg={}, miner_cfg={}, distance_cfg={}, **kwargs):
        super(JDELoss, self).__init__()
        # try:
        distance = getattr(distances, distance_cfg.pop('type'))(**distance_cfg)
        reducer = getattr(reducers, reducer_cfg.pop('type'))(**reducer_cfg)
        self.loss = getattr(losses, loss_fcn_cfg.pop('type'))(reducer=reducer, distance=distance, **loss_fcn_cfg)
        self.miner = getattr(miners, miner_cfg.pop('type'))(**miner_cfg)
        # except:
        #     raise ValueError("Invalid loss or miner type")
        
    def loss_per_batch(self, appearance_per_batch, mask_per_batch, obj_id_per_batch):
        embeddings, obj_ids = [], []
        for i in range(appearance_per_batch.shape[0]):
            embeddings.append(appearance_per_batch[i][mask_per_batch[i]>0])
            obj_ids.append(obj_id_per_batch[i][mask_per_batch[i]>0])
        embeddings = torch.cat(embeddings, dim=0)
        obj_ids = torch.cat(obj_ids, dim=0)
        if embeddings.shape[0] == 0:
            return torch.tensor(0., dtype=torch.float32).to(device=embeddings.device), 0, 0
        indices_tuple = self.miner(embeddings, obj_ids)
        loss_per_batch = self.loss(embeddings, obj_ids, indices_tuple)

        return loss_per_batch, self.miner.num_pos_pairs, self.miner.num_neg_pairs


    def forward(self, appearance_embedding, mask, ind, obj_id):
        appearance_embedding = _transpose_and_gather_feat(appearance_embedding, ind) # batch*2 x max_objects x embedding_dim
        batch_size = appearance_embedding.shape[0]//2
        # reshape to batch x 2 x max_objects x embedding_dim
        appearance_embedding = appearance_embedding.view(batch_size, 2, appearance_embedding.shape[1], appearance_embedding.shape[2])
        mask = mask.view(batch_size, 2, mask.shape[1])
        obj_id = obj_id.view(batch_size, 2, obj_id.shape[1])
        loss, num_pos, num_neg = [], 0, 0 # num_items is the number of triplets or pairs
        for i in range(batch_size):
            loss_per_batch, num_positive_per_batch, num_negative_per_batch = self.loss_per_batch(appearance_embedding[i], mask[i], obj_id[i])
            num_pos += num_positive_per_batch
            num_neg += num_negative_per_batch
            loss.append(loss_per_batch)

        return loss, num_pos, num_neg