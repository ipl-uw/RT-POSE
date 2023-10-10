# ------------------------------------------------------------------------------
# Portions of this code are from
# det3d (https://github.com/poodarchu/Det3D/tree/56402d4761a5b73acd23080f537599b0888cce07)
# Human Pose Estimation modified from CenterNet Head
# ------------------------------------------------------------------------------

import logging
from collections import defaultdict
from det3d.core import box_torch_ops
import torch
from det3d.torchie.cnn import kaiming_init
from torch import double, nn
from det3d.models.losses.centernet_loss import FastFocalLoss, RegLoss
from det3d.models.utils import Sequential
from ..registry import HEADS
import copy 
try:
    from det3d.ops.dcn import DeformConv
except:
    print("Deformable Convolution not built!")

from det3d.core.utils.circle_nms_jit import circle_nms

class FeatureAdaption(nn.Module):
    """Feature Adaption Module.

    Feature Adaption Module is implemented based on DCN v1.
    It uses anchor shape prediction rather than feature map to
    predict offsets of deformable conv layer.

    Args:
        in_channels (int): Number of channels in the input feature map.
        out_channels (int): Number of channels in the output feature map.
        kernel_size (int): Deformable conv kernel size.
        deformable_groups (int): Deformable conv group size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAdaption, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(
            in_channels, deformable_groups * offset_channels, 1, bias=True)
        self.conv_adaption = DeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()

    def forward(self, x,):
        offset = self.conv_offset(x)
        x = self.relu(self.conv_adaption(x, offset))
        return x


# Modified to GCR by Andy Cheng
class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        norm=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads 
        for head, (classes, num_conv) in self.heads.items():
            in_channels_head = in_channels
            fc = Sequential()
            for i in range(num_conv-1):
                if norm:
                    fc.add(nn.GroupNorm(num_groups=8, num_channels=in_channels_head))
                fc.add(nn.Conv3d(in_channels_head, head_conv,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))
                fc.add(nn.ReLU())
                in_channels_head = head_conv
            fc.add(nn.Conv3d(head_conv, classes,
                    kernel_size=final_kernel, stride=1, 
                    padding=final_kernel // 2, bias=True))    
            if 'hm' in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv3d):
                        kaiming_init(m)

            self.__setattr__(head, fc)
        

    def forward(self, x):
        ret_dict = dict()        
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict

class DCNSepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_cls,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(DCNSepHead, self).__init__(**kwargs)

        # feature adaptation with dcn
        # use separate features for classification / regression
        self.feature_adapt_cls = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4) 
        
        self.feature_adapt_reg = FeatureAdaption(
            in_channels,
            in_channels,
            kernel_size=3,
            deformable_groups=4)  

        # heatmap prediction head 
        self.cls_head = Sequential(
            nn.Conv2d(in_channels, head_conv,
            kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_cls,
                kernel_size=3, stride=1, 
                padding=1, bias=True)
        )
        self.cls_head[-1].bias.data.fill_(init_bias)

        # other regression target 
        self.task_head = SepHead(in_channels, heads, head_conv=head_conv, bn=bn, final_kernel=final_kernel)


    def forward(self, x):    
        center_feat = self.feature_adapt_cls(x)
        reg_feat = self.feature_adapt_reg(x)

        cls_score = self.cls_head(center_feat)
        ret = self.task_head(reg_feat)
        ret['hm'] = cls_score

        return ret


@HEADS.register_module
class CenterHead(nn.Module):
    def __init__(
        self,
        in_channels=128,
        tasks=[],
        dataset='cruw_pose',
        common_heads=dict(),
        logger=None,
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        weight=0.1,
        code_weights=[],
        dcn_head=False,
    ):
        super(CenterHead, self).__init__()
        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.weight = weight  # weight between hm loss and loc loss
        self.code_weights = code_weights
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()

        if not logger:
            logger = logging.getLogger("CenterHead")
        self.logger = logger

        logger.info(
            f"num_classes: {num_classes}"
        )
        # a shared convolution to to get the desired channel numbers : TODO: check if this is necessary 
        if not (in_channels == share_conv_channel):
            self.shared_conv = nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.Conv3d(in_channels, share_conv_channel,
                kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
        else:
            self.shared_conv = nn.Identity()
        self.tasks = nn.ModuleList()
        print("Use HM Bias: ", init_bias)

        if dcn_head:
            print("Use Deformable Convolution in the CenterHead!")

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            if not dcn_head:
                heads.update(dict(hm=(num_cls, num_hm_conv)))
                self.tasks.append(
                    SepHead(share_conv_channel, heads, head_conv=32, init_bias=init_bias, final_kernel=3)
                )
            else:
                self.tasks.append(
                    DCNSepHead(share_conv_channel, num_cls, heads, bn=True, init_bias=init_bias, final_kernel=3)
                )

        logger.info("Finish CenterHead Initialization")

    def forward(self, x, *kwargs):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts, x

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y

    def loss(self, example, preds_dicts, test_cfg, **kwargs):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict['hm'] = self._sigmoid(preds_dict['hm'])

            hm_loss = self.crit(preds_dict['hm'], example['hm'][task_id], example['ind'][task_id], example['mask'][task_id], example['cat'][task_id])

            ret = {}
            target_pose = example['anno_pose'][task_id]
            # Regression loss for 15 key points' x y z offset        
            reg_loss = self.crit_reg(preds_dict['reg'], example['mask'][task_id], example['ind'][task_id], target_pose)
            loc_loss = (reg_loss*reg_loss.new_tensor(self.code_weights)).sum()

            loss = hm_loss + self.weight*loc_loss

            ret.update({'loss': loss, 'hm_loss': hm_loss.detach().cpu(), 'loc_loss':loc_loss, 'loc_loss_elem': reg_loss.detach().cpu(), 'num_positive': example['mask'][task_id].float().sum()})
            rets.append(ret)
        
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets: 
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    @torch.no_grad()
    def predict(self, example, preds_dicts, test_cfg, app_emb_tasks=None, **kwargs):
        """decode, nms, then return the detection result. Additionaly support double flip testing 
        """
        # get loss info
        rets = []
        metas = []

        double_flip = test_cfg.get('double_flip', False)

        post_center_range = test_cfg.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=preds_dicts[0]['hm'].dtype,
                device=preds_dicts[0]['hm'].device,
            )

        for task_id, preds_dict in enumerate(preds_dicts):
            # convert N C H W L to N H W L C 
            for key, val in preds_dict.items():
                preds_dict[key] = val.permute(0, 2, 3, 4, 1).contiguous()
    
            batch_size = preds_dict['hm'].shape[0]

            if double_flip:
                assert batch_size % 4 == 0, print(batch_size)
                batch_size = int(batch_size / 4)
                for k in preds_dict.keys():
                    # transform the prediction map back to their original coordinate befor flipping
                    # the flipped predictions are ordered in a group of 4. The first one is the original pointcloud
                    # the second one is X flip pointcloud(y=-y), the third one is Y flip pointcloud(x=-x), and the last one is 
                    # X and Y flip pointcloud(x=-x, y=-y).
                    # Also please note that pytorch's flip function is defined on higher dimensional space, so dims=[2] means that
                    # it is flipping along the axis with H length(which is normaly the Y axis), however in our traditional word, it is flipping along
                    # the X axis. The below flip follows pytorch's definition yflip(y=-y) xflip(x=-x)
                    _, H, W, C = preds_dict[k].shape
                    preds_dict[k] = preds_dict[k].reshape(int(batch_size), 4, H, W, C)
                    preds_dict[k][:, 1] = torch.flip(preds_dict[k][:, 1], dims=[1]) 
                    preds_dict[k][:, 2] = torch.flip(preds_dict[k][:, 2], dims=[2])
                    preds_dict[k][:, 3] = torch.flip(preds_dict[k][:, 3], dims=[1, 2])

            meta_list = example['meta']

                    


            batch_hm = torch.sigmoid(preds_dict['hm'])

            batch_reg = preds_dict['reg']


            batch, H, W, L, num_cls = batch_hm.size()

            batch_reg = batch_reg.reshape(batch, H*W*L, 3)
            batch_hm = batch_hm.reshape(batch, H*W*L, num_cls)

            zs, ys, xs = torch.meshgrid([torch.arange(0, H), torch.arange(0, W), torch.arange(0, L)])
            zs = zs.view(1, H, W, L).repeat(batch, 1, 1, 1).to(batch_hm)
            ys = ys.view(1, H, W, L).repeat(batch, 1, 1, 1).to(batch_hm)
            xs = xs.view(1, H, W, L).repeat(batch, 1, 1, 1).to(batch_hm)

            xs = xs.view(batch, -1, 1) + batch_reg[:, :, 0:1]
            ys = ys.view(batch, -1, 1) + batch_reg[:, :, 1:2]
            zs = zs.view(batch, -1, 1) + batch_reg[:, :, 2:3]

            xs = xs * test_cfg.out_size_factor[2] * test_cfg.voxel_size[0] + test_cfg.pc_range[0]
            ys = ys * test_cfg.out_size_factor[1] * test_cfg.voxel_size[1] + test_cfg.pc_range[1]
            zs = zs * test_cfg.out_size_factor[0] * test_cfg.voxel_size[2] + test_cfg.pc_range[2]

            batch_app_emb = None
            if app_emb_tasks is not None:
                batch_app_emb = app_emb_tasks[task_id].permute(0, 2, 3, 1).contiguous().reshape(batch, H*W, -1)

            batch_pts_preds = torch.cat([xs, ys, zs], dim=2)

            metas.append(meta_list)

            if test_cfg.get('per_class_nms', False):
                raise NotImplementedError()
            else:
                rets.append(self.post_processing(batch_pts_preds, batch_hm, test_cfg, post_center_range, task_id, batch_app_emb)) 

        # Merge tasks results
        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            merged_task_key_points = []
            for k in range(len(rets)):
                merged_task_key_points += rets[k][i]
            ret['keypoints'] = merged_task_key_points
            ret['metadata'] = metas[0][i]
            ret_list.append(ret)
        return ret_list 

    @torch.no_grad()
    def post_processing(self, batch_pts_preds, batch_hm, test_cfg, post_center_range, task_id, batch_app_emb=None):
        batch_size = len(batch_hm)
        prediction_pose_batches = []
        for i in range(batch_size):
            pts_preds = batch_pts_preds[i]
            hm_preds = batch_hm[i]
            predicted_key_points = [] # (keypoint_id, x, y, z, score)
            for i in range(hm_preds.shape[-1]):
                label = i
                ind = torch.argmax(hm_preds[:, i])
                score = hm_preds[:, i][ind]
                if score > test_cfg.score_threshold:
                    predicted_key_points.append((label, *pts_preds[ind].cpu().tolist(), score.cpu().item()))
            # TODO: add more post-processing
            prediction_pose_batches.append(predicted_key_points)

        return prediction_pose_batches 

import numpy as np 
def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep  