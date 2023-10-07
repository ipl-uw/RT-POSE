from ..registry import DETECTORS
from .pose_net import PoseNet
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 
from .. import builder
from torch import nn

@DETECTORS.register_module
class RadarPoseNet(PoseNet):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        pose_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(RadarPoseNet, self).__init__(
            reader, backbone, neck, pose_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data['rdr_tensor'])
        x = self.backbone(
                input_features
            )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        x = self.extract_feat(example)
        # TODO: add radar lidar feature cube alignment
        preds, shared_conv_feat = self.pose_head(x)
        if return_loss:
            loss_dict = self.pose_head.loss(example, preds, self.test_cfg)
            return loss_dict
        
        else:
            return self.pose_head.predict(example, preds, self.test_cfg)
