from ..registry import DETECTORS
from .pose_net import PoseNet
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelNet(PoseNet):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        pose_head,
        sensor_type,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(VoxelNet, self).__init__(
            reader, backbone, sensor_type, neck, pose_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
            input_features = self.reader(data["features"], data['num_voxels'])

        x = self.backbone(
                input_features, data["coors"], data["batch_size"], data["input_shape"]
            )

        if self.with_neck:
            x = self.neck(x)

        return x

    def forward(self, example, return_loss=True, **kwargs):
        example_sensor = {}
        example_sensor.update(example[self.sensor_type])
        example_sensor.update({'meta': example['meta']})
        x = self.extract_feat(example_sensor)
        preds, _ = self.pose_head(x)

        if return_loss:
            return self.pose_head.loss(example_sensor, preds, self.test_cfg)
        else:
            return self.pose_head.predict(example_sensor, preds, self.test_cfg)