import numpy as np
try:
    import spconv.pytorch as spconv 
    from spconv.pytorch import ops
    from spconv.pytorch import SparseConv3d, SubMConv3d, SparseInverseConv3d
    from spconv.pytorch import functional as Fsp
except: 
    import spconv 
    from spconv import ops
    from spconv import SparseConv3d, SubMConv3d

from torch import nn
from torch.nn import functional as F

from ..registry import BACKBONES
from ..utils import build_norm_layer

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = SparseConv3d(inplanes, planes, 3, stride=stride, padding=1, bias=bias, indice_key=indice_key+"_conv1")
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = SparseConv3d(planes, planes, 3, stride=1, padding=1, bias=bias, indice_key=indice_key+"_conv2")
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = Fsp.sparse_add(out, identity)
        out = replace_feature(out, self.relu(out.features))

        return out


@BACKBONES.register_module
class SpMiddleResNetFHD(nn.Module):
    def __init__(self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # [128, 256, 160]
        self.conv_input = spconv.SparseSequential(
            SparseConv3d(num_input_features, 16, 3, padding=1, bias=False, indice_key="conv_input"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(        
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="conv1_block"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="conv1_block2"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(16, 32, 3, 2, padding=1, bias=False, indice_key="conv2"),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="conv2_block"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="conv2_block2"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(32, 64, 3, 2, padding=1, bias=False, indice_key="conv3"),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="conv3_block"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="conv3_block2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, 2, padding=1, bias=False, indice_key="conv4"),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="conv4_block"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="conv4_block2"),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        sparse_shape = np.array(input_shape[::-1])
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)


        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return None, None
