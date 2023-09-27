from torch import nn
from torch.nn import functional as F

from ..registry import READERS


@READERS.register_module
class RadarFeatureNet(nn.Module):
    def __init__(
        self, name="RadarFeatureNet"
    ):
        super(RadarFeatureNet, self).__init__()
        self.name = name

    def forward(self, rdr_cube):
        # todo: follow  nuscene to modify
        return rdr_cube
