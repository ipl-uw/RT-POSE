import torch
import torch.nn as nn

# TODO: implememnt a conv 2d + conv 1d module

class Conv2plus1D(nn.Module):
    r"""" testver
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True):
        super(Conv2plus1D, self).__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv2d_conv = nn.Conv3d(in_channels, in_channels, kernel_size=(1, kernel_size, kernel_size), 
                                      stride=(1, stride, stride), padding=(0, padding, padding), groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1d_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(kernel_size, 1, 1), 
                                       stride=(stride, 1, 1), padding=(padding, 0, 0), bias=bias)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(
                1, 16, kernel_size=3, stride=1, padding=1, bias=False
            )
    def forward(self, x):
        x = self.conv2d_conv(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.conv1d_conv(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.conv1(x)
        return x

if __name__ == '__main__':

    # test shape
    in_channels = 1
    out_channels = 1
    kernel_size = 3
    conv_2d_1d = Conv2plus1D(in_channels,out_channels,kernel_size)
    dummy_radar = torch.rand(in_channels ,out_channels, 24, 32, 176)
    output = conv_2d_1d(dummy_radar)
    print(output.shape)

