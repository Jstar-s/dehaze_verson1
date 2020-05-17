import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channel, rotio=16):
        super(ChannelAttention, self).__init__()
        self.avgPool = nn.AdaptiveAvgPool2d(1)
        self.maxPool = nn.AdaptiveMaxPool2d(1)
        self.shareMLP = nn.Sequential(
            nn.Conv2d(channel, channel//rotio, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//rotio, channel, kernel_size=1, stride=1, bias=False)
        )
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shareMLP(self.avgPool(x))
        maxout = self.shareMLP(self.maxPool(x))
        out = self.Sigmoid(avgout+maxout)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        if kernel_size == 7:
            padding =3
        else:
            padding = 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=2, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def main():
    x = torch.randn(5, 16, 256, 256)
    net = BasicBlock(16, 16)
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()


