import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import netron

def extract_layers(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class Inverted_Residual_Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Inverted_Residual_Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1else out
        return out


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


class CBAM(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAM, self).__init__()
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

#    在这个模块最后不要进行通道扩张，不然不好配置resnet的相加（好像不需要啊） 只要不在最后再把x加进来就行了
class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.block = Inverted_Residual_Block(in_planes, out_planes, expansion, stride)
        self.attention = CBAM(out_planes, out_planes)

    def forward(self, x):
        out = self.block(x)
        out = self.attention(out)
        return out


"""
cfgs = ([], [])
"""


class Groups(nn.Module):
    def __init__(self):
        super(Groups, self).__init__()
        self.layer1 = Block(64, 64, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.layer2 = Block(64, 64, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.layer3 = Block(64, 64, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.layer4 = Block(64, 64, 1, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.conv1(out)
        out = self.layer2(out)
        out = self.conv2(out)
        out = self.layer3(out)
        out = self.conv3(out)
        out = self.layer4(out)
        return out
#
# class BackBone(nn.Module):
#     def __init__(self):
#         super(BackBone, self).__init__()
#         self.group1 = Groups()
#         self.group2 = Groups()
#         self.group3 = Groups()
#
#     def forward(self, x):
#         out1= self.group1(x)
#         out2 = self.group2(out1)
#         out3 = self.group3(out2)
#
#

class Part2():
    pass


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


class FFT(nn.Module):
    def __init__(self,gps,blocks,):
        super(FFT, self).__init__()
        self.gps = gps
        self.blocks = blocks
        self.conv1 = extract_layers(3, 64, 3, False)
        self.group1 = Groups()
        self.group2 = Groups()
        self.group3 = Groups()
        self.attention = CBAM(192, 192)
        self.conv2 = extract_layers(192, 64, 3)
        self.conv3 = extract_layers(64, 3, 3)

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.group1(out)
        out2 = self.group2(out1)
        out3 = self.group3(out2)
        backbone = torch.cat([out1, out2, out3], dim=1)
        out = self.attention(backbone)
        out = self.conv2(out)
        out = self.conv3(out)
        return out




# def main():
#     inputs = torch.randn(1, 3, 128, 128)
#
#     net = Mynet()
#
#     """
#     问题出在注意力机制的rotio参数
#     """
#     output = net(inputs)
#     print(output.shape)
#     # torch.save(net, "test.pkl")
#     #
#     # onnx_path = "onnx_model_name.onnx"
#     # torch.onnx.export(net, inputs, onnx_path)
#     # netron.start(onnx_path)
#     num = get_parameter_number(net)
#     print(num)
#
# if __name__ == '__main__':
#     main()
#
#
# '''
# 目前参数量2M
# 问题是block堆叠的数量目前看起来还不够
# '''