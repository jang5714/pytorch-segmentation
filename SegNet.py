import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from math import ceil


def test():
    conv = nn.Conv2d(3,64,kernel_size=3, stride=2,padding=1)
    a = ''
    if isinstance(conv, nn.Conv2d):
        a = 'true'
        return a
    else:
        a = 'false'
        return a
    return a


def double_conv(input, output):
    conv = nn.Sequential(
        nn.Conv2d(input,output,kernel_size=3)
    )

class SegNet(nn.Module):
    def __init__(self,in_channels=3, pretrained=True, freeze_bn=False):
        super(SegNet, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained=pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        # vgg_bn 모델에서 Conv2D 슬라이싱
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)



    def forward(self):
        return ''



if __name__ == '__main__':
    test()