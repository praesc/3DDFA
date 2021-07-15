#!/usr/bin/env python3
# coding: utf-8

from __future__ import division

""" 
Creates a MobileNet Model as defined in:
Andrew G. Howard Menglong Zhu Bo Chen, et.al. (2017). 
MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. 
Copyright (c) Yang Lu, 2017

Modified By cleardusk
"""

import math
import torch
import torch.nn as nn

__all__ = ['mobilenet_2', 'mobilenet_1', 'mobilenet_075', 'mobilenet_05', 'mobilenet_025']


class DepthWiseBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes,
                                 bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)

        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)

        return out


class MobileNetV1(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=62, prelu=False, input_channel=3, seed: int = -1):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNetV1, self).__init__()


        block = DepthWiseBlock
        
        self.pilot = self._make_pilot(input_channel, widen_factor, prelu)

        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)

        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)

        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)

        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)

        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def _make_pilot(input_channel: int, widen_factor: float, prelu: bool) -> nn.Sequential:
        modules = []
        modules += [nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False)]
        modules += [nn.BatchNorm2d(int(32 * widen_factor))]
        if prelu:
            relu = nn.PReLU()
        else:
            relu = nn.ReLU(inplace=True)    
        modules += [relu]

        return nn.Sequential(*modules)

    def forward(self, x) -> torch.Tensor:
        x = self.pilot(x)

        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def mobilenetv1(widen_factor=1.0, num_classes=62):
    """
    Construct MobileNet.
    widen_factor=1.0  for mobilenet_1
    widen_factor=0.75 for mobilenet_075
    widen_factor=0.5  for mobilenet_05
    widen_factor=0.25 for mobilenet_025
    """
    model = MobileNetV1(widen_factor=widen_factor, num_classes=num_classes)
    return model


def mobilenetv1_2(num_classes=62, input_channel=3):
    model = MobileNetV1(widen_factor=2.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenetv1_1(num_classes=62, input_channel=3):
    model = MobileNetV1(widen_factor=1.0, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenetv1_075(num_classes=62, input_channel=3):
    model = MobileNetV1(widen_factor=0.75, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenetv1_05(num_classes=62, input_channel=3):
    model = MobileNetV1(widen_factor=0.5, num_classes=num_classes, input_channel=input_channel)
    return model


def mobilenetv1_025(num_classes=62, input_channel=3):
    model = MobileNetV1(widen_factor=0.25, num_classes=num_classes, input_channel=input_channel)
    return model
