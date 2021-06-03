import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from due.layers import spectral_norm_conv, spectral_norm_fc, SpectralBatchNorm2d

class ResNet18plus(torch.nn.Module):
    """
    Taken from https://gist.github.com/y0ast/d91d09565462125a1eb75acc65da1469
    """
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False, num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        out = self.output(x)
        return out


# The following classes have been taken and adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    Resnet modified to accept additional inputs to concatenate with feature maps before the final classification layers. 
    """
    def __init__(self, block, num_blocks, num_outputs=1, positive_output=True, num_additional_inputs=1):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512*block.expansion, 128)
        self.linear2 = nn.Linear(128 + num_additional_inputs, 64)
        self.linear3 = nn.Linear(64, num_outputs)
        self.positive_output = positive_output
        if self.positive_output:
            self.output_layer = nn.Softplus()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, feats):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = torch.cat([out, feats], axis=1).to(x.device)
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


def ResNet18(num_outputs=1, positive_output=True, num_additional_inputs=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_outputs, positive_output, num_additional_inputs)


def ResNet34(num_outputs=1, positive_output=True, num_additional_inputs=1):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_outputs, positive_output, num_additional_inputs)


def ResNet50(num_outputs=1, positive_output=True, num_additional_inputs=1):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_outputs, positive_output, num_additional_inputs)


def ResNet101(num_outputs=1, positive_output=True, num_additional_inputs=1):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_outputs, positive_output, num_additional_inputs)


def ResNet152(num_outputs=1, positive_output=True, num_additional_inputs=1):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_outputs, positive_output, num_additional_inputs)


def ResNet18Spec(input_size, spectral_normalization, n_power_iterations=1, batchnorm_momentum=0.9):
    return ResNet_DUE(SpecNormBasicBlock, [2, 2, 2, 2], input_size, spectral_normalization, num_outputs=None, n_power_iterations=n_power_iterations, batchnorm_momentum=batchnorm_momentum)


class SpecNormBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, wrapped_conv, wrapped_bn, input_size, in_planes, planes, stride=1):
        super(SpecNormBasicBlock, self).__init__()
        self.conv1 = wrapped_conv(input_size, 
            in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = wrapped_bn(planes)
        input_size = (input_size - 1) // stride + 1
        self.conv2 = wrapped_conv(input_size, planes, planes, kernel_size=3,
                               stride=1)
        self.bn2 = wrapped_bn(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                wrapped_conv(input_size, in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride),
                wrapped_bn(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_DUE(nn.Module):
    """
    Resnet modified to accept additional inputs to concatenate with feature maps before the final classification layers. 
    """
    def __init__(self, block, num_blocks, input_size, spectral_normalization, num_outputs=None, n_power_iterations=1, batchnorm_momentum=0.99, coeff=2):
        super(ResNet_DUE, self).__init__()
        self.in_planes = 64

        def wrapped_bn(num_features):
            if spectral_normalization:
                bn = SpectralBatchNorm2d(
                    num_features, coeff, momentum=batchnorm_momentum
                )
            else:
                bn = nn.BatchNorm2d(num_features, momentum=batchnorm_momentum)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv
        
        self.conv1 = wrapped_conv(input_size, 3, 64, kernel_size=3, stride=1)
        self.bn1 = wrapped_bn(64)
        self.layer1, input_size = self._make_layer(block, 64, num_blocks[0], stride=1, input_size=input_size)
        self.layer2, input_size = self._make_layer(block, 128, num_blocks[1], stride=2, input_size=input_size)
        self.layer3, input_size = self._make_layer(block, 256, num_blocks[2], stride=2, input_size=input_size)
        self.layer4, input_size = self._make_layer(block, 512, num_blocks[3], stride=2, input_size=input_size)
        self.num_classes = num_outputs
        # self.linear = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride, input_size):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.wrapped_conv, self.wrapped_bn, input_size, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            input_size = (input_size - 1) // stride + 1
        return nn.Sequential(*layers), input_size

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        # out = self.linear(out)
        # out = F.log_softmax(out, dim=1)

        return out
