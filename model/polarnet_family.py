import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
from torch.nn import init

from Config import PolarNetConfig as cfg

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class MKPM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.weight = nn.parameter.Parameter(torch.tensor([1., 1., 1., 1.], dtype=torch.float))

        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.relu = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        layer1 = F.upsample(self.conv1(self.pool1(x)), size=(h, w), mode='nearest')
        layer2 = F.upsample(self.conv2(self.pool2(x)), size=(h, w), mode='nearest')
        layer3 = F.upsample(self.conv3(self.pool3(x)), size=(h, w), mode='nearest')
        layer4 = F.upsample(self.conv4(self.pool4(x)), size=(h, w), mode='nearest')

        out = layer1 * self.weight[0] + layer2 * self.weight[1] + layer3 * self.weight[2] + layer4 * self.weight[3] + x
        out = self.relu(out)

        return out


class MKAC(nn.Module):
    def __init__(self, in_channels, out_channels=1):

        super().__init__()
        channel = in_channels

        self.weight = nn.parameter.Parameter(torch.tensor([1., 1., 1., 1.], dtype=torch.float))

        self.dilate1 = nn.Sequential(
            nn.Conv2d(channel, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU(inplace=False),
        )

        self.dilate2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0),
            nn.Conv2d(channel, out_channels, kernel_size=1, dilation=1, padding=0),
            nn.LeakyReLU(inplace=False),
        )

        self.dilate3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, out_channels, kernel_size=1, dilation=1, padding=0),
            nn.LeakyReLU(inplace=False),
        )

        self.dilate4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3),
            nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5),
            nn.Conv2d(channel, out_channels, kernel_size=1, dilation=1, padding=0),
            nn.LeakyReLU(inplace=False),
        )

        self.relu = nn.LeakyReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):

        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(x)
        dilate3_out = self.dilate3(x)
        dilate4_out = self.dilate4(x)

        out = dilate1_out * self.weight[0] + dilate2_out * self.weight[1] + dilate3_out * self.weight[
            2] + dilate4_out * self.weight[3]

        out = self.relu(out)

        return out


class PFEM(nn.Module):
    def __init__(self, in_channel, out_channel, dim=16):
        super().__init__()

        self.pac = MKAC(in_channels=in_channel, out_channels=1)
        self.pmp = MKPM(in_channels=in_channel, out_channels=dim - 1)

        self.cbam = ChannelGate(gate_channels=dim)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=dim, out_channels=out_channel, kernel_size=3, stride=2, padding=1),
        )

        self.weights_init_kaiming()

    def forward(self, x):

        x1 = self.pac(x)
        x2 = self.pmp(x)

        x = torch.cat((x1, x2), dim=1)
        x = self.cbam(x)

        x = self.conv(x)

        return x

    def weights_init_kaiming(self):
        classname = self.__class__.__name__
        if classname.find('Conv2d') != -1:
            init.kaiming_normal(self.weight.data)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max', 'lp']):
        super().__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class WeightCAM(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.feature = None
        self.gradient = None
        self.handlers = []
        self.handlers.append(module.register_forward_hook(self._get_features_hook))
        self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple
        """
        self.gradient = output_grad[0]

    def forward(self, net, target):
        net.eval()
        net.zero_grad()
        target.backward(retain_graph=True)
        # target.backward()
        gradient = self.gradient

        if gradient is None:
            return torch.zeros((1, 8, 3), dtype=torch.float).cuda()

        weight = torch.mean(gradient, dim=(2, 3))
        feature = self.feature
        weight = weight.unsqueeze(2).unsqueeze(2)
        cam = feature.cuda() * weight.cuda()
        cam = torch.sum(cam, dim=1)
        cam = torch.maximum(cam, torch.tensor([0.], dtype=torch.float).cuda())  # ReLU
        cam = F.adaptive_avg_pool2d(cam, (8, 3))
        net.zero_grad()
        return cam


class SubPolarNet(nn.Module):
    def __init__(self, block, layers, pretrain_dict_name: str, pretrained: bool = True, prior_k=None):

        super().__init__()
        self.prior_k = prior_k

        pfem_in_channel = 1
        self.inplanes = 64
        self.conv = PFEM(in_channel=pfem_in_channel, out_channel=64)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64 * 1, layers[0])
        self.layer2 = self._make_layer(block, 64 * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64 * 3, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64 * 4, layers[3], stride=2)

        self.weight_cam = WeightCAM(self.layer4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self.load_dict(pretrain_dict_name)

    def load_dict(self, pretrain_dict_name: str):
        checkpoint = model_zoo.load_url(model_urls[pretrain_dict_name])
        template_dict = self.state_dict()
        state_dict = {k: v for k, v in checkpoint.items() if k in template_dict and v.size() == template_dict[k].size()}
        self.load_state_dict(state_dict, False)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def get_weight_map(self, x):
        return self.weight_cam(self, x[0][1])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.prior_k is not None:
            self.prior_k = F.adaptive_avg_pool2d(self.prior_k.unsqueeze(0), (42, 7)).squeeze(0)
            self.prior_k = self.prior_k.to(x.device)
            x = x * self.prior_k
        return x


class PolarNet(nn.Module):
    def __init__(self, block, layers, pretrain_dict_name, pretrained, in_channel=3, num_classes=2):
        super().__init__()

        self.prior_k1 = cfg.prior_k[0, :, :] if cfg.use_pk is True else None
        self.prior_k2 = cfg.prior_k[1, :, :] if cfg.use_pk is True else None
        self.prior_k3 = cfg.prior_k[2, :, :] if cfg.use_pk is True else None

        self.conv1 = SubPolarNet(block, layers, pretrain_dict_name, pretrained, prior_k=self.prior_k1)
        self.conv2 = SubPolarNet(block, layers, pretrain_dict_name, pretrained, prior_k=self.prior_k2)
        self.conv3 = SubPolarNet(block, layers, pretrain_dict_name, pretrained, prior_k=self.prior_k3)

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(64 * 4 * block.expansion, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x, is_training: bool = False):

        x1 = self.conv1(x[0])
        x2 = self.conv2(x[1])
        x3 = self.conv3(x[2])

        if hasattr(self, "fc"):
            x = self.fc(x1 + x2 + x3)

        weight_cam1, weight_cam2, weight_cam3 = None, None, None
        if not is_training and cfg.is_cam:
            weight_cam1 = self.conv1.get_weight_map(x)
            weight_cam2 = self.conv2.get_weight_map(x)
            weight_cam3 = self.conv3.get_weight_map(x)

        return x, [weight_cam1, weight_cam2, weight_cam3]


def polarnet18(pretrained=False, **kwargs):
    pretrain_dict_name = "resnet18"
    model = PolarNet(BasicBlock, [2, 2, 2, 2], pretrain_dict_name, pretrained, **kwargs)
    return model


def polarnet34(pretrained=False, **kwargs):
    pretrain_dict_name = "resnet34"
    model = PolarNet(BasicBlock, [3, 4, 6, 3], pretrain_dict_name, pretrained, **kwargs)
    return model


def polarnet50(pretrained=False, **kwargs):
    pretrain_dict_name = "resnet50"
    model = PolarNet(Bottleneck, [3, 4, 6, 3], pretrain_dict_name, pretrained, **kwargs)
    return model


def polarnet101(pretrained=False, **kwargs):
    pretrain_dict_name = "resnet101"
    model = PolarNet(Bottleneck, [3, 4, 23, 3], pretrain_dict_name, pretrained, **kwargs)
    return model


def polarnet152(pretrained=False, **kwargs):
    pretrain_dict_name = "resnet152"
    model = PolarNet(Bottleneck, [3, 8, 36, 3], pretrain_dict_name, pretrained, **kwargs)
    return model
