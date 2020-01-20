import torch
import torch.nn as nn
import torch.nn.functional as F

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, Norm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.norm = Norm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, args): # backbone, output_stride, BatchNorm, abn=False, dec=True):
        super(ASPP, self).__init__()
        # dropout
        if args.model is not 'deeplabv2':
            self.dec = True
        else:
            self.dec = False
        # normalization
        if 'bn' == args.norm:
            Norm = nn.BatchNorm2d
        elif 'gn' in args.norm:
            groups = int(args.norm[2:])
            Norm = lambda x: nn.GroupNorm(groups, x)
        else:
            raise NotImplementedError
        # inplanes
        if 'resnet' in args.backbone or 'xception' == args.backbone:
            inplanes = 2048
        else:
            raise NotImplementedError
        # outplanes
        outplanes = args.dimension
        # output_stride
        output_stride = args.output_stride
        if output_stride == 32:
            dilations = [1, 3, 6, 9]
        elif output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, outplanes, 1, padding=0, dilation=dilations[0], Norm=Norm)
        self.aspp2 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[1], dilation=dilations[1], Norm=Norm)
        self.aspp3 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[2], dilation=dilations[2], Norm=Norm)
        self.aspp4 = _ASPPModule(inplanes, outplanes, 3, padding=dilations[3], dilation=dilations[3], Norm=Norm)

    
        self.gap1 = nn.AdaptiveAvgPool2d( (1,1) )
        self.gap2 = nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False)
        self.gap3 = Norm(outplanes)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                            nn.Conv2d(inplanes, outplanes, 1, stride=1, bias=False),
                                            Norm(outplanes),
                                            nn.ReLU())
        self.conv1 = nn.Conv2d(outplanes * 5, outplanes, 1, bias=False)
        self.conv2 = nn.Conv2d(outplanes, 3, 1, bias=False)
        self.bn1 = Norm(outplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.gap1(x)
        x5 = self.gap2(x5)
        if x.shape[0] > 1:
            x5 = self.gap3(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.dec:
            return self.dropout(x)
        return self.conv2(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()