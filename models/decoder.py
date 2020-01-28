import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        backbone = args.backbone

        if 'resnet' in backbone or backbone == 'drn' or backbone == 'ibn':
            low_level_inplanes = 256
        elif backbone == 'efficientnet-b7':
            low_level_inplanes = 48
        elif backbone == 'efficientnet-b6' or backbone == 'efficientnet-b5':
            low_level_inplanes = 40
        elif backbone == 'efficientnet-b4':
            low_level_inplanes = 32
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        if args.norm == 'bn':
            norm = nn.BatchNorm2d
        elif 'gn' in args.norm:
            n = int(args.norm[2:])
            norm = lambda x:nn.GroupNorm(n, x)
        else:
            raise NotImplementedError("Normalization {} is not supported".format(args.norm))

        dim = args.dimension

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = norm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(348, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                    norm(dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                    norm(dim),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
        )
                                    #nn.Conv2d(dim, dim, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat, shape):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = F.interpolate(x, size=shape[2:], mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()