from models.aspp import ASPP
from models.backbone import build_backbone

import torch
import torch.nn as nn
import torch.nn.functional as functional

class _DeepLab(nn.Module):
    def __init__(self, aspp, dec):
        super(_DeepLab, self).__init__()
        self.aspp = aspp
        self.dec = dec
        self.hasDec = self.dec is not None
    def forward(self, x, low_level_feat):
        x = self.aspp(x)
        if self.hasDec:
            x = self.dec(x, low_level_feat)
        return x

def build_deeplab(args):
    model = args.model
    aspp = ASPP(args)
    dec = None #Decoder(args, nclass)
    if model == 'deeplabv2':
        return _DeepLab(aspp, dec)
    #elif model == 'deeplabv3':
    #elif model == 'deeplabv3+':
    else:
        raise NotImplementedError

class DeepLab(nn.Module):
    def __init__(self, args):
        super(DeepLab, self).__init__()        
        self.deeplab = build_deeplab(args)

    def forward(self, x, low_level_feat):
        x = self.deeplab(x, low_level_feat)
        return x