from models.backbone import resnet
import torch.nn as nn

def build_backbone(args):
    backbone = args.backbone
    output_stride = args.output_stride

    # normalization
    if 'bn' == args.norm:
        Norm = nn.BatchNorm2d
    elif 'gn' in args.norm:
        groups = int(args.norm[2:])
        Norm = lambda x: nn.GroupNorm(groups, x)
    else:
        raise NotImplementedError

    if 'resnet' in backbone:
        return resnet.build_ResNet (args, Norm)
    else:
        raise NotImplementedError
