import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, dimension, num_classes):
        super(Classifier, self).__init__()
        self.last_fc = nn.Conv2d(dimension, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        return self.last_fc(x)
