import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, dimension, num_classes, confidence=0.5):
        super(Classifier, self).__init__()
        self.last_fc = nn.Conv2d(dimension, num_classes, kernel_size=1, stride=1)
        self._init_weight(confidence)
        
    def forward(self, x):
        return self.last_fc(x)
    
    def _init_weight(self, confidence):
        self.last_fc.weight.data.fill_(1)
        self.last_fc.bias.data.zero_()
        self.last_fc.weight.data[0] = 0.0
        self.last_fc.bias.data[0] = confidence