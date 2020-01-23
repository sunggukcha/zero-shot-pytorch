from models.backbone import build_backbone
from models.classifier import Classifier
from models.deeplab import build_deeplab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
	def __init__(self, args, train_class, test_class):
		super(Model, self).__init__()

		if args.model == None: # Classification
			self.backbone = models.__dict__[args.bacbkone](pretrained=args.pretrained)
			final_feature = self.backbone.fc.in_features
			self.backbone.fc = nn.Linear(final_feature, args.dimension)
			self.deeplab = None
		elif 'deeplab' in args.model:
			self.backbone = build_backbone(args)
			self.deeplab = build_deeplab(args)

		self.classifier_train = Classifier(args.dimension, train_class)
		self.classifier_test = Classifier(args.dimension, test_class)

	def forward(self, input):
		x, low_level_feat = self.backbone(input)
		if self.deeplab is not None:
			x = self.deeplab(x, low_level_feat)
		x = self.classifier(x)
		return x