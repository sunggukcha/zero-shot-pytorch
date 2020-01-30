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

		self.classifier_train = Classifier(args.dimension, train_class, confidence=args.confidence)
		self.classifier_test = Classifier(args.dimension, test_class, confidence=args.confidence)

	def forward(self, input, train=True):
		x, low_level_feat = self.backbone(input)
		if self.deeplab is not None:
			x = self.deeplab(x, low_level_feat, input.shape)
		if train:
			x = self.classifier_train(x)
		else:
			x = self.classifier_test(x)
		return x

	def load_train(self, cseen):
		self.classifier_train.load_state_dict(torch.load(cseen))

	def load_test(self, ctest):
		self.classifier_test.load_state_dict(torch.load(ctest))