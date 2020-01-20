from models.classifier import Classifier
from models.deeplab import build_deeplab
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Model(nn.Module):
	def __init__(self, args, nclass):
		super(Model, self).__init__()

		self.nclass = nclass

		if args.model == None: # Classification
			self.model = models.__dict__[args.bacbkone](pretrained=args.pretrained)
			final_feature = self.model.fc.in_features
			self.model.fc = nn.Linear(final_feature, args.dimension)
		elif 'deeplab' in args.model:
			self.model = build_deeplab(args)

		self.classifier = Classifier(args.dimension, nclass)

	def forward(self, input):
		x = self.model(input)
		x = self.classifier(x)
		return x