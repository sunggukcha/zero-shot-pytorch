from dataloaders import make_data_loader
from models.model import Model
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.metrics_cls import *
from utils.metrics_seg import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utiuls.visualize import Visualize as Vs

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models

class Tester(object):
	def __init__(self, args):
		self.args = args
		
		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
		# self.id2class = make_id2class(args)

		if self.args.task == 'segmentation':
			self.vs = Vs(args.dataset)
			self.evaluator = Evaluator(self.nlcass)

		# Define Network
		self.model = Model(args, self.nclass)
		self.criterion = nn.CrossEntropyLoss()

		# Using CUDA
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			self.model = self.model.cuda()

		# Resuming checkpoint
		self.best_pred = 0.0
		if args.resume is None:
			raise RuntimeError("Checkpoint for test is required")
		else:
			if not os.path.isfile(args.resume):
				raise RuntimeError("{}: No such checkpoint exists".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']

			if args.cuda:
				pretrained_dict = checkpoint['state_dict']
				model_dict = {}
				state_dict = self.model.module.state_dict()
				for k, v in pretrained_dict.items():
					if k in state_dict:
						model_dict[k] = v
				state_dict.update(model_dict)
				self.model.module.load_state_dict(state_dict)
			else:
				print("Please use CUDA")
				raise NotImplementedError

			if not args.ft:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.best_pred = checkpoint['best_pred']
			print("Loading {} (epoch {}) successfully done".format(args.resume, checkpoint['epoch']))

	def test(self):
		self.model.eval()
		
		tbar = tqdm(self.test_loader)
		for i, sample in enumerate(tbar):
			images = sample['image'].cuda()
			names = sample['name']

			with torch.no_grad():
				output = self.model(images)

			preds = torch.argmax(output, dim=1)

			if self.args.id:
				self.vs.predict_id(preds, names, self.args.save_dir)
			if self.args.color:
				self.vs.predict_color(preds, images, names, self.args.save_dir)
			#tbar.set_description("{} : {}".format(names[0], self.id2class(preds[0])))
	
	def val(self):
		self.model.eval()
		tbar = tqdm(self.val_loader)
