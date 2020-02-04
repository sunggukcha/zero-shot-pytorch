from dataloaders import make_data_loader
from models.model import Model
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.metric_cls import *
from utils.metric_seg import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.visualize import Visualize as Vs

import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_iou(pred, gt, n_classes=21, ignore0=False):
	# original source: https://github.com/jfzhang95/pytorch-deeplab-xception/issues/16
    total_miou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un
        iou = []
        for k in range(1 if ignore0 else 0, n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        miou = 0 if len(iou) == 0 else sum(iou) / len(iou)
        total_miou += miou

    return total_miou

class Tester(object):
	def __init__(self, args, verbose=True):
		self.args = args
		self.verbose = verbose

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclasses = make_data_loader(args, verbose)
	
		if self.args.task == 'segmentation':
			self.vs = Vs(args.dataset)
			self.evaluator = Evaluator(self.nclasses['val'])

		# Define Network
		model = Model(args, self.nclasses['train'], self.nclasses['test'])
				
		# Define Criterion
		criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

		self.model, self.criterion = model, criterion

		# Loading Classifier (SPNet style)
		if args.call is None or args.cseen is None or args.cunseen is None:
			raise NotImplementedError("Classifiers for 'all', 'seen', 'unseen' should be loaded")
		else:
			if args.test_set == 'unseen':
				ctest = args.cunseen
			elif args.test_set == 'all':
				ctest = args.call
			elif args.test_set == 'seen':
				ctest = args.cseen
			else:
				raise RuntimeError("{}".format(args.test_set))

			if not os.path.isfile(ctest):
				raise RuntimeError("=> no checkpoint for clasifier found at '{}'".format(ctest))

			self.model.load_test(ctest)

			if verbose: print("Classifiers checkpoint successfully loaded from {}, {}".format(args.cseen, ctest))

		# Resuming checkpoint
		self.best_pred = 0.0
		if args.resume is not None:
			if not os.path.isfile(args.resume):
				raise RuntimeError("{}: No such checkpoint exists".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']

			if args.cuda:
				pretrained_dict = checkpoint['state_dict']
				model_dict = {}
				state_dict = self.model.state_dict()
				for k, v in pretrained_dict.items():
					if 'classifier' in k: continue
					if k in state_dict:
						model_dict[k] = v
				state_dict.update(model_dict)
				self.model.load_state_dict(state_dict)
			else:
				print("Please use CUDA")
				raise NotImplementedError

			self.best_pred = checkpoint['best_pred']
			if verbose: print("Loading {} (epoch {}) successfully done".format(args.resume, checkpoint['epoch']))

		# Using CUDA
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			self.model = self.model.cuda()

		if args.ft:
			args.start_epoch = 0

	def test(self):
		self.model.eval()
		
		tbar = tqdm(self.test_loader)
		logits = [0.0] * self.nclasses['test']

		for i, sample in enumerate(tbar):
			images = sample['image'].cuda()
			names = sample['name']
			falses = torch.from_numpy(np.array([False] * images.shape[0])).cuda()

			with torch.no_grad():
				output = self.model(images, falses)

			preds_np = output.cpu().numpy()
			for i in range(preds_np.shape[1]):
				logits[i] = np.mean(preds_np[:, i, :, :])

			preds = torch.argmax(output, axis=1)

			if self.args.id:
				self.vs.predict_id(preds, names, self.args.save_dir)
			if self.args.color:
				self.vs.predict_color(preds, images, names, self.args.save_dir)
			#tbar.set_description("{} : {}".format(names[0], self.id2class(preds[0])))
		print(logits)
	
	def val(self):
		if self.args.task == 'classification':
			top1 = AverageMeter('Acc@1', ':6.2f')
			top5 = AverageMeter('Acc@5', ':6.2f')
		elif self.args.task == 'segmentation':
			self.evaluator.reset()

		if self.args.id or self.args.color or self.args.examine:
			if not os.path.exists(self.args.save_dir):
				os.makedirs(self.args.save_dir)

		self.model.eval()		
		tbar = tqdm(self.test_loader)
		miou = 0.0
		count = 0
		for i, sample in enumerate(tbar):
			images, targets, names = sample['image'].cuda(), sample['label'].cuda().long(), sample['name']
			falses = torch.from_numpy(np.array([False] * images.shape[0])).cuda()
			with torch.no_grad():
				outputs = self.model(images, falses)

			# Score record
			if self.args.task == 'classification':
				acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
				top1.update(acc1[0], images.size(0))
				top5.update(acc5[0], images.size(0))
			elif self.args.task == 'segmentation':
				preds = torch.argmax(outputs, axis=1)
				count += preds.shape[0]
				miou += get_iou(preds, targets, n_classes=self.nclasses['test'], ignore0=self.args.test_set == 'seen' or self.args.test_set == 'unseen')
				if self.args.id:
					self.vs.predict_id(preds, names, self.args.save_dir)
				if self.args.color:
					self.vs.predict_color(preds, images, names, self.args.save_dir)
				if self.args.examine:
					self.vs.predict_examine(preds, targets, images, names, self.args.save_dir)

		if self.args.task == 'classification':
			_top1 = top1.avg
			_top5 = top5.avg
			print("Top-1: %.3f, Top-5: %.3f" % (_top1, _top5))
		elif self.args.task == 'segmentation':
			'''
			acc = self.evaluator.Pixel_Accuracy()
			acc_class = self.evaluator.Pixel_Accuracy_Class()
			miou = self.evaluator.Mean_Intersection_over_Union()
			fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
			print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU:{}".format(acc, acc_class, miou, fwiou))
			'''
			print("confidence:{} mIoU:{}".format(self.args.confidence, miou / count))
			return miou / count