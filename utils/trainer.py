from dataloaders import make_data_loader
from models.model import Model
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.metric_cls import *
from utils.metric_seg import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.visualize import Visualize as Vs

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()

class Trainer(object):
	def __init__(self, args):
		self.args = args

		# Define saver
		self.saver = Saver(args)
		self.saver.save_experiment_config()

		# Define TensorBoard summary
		self.summary = TensorboardSummary(self.saver.experiment_dir)
		if not args.test:
			self.writer = self.summary.create_summary()

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclasses = make_data_loader(args, **kwargs)
	
		if self.args.task == 'segmentation':
			self.vs = Vs(args.dataset)
			self.evaluator = Evaluator(self.nclasses['val'])

		# Define Network
		model = Model(args, self.nclasses['train'], self.nclasses['val'])
		
		if args.model == None:
			train_params = [{'params': model.parameters(), 'lr': args.lr}]
		elif 'deeplab' in args.model:
			train_params = [{'params': model.backbone.parameters(), 'lr': args.lr},
							{'params': model.deeplab.parameters(), 'lr': args.lr * 10}]
		
		# Define Optimizer
		optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)
		criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)

		self.model, self.optimizer, self.criterion = model, optimizer, criterion

		# Define lr scheduler
		self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))


		# Loading Classifier (SPNet style)
		if args.call is None or args.cseen is None or args.cunseen is None:
			raise NotImplementedError("Classifiers for 'all', 'seen', 'unseen' should be loaded")
		else:
			if not os.path.isfile(args.cseen):
				raise RuntimeError("=> no checkpoint for clasifier found at '{}'".format(args.classifier))
			
			self.model.load_train(args.cseen)

			if args.test_set == 'unseen':
				ctest = args.cunseen
			else:
				ctest = args.call

			if not os.path.isfile(ctest):
				raise RuntimeError("=> no checkpoint for clasifier found at '{}'".format(ctest))

			print("Classifiers checkpoint successfully loaded from {}, {}".format(args.cseen, ctest))

		# Using CUDA
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			self.model = self.model.cuda()

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

		if args.ft:
			args.start_epoch = 0

	def train(self, epoch):
		train_loss = 0.0
		self.model.train()
		tbar = tqdm(self.train_loader)
		num_img_tr = len(self.train_loader)
		for i, sample in enumerate(tbar):
			image, target = sample['image'].cuda(), sample['label']
			if self.args.model == 'deeplabv2':
				target = resize_target(target, image.shape(2)).cuda()
			else:
				target = target.cuda().long()
			trues = torch.from_numpy(np.array([True] * image.shape[0])).cuda()
			self.scheduler(self.optimizer, i, epoch, self.best_pred)
			self.optimizer.zero_grad()
			output = self.model(image, trues)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			tbar.set_description('Train loss: %.3f' % (train_loss / (i+1)))
			self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
		self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
		print("[epoch: %d, loss: %.3f]" % (epoch, train_loss))

		if self.args.no_val:
			is_best = False
			self.saver.save_checkpoint(
				{'epoch' : epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,},
				is_best
				)

	def val(self, epoch):
		if self.args.task == 'classification':
			top1 = AverageMeter('Acc@1', ':6.2f')
			top5 = AverageMeter('Acc@5', ':6.2f')
		elif self.args.task == 'segmentation':
			self.evaluator.reset()

		self.model.eval()
		
		tbar = tqdm(self.val_loader, desc='\r')
		test_loss = 0.0
		for i, sample in enumerate(tbar):
			images, targets, names = sample['image'].cuda(), sample['label'].cuda().long(), sample['name']
			falses = torch.from_numpy(np.array([False] * images.shape[0])).cuda()
			with torch.no_grad():
				outputs = self.model(images, falses)
			if self.args.model == 'deeplabv2':
				outputs = F.interpolate(outputs, size=images.shape[2:], mode="bilinear", align_corners=False)
				outputs = F.softmax(outputs, dim=1)
				outputs = outputs.data.cpu().numpy()
				crf_outputs = np.zeros(outputs.shape)
				_images = images.data.cpu().numpy().astype(np.uint8)
				for i, (image, pred) in enumerate(zip(images, outputs)):
					image = image.transpose(1, 2, 0)
					crf_outputs[i] = dense_crf(image, pred)
				outputs = crf_outputs

			loss = self.criterion(outputs, targets)
			test_loss += loss.item()

			# Score record
			if self.args.task == 'classification':
				acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
				top1.update(acc1[0], images.size(0))
				top5.update(acc5[0], images.size(0))
			elif self.args.task == 'segmentation':
				preds = torch.argmax(outputs, axis=1)
				self.evaluator.add_batch(targets.cpu().numpy(), preds.cpu().numpy())
				if self.args.id:
					self.vs.predict_id(preds, names, self.args.save_dir)
				if self.args.color:
					self.vs.predict_color(preds, images, names, self.args.save_dir)
				if self.args.examine:
					self.vs.predict_examine(preds, targets, images, names, self.args.save_dir)

			tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))

		# Fast test during the training
		self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
		print('Loss: %.3f' % test_loss)
		
		if self.args.task == 'classification':
			_top1 = top1.avg
			_top5 = top5.avg
			self.writer.add_scalar('val/top1', _top1, epoch)
			self.writer.add_scalar('val/top5', _top5, epoch)
			print("Top-1: %.3f, Top-5: %.3f" % (_top1, _top5))
			new_score = _top1
		elif self.args.task == 'segmentation':
			acc = self.evaluator.Pixel_Accuracy()
			acc_class = self.evaluator.Pixel_Accuracy_Class()
			miou = self.evaluator.Mean_Intersection_over_Union()
			fwiou = self.evaluator.Frequency_Weighted_Intersection_over_Union()
			self.writer.add_scalar('val/Acc', acc, epoch)
			self.writer.add_scalar('val/Acc_class', acc_class, epoch)
			self.writer.add_scalar('val/mIoU', miou, epoch)
			self.writer.add_scalar('val/fwIoU', fwiou, epoch)
			print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU:{}".format(acc, acc_class, miou, fwiou))
			new_score = miou

		if new_score > self.best_pred:
			is_best = True
			self.best_pred = float(new_score)
			self.saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,
				}, is_best)
