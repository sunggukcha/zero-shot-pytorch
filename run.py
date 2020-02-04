r'''
    Author: Sungguk Cha
    eMail : navinad@naver.com

    Zero-shot classification & semantic segmentation implementation in PyTorch
'''

import random
from utils.tester import Tester
from utils.trainer import Trainer

import argparse
import os
import time
import torch
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_args():
	parser = argparse.ArgumentParser()

	# etc
	parser.add_argument('--verbose', default=True, action='store_false')

	# zeroshot params	
	parser.add_argument('--task', type=str, default='classification',
                            choices=['classification', 'segmentation'])
	parser.add_argument('--ignore-index', type=int, default=-100)
	parser.add_argument('--test-set', type=str, default='unseen',
						help='"unseen" for Zero-shot\n \
							"all" for generalized Zero-shot',
							choices=['unseen', 'all', 'seen'])

	# external params
	parser.add_argument('--seed', type=int, default=1,
						help='Random seed')
	parser.add_argument('--checkname', type=str, default=None,
						help='Checkpoint name')
		# GPU
	parser.add_argument('--cuda', type=bool, default=True,
						help='CUDA usage')
	parser.add_argument('--gpu-ids', type=str, default='0',
						help='E.g., 0 | 0,1,2,3 ')
		# CPU
	parser.add_argument('--workers', type=int, default=0,
						help='Number of workers for dataloader')

	# training options
	parser.add_argument('--dataset', type=str, default='pascal')
	parser.add_argument('--backbone', type=str, default='resnet18')
	parser.add_argument('--model', type=str, default=None)
	parser.add_argument('--norm', type=str, default='bn')
	parser.add_argument('--output-stride', type=int, default=16)
	parser.add_argument('--dimension', type=int, default=300)

	parser.add_argument('--pretrained', default=True, action='store_false',
						help='True if load pretrained model')
	parser.add_argument('--ft', type=bool, default=None,
						help='True if finetune')
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--no-val', default=False, action='store_true',
						help='True if train without validation')

	# test options
	parser.add_argument('--test', default=False, action='store_true',
						help='True if test mode')
	parser.add_argument('--test-val', default=False, action='store_true',
						help='True if test-validation mode')
	parser.add_argument('--save-dir', type=str, default=None,
						help='save directory')
	parser.add_argument('--confidence', type=float, default=0.5)

	# loading checkpoints
	parser.add_argument('--call', type=str, default=None)
	parser.add_argument('--cseen', type=str, default=None)
	parser.add_argument('--cunseen', type=str, default=None)
	parser.add_argument('--resume', type=str, default=None)
	
	# hyper params
	parser.add_argument('--lr', type=float, default=None,
						help='Initial learning rate')
	parser.add_argument('--lr-scheduler', type=str, default='poly',
						choices=['poly', 'step', 'cos'],
						help='lr scheduler mode: (default: poly)')
	parser.add_argument('--epochs', type=int, default=None,
						help='Training epochs')
	parser.add_argument('--batch-size', type=int, default=1,
						help='Batch size')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='Momentum')
	parser.add_argument('--weight-decay', type=float, default=1e-4,
						help='Weight decay')

	# inference
	parser.add_argument('--id', default=False, action='store_true')
	parser.add_argument('--color', default=False, action='store_true')
	parser.add_argument('--examine', default=False, action='store_true')

	return parser.parse_args()

if __name__ == "__main__":
	args = get_args()
	args.cuda = torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
	
	if args.checkname == None:
		if args.model is not None:
			args.checkname = args.model + "-" + args.backbone
		else:
			args.checkname = args.backbone

	if args.verbose: print(args)
	torch.manual_seed(args.seed)
	if not args.test:
		trainer = Trainer(args)
		print("Starting epoch: {}".format(trainer.args.start_epoch))
		print("Total epochs: {}".format(trainer.args.epochs))
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
			trainer.train(epoch)
			if not trainer.args.no_val:
				trainer.val(epoch)
		trainer.writer.close()
	else:
		if args.test_val:
			tester = Tester(args, verbose=False)
			tester.val()
			exit()
			
			# RANDOM search mode
			count = 100
			between = [-10.0, 0.0]
			score = 0.0
			mxc = 0

			while count > 0:
				args.confidence = random.uniform(between[0], between[1])
				tester = Tester(args, verbose=False)
				new_score = tester.val()
				if new_score > 0.0:
					print("\n\n\n\n\n{}: {}\n\n\n\n\n".format(args.confidence, new_score))
				if new_score > score:
					score = new_score
					mxc = args.confidence
				else:
					count -= 1
			print(score, mxc)
			exit()
			'''

			# exponential search mode
			count = 100
			args.confidence = 0.1
			tester = Tester(args, verbose=False)
			score = tester.val()

			while count > 0:
				args.confidence /= 10
				tester = Tester(args, verbose=False)
				new_score = tester.val()
				if new_score > 0.0:
					print("\n\n\n\n\n{}: {}\n\n\n\n\n".format(args.confidence, new_score))
				if new_score > score:
					score = new_score
				else:
					count -= 1
			exit()
			'''
			# binary search mode
			count = 100
			between = [-0.5, 0]

			args.confidence = between[0]
			tester = Tester(args, verbose=False)
			cl = between[0]
			confidence_low = tester.val()

			args.confidence = between[1]
			tester = Tester(args, verbose=False)
			ch = between[1]
			confidence_high = tester.val()

			mx = max(confidence_high, confidence_low)
			mxc = cl if confidence_low > confidence_high else ch
			while True:
				cf = (cl + ch)/2
				args.confidence = cf
				tester = Tester(args, verbose=False)
				if confidence_high > confidence_low:
					cl = cf
					confidence_low = tester.val()
				else:
					ch = cf
					confidence_high = tester.val()

				neo_mx = max(confidence_high, confidence_low)
				if mx < neo_mx:
					mx = neo_mx
					mxc = cl if confidence_low > confidence_high else ch
				else:
					if count > 0:
						count -= 1
					else:
						break

			print(mx, mxc)

		else:
			tester = Tester(args)
			tester.test()
