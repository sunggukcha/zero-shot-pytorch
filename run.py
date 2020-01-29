r'''
    Author: Sungguk Cha
    eMail : navinad@naver.com

    Zero-shot classification & semantic segmentation implementation in PyTorch
'''

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
	
	parser.add_argument('--task', type=str, default='classification',
                            choices=['classification', 'segmentation'])
	parser.add_argument('--ignore-index', type=int, default=-100)
	parser.add_argument('--test-set', type=str, default='unseen',
						help='"unseen" for Zero-shot\n \
							"all" for generalized Zero-shot',
							choices=['unseen', 'all'])

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
	parser.add_argument('--test', default=False, action='store_true',
						help='True if test mode')
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--no-val', type=bool, default=False,
						help='True if train without validation')
	parser.add_argument('--time', default=False, action='store_true')

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

	print(args)
	torch.manual_seed(args.seed)
	if not args.test:
		trainer = Trainer(args)
		print("Starting epoch: {}".format(trainer.args.start_epoch))
		print("Total epochs: {}".format(trainer.args.epochs))
		if trainer.args.time: start = time.time()
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
			trainer.train(epoch)
			if not trainer.args.no_val:
				trainer.val(epoch)
		trainer.writer.close()
	else:
		tester = Tester(args)
		tester.test()
