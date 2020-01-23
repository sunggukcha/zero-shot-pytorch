from dataloaders import pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

	if args.dataset == 'pascal':
		train_set = pascal.VOCSegmentation(args, split='train_aug', csplit='seen')
		val_set = pascal.VOCSegmentation(args, split='val_aug', csplit=args.test_set)
		test_set = pascal.VOCSegmentation(args, split='test', csplit=args.test_set)

		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader, test_loader, {'train': train_set.NUM_CLASSES, 'val': val_set.NUM_CLASSES, 'test': test_set.NUM_CLASSES}
	else:
		print("Dataloader for {} is not implemented".format(args.dataset))
		raise NotImplementedError


