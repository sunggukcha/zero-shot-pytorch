from dataloaders import pascal
from torch.utils.data import DataLoader

def make_data_loader(args, verbose=True):

	if args.dataset == 'pascal':
		train_set = pascal.VOCSegmentation(args, split='train_aug', csplit='seen', verbose=verbose)
		val_set = pascal.VOCSegmentation(args, split='val_aug', csplit=args.test_set, verbose=verbose)
		test_set = pascal.VOCSegmentation(args, split='test', csplit=args.test_set, verbose=verbose)

		train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
		val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
		test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
		
		return train_loader, val_loader, test_loader, {'train': train_set.NUM_CLASSES, 'val': val_set.NUM_CLASSES, 'test': test_set.NUM_CLASSES}
	else:
		print("Dataloader for {} is not implemented".format(args.dataset))
		raise NotImplementedError


