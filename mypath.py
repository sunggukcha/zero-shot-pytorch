'''
	Author:	Sungguk Cha
	eMail :	navinad@naver.com
'''

class Path(object):
	@staticmethod
	def db_root_dir(dataset):
		if dataset == 'pascal':
			return '../../datasets/VOCdevkit/VOC2012'
		else:
			print("Dataset {} is not available.".format(dataset))
			raise NotImplementedError
