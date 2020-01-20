r'''
	Author: Sungguk Cha
	eMail : sungguk@unist.ac.kr
	==========
	arguments
	1. pred		: tensor
		predicted tensor with shape [n, 3, 720, 1280], n * c * h * w
	2. origin	: tensor
		original image with shape [n, 3, 720, 1280, n * c * h * w
	3. target	: tensor
		target label with shape [n, 3, 720, 1280], n * c * h * w
	3. filename	: string
		file name. usually match the original name via dataloader. 
	4. save_dir	: string
		directory to save image. images are saved in save_dir + filename
		it should be full name. e.g., abcd1234-abcd1234.png

	NOW IT SUPPORTS GPU PROCESSING!
'''

from PIL import Image
import os.path
import numpy as np
import torch.nn.functional as F

def get_palette(dataset):
    if 'bdd' in dataset:
        return np.array([[0,0,0], [217, 83, 70], [91, 192, 222]], dtype=np.uint8)
    if 'pascal' in dataset:
        n = 21
        palette = [0] * (n*3)
        for j in range(0, n):
            lab = j
            palette[j*3+0] = 0
            palette[j*3+1] = 0
            palette[j*3+2] = 0
            i = 0
            while (lab > 0):
                palette[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                palette[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                palette[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
        res = np.array(palette)
        res = np.array([ [res[i], res[i+1], res[i+2]] for i in range(0, len(res), 3)], dtype=np.uint8)
        return res

class Visualize(object):
	def __init__(self, dataset):
		self.dataset = dataset
		self.palette = get_palette(dataset)
		self.bdd_res		= 720, 1280
		if 'bdd' in dataset:
			self.std	= np.array([0.229, 0.224, 0.225])
			self.mean	= np.array([0.485, 0.456, 0.406])
		elif 'pascal' in dataset:
			self.std		= np.array([0.197, 0.198, 0.201])
			self.mean		= np.array([0.279, 0.293, 0.290])

	def restore(self, images, mean=None, std=None):
		'''
			given tensor (images)
			return restored tensor (images)
		'''
		if mean == None:
			mean = self.mean
		if std	== None:
			std  = self.std
		images = images.float()
		if 'bdd' in self.dataset:
			images = images.permute(0, 1).permute(1, 2)
		elif 'pascal' in self.dataset:
			images = images.permute(0, 2)
		images = images * std + mean
		images = images * 255
		return images

	def predict_id(self, preds, names, save_dir='./'):
		'''
			given tensor (images)
			save images
		'''
		preds = preds.cpu().numpy()
		for i in range(len(preds)):
			pred	= preds[i]
			name	= names[i]
			saveas	= os.path.join(save_dir, name)
			result	= Image.fromarray(pred.astype('uint8'))
			result.save(saveas)
	
	def predict_color(self, preds, origins, names, save_dir='./'):
		preds = preds.cpu().numpy()
		origins = self.restore(origins).cpu().numpy()
		for i in range(len(preds)):
			pred	= preds[i]
			origin	= origins[i]
			name	= names[i]
			saveas	= os.path.join(save_dir, name)
			origin	= origins[i]
			img	= np.array(self.palette[pred.squeeze()])
			result	= np.array(np.zeros(origin.shape))
			#result[pred==0] = origin[pred==0]
			#result[pred!=0] = origin[pred!=0] /2 + img[pred!=0] /2
			result[pred!=0] = img[pred!=0]
			if self.dataset == 'pascal' or self.dataset == 'pascal_toy': result = result.swapaxes(0, 1)
			result	= Image.fromarray(result.astype('uint8'), 'RGB')
			result.save(saveas)

	def predict_examine(self, preds, targets, origins, names, save_dir='./'):
		targets = targets.cpu().numpy()
		preds = preds.cpu().numpy()
		origins = self.restore(origins)
		for i in range(len(preds)):
			pred	= preds[i]
			target	= targets[i]
			origin	= origins[i]
			name	= names[i]
			saveas	= os.path.join(save_dir, name)
			result	= np.array(np.zeros(pred.shape))
			result	= origin
			_correct	= np.multiply(pred!=0, pred==target)
			_wrong		= np.multiply(pred!=0, pred!=target)
			_notfound	= np.multiply(pred==0, pred!=target)
			result[_correct]	= result[_correct]/2 + np.array([7, 235, 126])/2
			result[_wrong]		= result[_wrong]/2 + np.array([242, 0, 0])/2
			result[_notfound]	= result[_notfound]/2 + np.array([226, 210, 16])/2
			result	= Image.fromarray(result.astype('uint8'), 'RGB')
			result.save(saveas)
