import fasttext
import os
import torch
import random
import numpy as np

from mypath import Path
from PIL import Image, ImageOps, ImageFilter

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        if 'label' in sample:
            mask = sample['label']
            mask = np.asarray(mask).astype(np.float32)
            mask = torch.from_numpy(mask).float()
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.), seg=False):
        self.mean = mean
        self.std = std
        self.seg = seg

    def __call__(self, sample):
        def norm(img):
            img = np.array(img).astype(np.float32)
            img /= 255.0
            img -= self.mean
            img /= self.std
            return img

        img = sample['image']
        img = norm(img)

        if 'label' in sample:
            label = sample['label']
            if self.seg:
                label = norm(label)

            return {'image': img, 'label': label, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class RandomHorizontalFlip(object):
    def __init__(self, seg=False):
        self.seg = seg
    def __call__(self, sample):
        img = sample['image']
        if 'label' in sample:
            mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.seg:
                if 'label' in sample:
                    mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if 'label' in sample:
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class MaskIgnores(object):
    def __init__(self, ignores, mask):
        self.ignores=ignores
        self.mask=mask
    def __call__(self, sample):
        label = sample['label']
        label = np.array(label).astype(np.int32)
        for ig in self.ignores:
            label[label == ig] = self.mask
        return {'image': sample['image'], 'label': label, 'name': sample['name']}
        
#
# 
#


class ToTensorE(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, ft, reverse):
        self.ft = ft
        self.reverse = reverse
        
    def get_embedding(self, target):
        if target == 0 or target == 255: return np.zeros(300)
        return self.ft[self.reverse[int(target)]]

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # embedding applied: H x W x dw
        # torch image: C X H X W
        img = sample['image']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        if 'label' in sample:
            mask = sample['label']
            mask = np.asarray(mask).astype(np.float32)
            neo_shape = mask.shape[0], mask.shape[1], 300
            masked_label = np.zeros(neo_shape)
            for i in range(masked_label.shape[0]):
                for j in range(masked_label.shape[1]):
                    masked_label[i, j, :] = self.get_embedding(mask[i, j])
            masked_label = masked_label.transpose((2, 0, 1))
            mask = torch.from_numpy(masked_label).float()
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        if 'label' in sample:
            return {'image': img, 'label': sample['label'], 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            if 'label' in sample:
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        if 'label' in sample:
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize((ow, oh), Image.NEAREST)
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            return {'image': img, 'label': mask, 'name': sample['name']}

        return {'image': img,
                'name' : sample['name']}

class RandomCrop2(object):
	'''
		Keeping BDD100k aspect ratio in training crops
		It takes great advantage in securing width-wise wide crop
	'''
	def __init__(self, crop_size):
		self.width	= crop_size
		self.height	= int(crop_size / 1.7777777777777777)
	def __call__(self, sample):
		img 	= sample['image']
		w, h	= img.size
		x	= random.randint(0, w - self.width)
		y	= random.randint(0, h - self.height)
		img	= img.crop( (x, y, x + self.width, y + self.height) )
		if 'label' in sample:
			label	= sample['label']
			label	= label.crop((x, y, x + self.width, y + self.height))
			return {'image': img, 'label': label, 'name': sample['name']}
		return {'image': img,
                	'name' : sample['name']}

'''
	Author: Sungguk Cha
	RandomCrop transform assumes 720p as input and 720 cropsize
	it randomly choose x value from [0, 1280-720) and return the cropped image
	it may reduce much much more time than FixScaleCrop above for the given assumption
'''

class RandomCrop(object):
	def __init__(self, crop_size):
		self.crop_size = crop_size

	def __call__(self, sample):
		img = sample['image']
		w, h = img.size
		x = max(0, random.randint(0, w - self.crop_size))
		y = max(0, random.randint(0, h - self.crop_size))
		img = img.crop( (x, y, x + self.crop_size, y + self.crop_size) )
		if 'label' in sample:
			label   = sample['label']
			label = label.crop( (x, 0, x + self.crop_size, 720) )
			return {'image': img, 'label': label, 'name': sample['name']}
		return {'image': img,
                        'name' : sample['name']}

		
class Rescale(object):
    def __init__(self, ratio):
        self.ratio = ratio
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        shape = ( int(img.size[0] * self.ratio), int(img.size[1] * self.ratio) )
        #print(img.size, shape)

        img = img.resize(shape, Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize(shape, Image.NEAREST)
            return {'image': img, 'label': mask, 'name': sample['name']}
        return {'image': img, 'name': sample['name']}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        if 'label' in sample:
            mask = sample['label']
            mask = mask.resize(self.size, Image.NEAREST)
            return {'image': img, 'label': mask, 'name': sample['name']}
        return {'image': img, 'name': sample['name']}
