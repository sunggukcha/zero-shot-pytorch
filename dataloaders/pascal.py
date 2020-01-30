from __future__ import print_function, division

import numpy as np
import os
import platform

from PIL import Image
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

classes = {
    "seen": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "unseen": [16, 17, 18, 19, 20],
    "all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
}

def PathJoin(*args):
    if "Windows" in platform.platform():
        res = ""
        for arg in args:
            res += arg
        return res
    else:
        return os.path.join(*args)
        

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 csplit='seen',
                 verbose=True
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.args = args
        self.split = split
        self.csplit = csplit
        self.NUM_CLASSES = len(classes[csplit]) + 1

        self._base_dir = base_dir
        _splits_dir = PathJoin(self._base_dir, '/ImageSets', '/Segmentation')
        
        if 'aug' not in split:
            self._image_dir = PathJoin(self._base_dir, '/JPEGImages')
            self._label_dir = PathJoin(self._base_dir, '/SegmentationClass')
            self.aug = False
        else:
            self.aug = True



        self.images = []
        self.labels = []
        self.names = []
        '''
        classes = {
            'pascal' : ['aeroplane','bicycle','bird','boat',
                 'bottle','bus','car','cat',
                 'chair','cow','diningtable','dog',
                 'horse','motorbike','person','pottedplant',
                 'sheep','sofa','train','tvmonitor']
            }        
        forward = classes[args.dataset]
        self.reverse = {}
        
        for i in range(len(forward)):
            self.reverse[i+1] = forward[i]
        '''
        if 'Windows' in platform.platform(): split = "/" + split
        if self.aug:
            with open(PathJoin(_splits_dir, split + '.txt'), "r") as f:
                lines = f.read().splitlines()
            
            for line in lines:
                L1 = line.split(" ")[0]
                L2 = line.split(" ")[1]
                _image = PathJoin(self._base_dir, L1)
                _label = PathJoin(self._base_dir, L2)
                _name  = L2.split("/")[-1]
                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)
                self.names.append(_name)

        else:
            with open(PathJoin(_splits_dir, split + '.txt'), "r") as f:
                lines = f.read().splitlines()
            
            for line in lines:
                if 'Windows' in platform.platform(): line = '/' + line
                _image = PathJoin(self._image_dir, line + ".jpg")
                _label = PathJoin(self._label_dir, line + ".png")
                _name  = line + ".png"
                assert os.path.isfile(_image)
                assert os.path.isfile(_label)
                self.images.append(_image)
                self.labels.append(_label)
                self.names.append(_name)

        assert (len(self.images) == len(self.labels))

        # Display stats
        if verbose:
            print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target, _name = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'name': _name}

        if "train" in self.split:
            return self.transform_tr(sample)
        elif 'val' in self.split or 'test' in self.split:
            return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.labels[index])
        _name = self.names[index]

        return _img, _target, _name

    def transform_tr(self, sample):
        if self.csplit == 'all': ignores=[]
        elif self.csplit == 'seen': ignores=classes['unseen']
        else: raise RuntimeError("Training Unseen data is not legal.")
        composed_transforms = transforms.Compose([
            tr.MaskIgnores(ignores=ignores,mask=255),
            tr.RandomHorizontalFlip(),
#            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
#            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seg=True),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        if self.csplit == 'all':
            ignores=[]
            remap={}
        elif self.csplit == 'unseen':
            ignores=classes['seen']
            remap={16:1, 17:2, 18:3, 19:4, 20:5}
        elif self.csplit == 'seen':
            ignores=classes['unseen']
            remap={}
        else: raise RuntimeError("{} ???".format(self.csplit))
        composed_transforms = transforms.Compose([
#            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.MaskIgnores(ignores=ignores,mask=0),
            tr.ReMask(remap=remap),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), seg=True),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


