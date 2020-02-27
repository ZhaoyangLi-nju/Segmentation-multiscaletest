from __future__ import print_function, division
import os

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
from skimage import color
from torchvision import transforms
from data import custom_transforms as tr
from util.utils import color_label_np


class SBDSegmentation(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self,
                 cfg,
                 base_dir='/data/lzy/benchmark_RELEASE/',
                 split='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self.ms_targets = []
        self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        self._image_dir = os.path.join(self._dataset_dir, 'img')
        self._cat_dir = os.path.join(self._dataset_dir, 'cls')
        self.ignore_label=255
        self.class_weights=None
        self.split = split

        self.cfg = cfg
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split


        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ= os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):
        image, label = self._make_img_gt_point_pair(index)
        label_copy = label
        seg = None

        if self.cfg.NO_TRANS == False:
            if 'seg' == self.cfg.TARGET_MODAL:
                seg = Image.fromarray((color_label_np(label_copy, ignore=self.ignore_label).astype(np.uint8)), mode='RGB')
                # seg = np.load(seg_path)
            if 'lab' ==self.cfg.TARGET_MODAL:
                seg=color.rgb2lab(image)
            sample = {'image': image, 'label': label, 'seg': seg}
        else:
            # print(image.size)
            sample = {'image': image, 'label': label}
        for key in list(sample.keys()):
            if sample[key] is None:
                sample.pop(key)
        return self.transform(sample)

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.fromarray(scipy.io.loadmat(self.categories[index])["GTcls"][0]['Segmentation'][0])

        return _img, _target

    # def transform(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomScaleCrop(base_size=self.cfg.base_size, crop_size=self.cfg.crop_size),
    #         tr.RandomGaussianBlur(),
    #         tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])

    #     return composed_transforms(sample)
    def transform(self, sample):
        train_transforms = list()
        # train_transforms.append(tr.RandomScale(base_size=self.cfg.LOAD_SIZE, crop_size=self.cfg.FINE_SIZE))
        train_transforms.append(tr.Resize(self.cfg.LOAD_SIZE))
        train_transforms.append(tr.RandomScale(self.cfg.RANDOM_SCALE_SIZE))
        train_transforms.append(tr.RandomCrop(self.cfg.FINE_SIZE, pad_if_needed=True, fill=0))
        train_transforms.append(tr.RandomGaussianBlur())
        train_transforms.append(tr.RandomHorizontalFlip())

        if self.cfg.MULTI_SCALE:
            for item in self.cfg.MULTI_TARGETS:
                self.ms_targets.append(item)
            train_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        train_transforms.append(tr.ToTensor())
        train_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(train_transforms)
        return composed_transforms(sample)


