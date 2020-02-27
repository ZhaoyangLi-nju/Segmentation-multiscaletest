
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .rec_sunrgbd import *

class Rec_MIT67(Rec_SUNRGBD):

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        sample = {'image': img, 'label': label}
        if self.split=='train':
            return self.transform_tr(sample)
        elif self.split=='val':
            return self.transform_val(sample)


        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
    def transform_tr(self, sample):
        train_transforms = list()
        if self.cfg.TASK_TYPE!='infomax':
            train_transforms.append(tr.RandomScale(self.cfg.RANDOM_SCALE_SIZE))
        train_transforms.append(tr.RandomRotate())
        train_transforms.append(tr.RandomCrop_Unaligned(self.cfg.FINE_SIZE, pad_if_needed=True, fill=0))
        train_transforms.append(tr.RandomHorizontalFlip())
        if self.cfg.TARGET_MODAL == 'lab':
            train_transforms.append(tr.RGB2Lab())
        if self.cfg.MULTI_SCALE:
            for item in self.cfg.MULTI_TARGETS:
                self.ms_targets.append(item)
            train_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        train_transforms.append(tr.ToTensor())
        train_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(train_transforms)
        return composed_transforms(sample)

    def transform_val(self, sample):
        val_transforms = list()
        val_transforms.append(tr.Resize(self.cfg.LOAD_SIZE))
        if self.cfg.MULTI_SCALE:
            val_transforms.append(tr.MultiScale(size=self.cfg.FINE_SIZE,scale_times=self.cfg.MULTI_SCALE_NUM, ms_targets=self.ms_targets))
        val_transforms.append(tr.ToTensor())
        val_transforms.append(tr.Normalize(mean=self.cfg.MEAN, std=self.cfg.STD, ms_targets=self.ms_targets))
        composed_transforms = transforms.Compose(val_transforms)