import torchvision.datasets as datasets
from PIL import Image
from torchvision import transforms
import torch
import glob
import os
import cv2
from gaussian_blur import GaussianBlur
import numpy as np
from bisect import bisect_right

# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}",
                             milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted"
                             "got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]



class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class Melanoma_pair(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path = path
        # get all images paths
        self.transform = train_transform_2
        self.all_img_path = []
        self.all_img_path = sorted(glob.glob(os.path.join(self.path, "*.jpg")))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.all_img_path)

    def __getitem__(self, index):
        img_path = self.all_img_path[index]
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path)

        # img = Image.fromarray(img)
        # img = Image.fromarray(np.uint8(img))
        # img = Image.fromarray(img_path)
        pos_1 = self.transform(img)
        pos_2 = self.transform(img)

        return pos_1, pos_2, 0


class Melanoma_rotate_pair(torch.utils.data.Dataset):
    def __init__(self,path):
        self.path = path
        # get all images paths
        self.transform_1 = train_transform_1
        self.transform_2 = train_transform_2
        self.all_img_path = []
        self.all_img_path = sorted(glob.glob(os.path.join(self.path, "*.jpg")))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.all_img_path)

    def __getitem__(self, index):
        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        img_path = self.all_img_path[index]
        img = Image.open(img_path)

        img_r = img.rotate(rot_angle)

        pos_r = self.transform_1(img_r)
        pos_2 = self.transform_2(img)

        return pos_r, pos_2, rot_class


train_transform_2 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # GaussianBlur(kernel_size=int(0.1 * 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.62485343, 0.62214255, 0.620066], [0.17797484, 0.1801317, 0.18257397]),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

train_transform_1 = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    # GaussianBlur(kernel_size=int(0.1 * 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.62485343, 0.62214255, 0.620066], [0.17797484, 0.1801317, 0.18257397]),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
