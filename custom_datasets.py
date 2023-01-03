from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy
import torch
import os
import pandas as pd

from utils import *
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar10_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
           transforms.Normalize(mean=[0.5, 0.5, 0.5,],
                                std=[0.5, 0.5, 0.5]),
       ])

def augmentations_light():
    return transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop((96, 96)),
    transforms.ToTensor(),
    #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def augmentations_medium():
    return transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomResizedCrop((96, 96)),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    ]
)

class CIFAR10(Dataset):
    def __init__(self, path):
        self.cifar10 = datasets.CIFAR10(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar10[index]

        return data, target, index

    def __len__(self):
        return len(self.cifar10)


class CIFAR100(Dataset):
    def __init__(self, path):
        self.cifar100 = datasets.CIFAR100(root=path,
                                        download=True,
                                        train=True,
                                        transform=cifar10_transformer())

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)

        data, target = self.cifar100[index]

        # Your transformations here (or set it in CIFAR10)

        return data, target, index

    def __len__(self):
        return len(self.cifar100)


class ImageNet(Dataset):
    def __init__(self, path):
        self.imagenet = datasets.ImageFolder(root=path, transform=imagenet_transformer)

    def __getitem__(self, index):
        if isinstance(index, numpy.float64):
            index = index.astype(numpy.int64)
        data, target = self.imagenet[index]

        return data, target, index

    def __len__(self):
        return len(self.imagenet)

class BoundingBoxImageLoader(Dataset):
    """Animal Bounding Box Crop."""

    def __init__(self, pickle_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_pickle(pickle_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 0]+'.JPG')
        # try:
        # image = read_image(img_name)
        image = Image.open(img_name)
    
        bbox_im = self.dataframe.iloc[idx, 1]
        image_croped = transforms.functional.crop(
            image, int(bbox_im[1]), int(bbox_im[0]), int(bbox_im[3]), int(bbox_im[2])) # top, left, height, width
        
        sample = image_croped
        target = self.dataframe.iloc[idx, 2]
        
        if self.transform:
            sample = self.transform(sample)

        return sample, target, idx
        
        # except:
        #     print(img_name)
        #     return 0,0,0